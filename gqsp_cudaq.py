"""

Generalized QSP for P(U), with a CUDA-Q statevector.
The FFT phase synthesis for low-to-moderate degrees and strictly bounded polynomials.
"""

import argparse

import cudaq
import numpy as np
from numpy.polynomial.polynomial import polyval


def _coefficients(values):
    coeffs = np.asarray(values, dtype=np.complex128)
    if coeffs.ndim != 1 or coeffs.size == 0 or not np.all(np.isfinite(coeffs)):
        raise ValueError("Expected a nonempty vector of finite coefficients.")
    last = np.flatnonzero(coeffs)
    return coeffs[:last[-1] + 1].copy() if last.size else coeffs[:1].copy()


def _complement_error(p, q):
    """Coefficient l1 bound on max_{|z|=1} ||P|² + |Q|² - 1|."""
    residual = np.correlate(p, p, "full") + np.correlate(q, q, "full")
    residual[len(p) - 1] -= 1
    return float(np.sum(np.abs(residual)))


def _fft_complement(p, tolerance):
    # On the circle, real(log Q) = log(1 - |P|²)/2 for the outer factor.
    # Keep the positive frequencies, with half weight on the constant term.
    size = 1 << (max(256, 8 * len(p)) - 1).bit_length()
    max_size = max(size, 1 << 18)
    while size <= max_size:
        samples = np.fft.ifft(p, n=size) * size
        abs_p = np.abs(samples)
        if np.any(abs_p >= 1):
            raise ValueError(
                "FFT synthesis requires |P| < 1 on the unit circle. "
                "Scale P or supply a complementary polynomial."
            )
        log_gap = np.log1p(-abs_p**2)
        spectrum = np.fft.fft(log_gap)
        spectrum[0] *= 0.5
        spectrum[size // 2:] = 0
        outer_samples = np.exp(np.fft.ifft(spectrum))
        q = np.fft.fft(outer_samples)[:len(p)] / size
        if _complement_error(p, q) <= tolerance:
            return q
        size *= 2
    raise ValueError("FFT complementary factor did not reach the requested tolerance.")


def _zyz(a, b):
    """Gate order for an SU(2) matrix with first column (a, b)."""
    arg_a, arg_b = np.angle(a), np.angle(b)
    return (-arg_a - arg_b, 2 * np.arctan2(abs(b), abs(a)), arg_b - arg_a)


def _first_column(angles):
    first_z, y, last_z = angles
    a = np.exp(-0.5j * (first_z + last_z)) * np.cos(y / 2)
    b = np.exp(0.5j * (last_z - first_z)) * np.sin(y / 2)
    return a, b


def synthesize_gqsp(coefficients, complement=None, *, tolerance=1e-9):
    """Return a flat list of Rz, Ry, Rz angles in circuit order.

    Pass complement=... to supply Q with |P(z)|² + |Q(z)|² = 1 on |z|=1.
    The returned circuit polynomial is checked in coefficient l1 norm.
    """
    if not np.isfinite(tolerance) or not 0 < tolerance < 1:
        raise ValueError("tolerance must be finite and between 0 and 1.")
    p = _coefficients(coefficients)
    if np.max(np.abs(p)) > 1 + tolerance:
        raise ValueError("A bounded polynomial cannot have a coefficient of modulus > 1.")

    if complement is not None:
        q = _coefficients(complement)
        if np.max(np.abs(q)) > 1 + tolerance:
            raise ValueError("A complementary polynomial cannot have a coefficient of modulus > 1.")
        width = max(len(p), len(q))
        p = np.pad(p, (0, width - len(p)))
        q = np.pad(q, (0, width - len(q)))
    elif np.count_nonzero(p) <= 1:
        magnitude = abs(p[-1])
        q = np.zeros_like(p)
        q[0] = np.sqrt(max(0, (1 - magnitude) * (1 + magnitude)))
    else:
        q = _fft_complement(p, tolerance / 10)

    if _complement_error(p, q) > tolerance:
        raise ValueError("P and Q do not satisfy the complementary identity to tolerance.")

    # F = (P,Q)^T. Its leading and constant coefficients are orthogonal.
    # Use the larger vector to align them with |0> and |1>, respectively.
    # Undo diag(z,1): the first row loses its constant term and shifts down.
    f, g = p.copy(), q.copy()
    reversed_rotations = []
    for _ in range(len(p) - 1):
        leading_norm = np.hypot(abs(f[-1]), abs(g[-1]))
        constant_norm = np.hypot(abs(f[0]), abs(g[0]))
        if constant_norm > leading_norm:
            a = g[0].conjugate() / constant_norm
            b = -f[0].conjugate() / constant_norm
        elif leading_norm > 0:
            a, b = f[-1] / leading_norm, g[-1] / leading_norm
        else:
            raise ValueError("A layer has vanished at working precision.")
        reversed_rotations.append(_zyz(a, b))
        f, g = (a.conjugate() * f + b.conjugate() * g)[1:], (-b * f + a * g)[:-1]
    norm = np.hypot(abs(f[0]), abs(g[0]))
    #if norm == 0:
    #    raise ValueError("The final rotation has vanished at working precision.")
    
    assert np.abs(norm) > tolerance
    reversed_rotations.append(_zyz(f[0] / norm, g[0] / norm))
    rotations = reversed_rotations[::-1]

    # Reconstruct the polynomial using the actual gate angles. The triangle
    # inequality bounds the uniform error on the circle and the error for U.
    a, b = _first_column(rotations[0])
    f, g = np.array([a]), np.array([b])
    for rotation in rotations[1:]:
        a, b = _first_column(rotation)
        f, g = np.pad(f, (1, 0)), np.pad(g, (0, 1))
        f, g = a * f - b.conjugate() * g, b * f + a.conjugate() * g
    if max(np.sum(np.abs(f - p)), np.sum(np.abs(g - q))) > tolerance:
        raise ValueError("Rotation synthesis did not reach the requested tolerance.")
    return [float(angle) for rotation in rotations for angle in rotation]


def append_gqsp(circuit, signal, data, ancilla, angles, *signal_arguments):
    """append GQSP to a CUDA-Q builder circuit.

    signal is a controllable CUDA-Q kernel with signature
    signal(data, *signal_arguments). The data register and ancilla are
    allocated by the caller. Initialize ancilla to |0> for the P(U) block.
    any global phase of U must be preserved by the controlled signal.
    """
    if len(angles) < 3 or len(angles) % 3:
        raise ValueError("Expected one or more Rz, Ry, Rz angle triples.")
    for offset in range(0, len(angles), 3):
        if offset:
            circuit.x(ancilla)
            circuit.control(signal, ancilla, data, *signal_arguments)
            circuit.x(ancilla)
        circuit.rz(angles[offset], ancilla)
        circuit.ry(angles[offset + 1], ancilla)
        circuit.rz(angles[offset + 2], ancilla)


@cudaq.kernel
def rx_signal(data: cudaq.qview, alpha: float):
    rx(alpha, data[0])


def gqsp_circuit(angles, column):
    circuit, alpha = cudaq.make_kernel(float)
    ancilla = circuit.qalloc()
    data = circuit.qalloc(1)
    if column == 1:
        circuit.x(data[0])
    append_gqsp(circuit, rx_signal, data, ancilla, angles, alpha)
    return circuit


def single_qubit_unitary_example():
   
    target = "qpp-cpu" 
    tolerance = 1e-9

    cudaq.set_target(target)

    # mixed even/odd powers and genuinely complex coefficients.
    p = [0.20, 0.25j, -0.15, 0.10 - 0.05j] # coefficients are in ascending monomial order. 
    angles = synthesize_gqsp(p, tolerance=tolerance)
    alpha = 0.83

    # basis labels follow allocation order: (ancilla, data).
    # read the unnormalized amplitudes after projection onto ancilla |0>.
    columns = []
    for column in range(2):
        gqsp = gqsp_circuit(angles, column)
        state = cudaq.get_state(gqsp, alpha)
        columns.append(state.amplitudes(["00", "01"]))
    observed = np.column_stack(columns)

    print(cudaq.draw(gqsp_circuit(angles, 0), alpha))

    # R_x(alpha) = H diag(exp(-i alpha/2), exp(i alpha/2)) H.
    hadamard = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    eigenvalues = np.exp(0.5j * alpha * np.array([-1, 1]))
    expected = hadamard @ np.diag(polyval(eigenvalues, p)) @ hadamard
    error = np.max(np.abs(observed - expected))
    print(f"# controlled-U calls: {len(angles) // 3 - 1}")
    print("# CUDA-Q ancilla-|0> block:")
    print(observed)
    print(f"# max entrywise error vs P(U): {error:.3e}")

    assert np.isfinite(error) and np.abs(error) < tolerance, "fail:CUDA-Q block \\approx P(U) beyond tolerance."



def multiqubit_unitary_example():
    @cudaq.kernel
    def cyclic_shift(data: cudaq.qview):
        swap(data[0], data[1])
        swap(data[1], data[2])
        swap(data[2], data[3])


    cudaq.set_target("qpp-cpu")

    # P(z) = 0.20 + 0.25i z - 0.15 z² + (0.10 - 0.05i) z³
    p = [0.20, 0.25j, -0.15, 0.10 - 0.05j]
    angles = synthesize_gqsp(p)

    circuit = cudaq.make_kernel()
    ancilla = circuit.qalloc()
    data = circuit.qalloc(4)
    circuit.x(data[0])  # prepare |1000> on the data register.

    append_gqsp(circuit, cyclic_shift, data, ancilla, angles)

    state = cudaq.get_state(circuit)

    # the first bit selects ancilla |0>. the other four label the data.
    for bits in ["1000", "0001", "0010", "0100"]:
        print(bits, state.amplitude("0" + bits))

    # verify all 16 amplitudes in the ancilla-|0> branch.
    labels = ["0" + format(j, "04b") for j in range(16)]
    expected = np.zeros(16, dtype=complex)
    expected[[8, 1, 2, 4]] = p
    np.testing.assert_allclose(
        state.amplitudes(labels), expected, atol=1e-12, rtol=0
    )

    print(cudaq.draw(circuit))



if __name__ == "__main__":
    print("example: single-qubit unitary")
    single_qubit_unitary_example()
    print("example: multi-qubit unitary")
    multiqubit_unitary_example()