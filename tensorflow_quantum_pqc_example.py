# this example trains a one-qubit classifier to distinguish \ket{+} from \ket{-}.
# cirq is used to prepare the two input circuits, tensorflow Quantum’s PQC layer appends a trainable R_y(\theta) gate and returns the simulated expectation.
# tensorflow’s GradientTape uses TFQ’s parameter-shift rule to differentiate the loss, and SGD update \theta.
# see https://www.tensorflow.org/quantum/api_docs/python/tfq/layers/PQC
# fwiw: classes of parametrized quantum circuits have efficient classical surrogates, see for example, Bermejo, Pablo, et al. "Quantum convolutional neural networks are effectively classically simulable." Prx Quantum 7.2 (2026): 020304 

import os


# tensorflow Quantum uses legacy tf.keras
# the versioning is finicky. python 3.12 with: "numpy==2.0.2" "tensorflow==2.18.1" "tf-keras==2.18.0" "tensorflow-quantum==0.7.6" works
os.environ["TF_USE_LEGACY_KERAS"] = "1" 

import cirq
import sympy
import tensorflow as tf
import tensorflow_quantum as tfq


def make_inputs(qubit):
    """Prepare |+> = H|0> and |-> = H X|0>, in that order."""
    circuits = [
        cirq.Circuit(cirq.H(qubit)),
        cirq.Circuit(cirq.X(qubit), cirq.H(qubit)),
    ]
    # TFQ passes serialized Cirq circuits to its tensorflow operations.
    inputs = tfq.convert_to_tensor(circuits)
    labels = tf.constant([[1.0], [-1.0]], dtype=tf.float32)
    return inputs, labels


def make_model(qubit):
    theta = sympy.Symbol("theta")
    model_circuit = cirq.Circuit(cirq.ry(theta)(qubit))
    return tfq.layers.PQC(
        model_circuit,
        operators=[cirq.Z(qubit)],
        differentiator=tfq.differentiators.ParameterShift(),
        initializer=tf.keras.initializers.Constant(0.2),
    )


def train_step(model, optimizer, inputs, labels):
    with tf.GradientTape() as tape:
        # shape (2, 1): one expectation value in [-1, 1] for each circuit.
        scores = model(inputs)
        # zero when each score agrees with its target at +1 or -1.
        loss = 1.0 - tf.reduce_mean(labels * scores)
    params = model.trainable_variables
    gradients = tape.gradient(loss, params)
    optimizer.apply_gradients(zip(gradients, params))
    return float(loss.numpy())


def show_scores(stage, model, inputs):
    plus, minus = model(inputs).numpy().ravel()
    theta = float(model.trainable_variables[0].numpy()[0])
    print(
        f"{stage:>6} | theta = {theta:+.4f} rad"
        f" | <Z> for |+> = {plus:+.4f} | <Z> for |-> = {minus:+.4f}"
    )


def main():
    qubit = cirq.GridQubit(0, 0)
    inputs, labels = make_inputs(qubit)
    model = make_model(qubit)
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.2)

    # for these inputs, the scores are -sin(theta) and +sin(theta).
    # and the loss is 1 + sin(theta), minimized near theta = -pi/2.
    show_scores("before", model, inputs)
    for step in range(1, 41):
        loss = train_step(model, optimizer, inputs, labels)
        if step == 1 or step % 10 == 0:
            print(f"step {step:2d} | loss before update = {loss:.6f}")
    show_scores("after", model, inputs)


if __name__ == "__main__":
    main()
