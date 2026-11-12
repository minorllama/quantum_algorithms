# Quantum algorithms, error correction and open quantum systems. 

This repository contains quantum algorithms implemented with `cudaq` and `qiskit`, some experiments with quantum error correction and open quantum systems. 


### Using jupyter notebooks with uv 

To make sure `vscode` can discover a kernel available through `uv` `.venv`, activate the virtual environment and then add and register the kernel:
```bash
    # after activating the uv virtual env
    uv pip install ipykernel
    python -m ipykernel install --user --name=uv-kernel --display-name="uv-kernel"
``` 
