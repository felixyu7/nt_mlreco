# nt_mlreco
Machine learning pipeline for reconstruction in neutrino telescopes. Contains code for running SSCNN and neutrino telescope super-resolution on Prometheus events, as described in (PAPERS HERE). The working environment used to implement, run and test this code are as follows:

- Python 3.11.2
- PyTorch 2.2
- CUDA 12.1
- [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine) (for SSCNN) 0.5.4
- [Segmentation Models PyTorch](https://github.com/qubvel-org/segmentation_models.pytorch)
- Weights&Biases
- NumPy
- Awkward Arrays

## Usage

You can define a configuration file and run it with `python train.py -c config.cfg`.
TODO
