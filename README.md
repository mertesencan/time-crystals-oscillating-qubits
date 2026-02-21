# Dissipative Time Crystals as Passively Protected Oscillating Qubits

**arXiv:** https://arxiv.org/abs/XXXX.XXXXX *(placeholder)*

Numerical simulations and Liouvillian spectral analysis of dissipative time crystals and passive quantum error protection in the driven-dissipative Bose–Hubbard dimer.

<p align="center">
  <img src="./figures/bhd_phase_kick.gif" width="600">
</p>


This repository contains the numerical simulations and Liouvillian spectral analysis underlying our study of dissipative time-crystalline phases in the driven–dissipative Bose–Hubbard dimer (BHD).

We show that the time-crystalline phase hosts a long-lived oscillatory mode corresponding to a noiseless subsystem, enabling passive protection of an oscillating qubit encoded in the Liouvillian spectrum.

The GIF shows the response of the BDH time-crystalline state to a phase kick.

## Repository Contents

### 1. Exact Diagonalization (Dense)

- Full Liouvillian construction in symmetry-resolved sectors  
- Dense exact diagonalization  
- Script containing CPU implementation (NumPy / SciPy) GPU implementation (PyTorch)

The repository includes the parameter set used in the paper for:

F = 1.8, N = 1

Eigenvectors for other drive strengths (F) and scaling values (N) can be generated directly using the provided scripts.

---

### 2. Sparse Spectral Computations (Cluster)

- Sparse Liouvillian construction  
- Krylov / ARPACK-based eigensolvers  
- Designed for cluster-scale simulations  

All sparse eigenpairs used in the paper are included, which were obtained using the High Performance Compute (HPC) compute cluster at the Advanced Research Computing (ARC) services provided to University of Oxford.

These correspond to the isolated imaginary eigenmodes responsible for time-crystalline oscillations and their finite-size scaling behavior.

---

### 3. Eigenstate Visualization

- Reconstruction of density matrices from eigenvectors  
- Observable projections  
- Symmetry-sector analysis  
- Visualization of oscillatory eigenoperators  

These tools allow direct inspection of the Liouvillian eigenoperators forming the oscillatory qubit subspace.

---

### 4. Time Evolution

- Direct Lindblad master-equation evolution  
- Phase-kick protocol simulations  
- Extraction of oscillation frequency and decay rates  

This connects the spectral structure to real-time dynamics and demonstrates the emergence of a dynamically protected oscillatory mode.


## Environment Setup

This project was developed with:

Python 3.13.2

To reproduce the environment, install **conda**, and run:

```bash
conda create -n myenv python=3.13.2
conda activate myenv
conda install --file conda-spec.txt
pip install -r requirements.txt