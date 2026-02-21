# Dissipative Time Crystals as Passively Protected Oscillating Qubits

**arXiv:** https://arxiv.org/abs/XXXX.XXXXX *(placeholder)*

Numerical simulations and Liouvillian spectral analysis of dissipative time crystals and passive quantum error protection in the driven-dissipative Bose–Hubbard dimer.

<p align="center">
  <img src="figures/bhd_phase_kick.gifbhd_phase_kick.gif" width="600">
</p>


This repository contains the numerical simulations and Liouvillian spectral analysis underlying our study of dissipative time-crystalline phases in the driven–dissipative Bose–Hubbard dimer (BHD).

We show that the time-crystalline phase hosts a long-lived oscillatory mode corresponding to a noiseless subsystem, enabling passive protection of an oscillating qubit encoded in the Liouvillian spectrum.

The GIF shows the response of the BDH time-crystalline state to a phase kick.

## Repository Contents

### 1. Exact Diagonalization (Dense)

- Full Liouvillian construction in symmetry-resolved sectors  
- Dense exact diagonalization  
- CPU implementation (NumPy / SciPy)  
- GPU implementation (PyTorch)

The repository includes the parameter set used in the paper for:

F = 1.8

Other drive strengths can be generated directly using the provided scripts.

---

### 2. Sparse Spectral Computations (Cluster)

- Sparse Liouvillian construction  
- Krylov / ARPACK-based eigensolvers  
- Designed for cluster-scale simulations  

All sparse eigenpairs used in the paper are included. These correspond to the isolated imaginary eigenmodes responsible for time-crystalline oscillations and their finite-size scaling behavior.

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