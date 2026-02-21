
import torch
import numpy as np
import scipy as sp
import scipy.sparse.linalg
import psutil

from qutip import *
import qutip as qt

import time
import os

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
torch.set_printoptions(threshold=torch.inf)


def leftmult(op, cutoff_A, cutoff_B):
    I = torch.eye(cutoff_A*cutoff_B).to(device)
    return torch.einsum('ij,kl->ijkl', op, I)
def rightmult(op, cutoff_A, cutoff_B):
    I = torch.eye(cutoff_A*cutoff_B).to(device)
    return torch.einsum('ij,kl->ijkl', I, op)
def bothsidemult(op1,op2):
    return torch.einsum('ij,kl->ijkl', op1, op2)

#Define the annihilation operator for the first mode
def annihilation_operator_first_mode(cutoff_A, cutoff_B):
    a = torch.zeros((cutoff_A, cutoff_B, cutoff_A, cutoff_B), dtype=torch.complex128)
    for n in range(1, cutoff_A):
        for m in range(cutoff_B):
            a[n-1, m, n , m] = torch.sqrt(torch.tensor(n, dtype=torch.float64))
    return a

#Define the annihilation operator for the second mode
def annihilation_operator_second_mode(cutoff_A, cutoff_B):
    a = torch.zeros((cutoff_A, cutoff_B, cutoff_A, cutoff_B), dtype=torch.complex128)
    for n in range(cutoff_A):
        for m in range(1, cutoff_B):
            a[n, m-1, n, m] = torch.sqrt(torch.tensor(m, dtype=torch.float64))
    return a

#Define the creation operator for the first mode
def creation_operator_first_mode(cutoff_A, cutoff_B):
    a_dagger = torch.zeros((cutoff_A, cutoff_B, cutoff_A, cutoff_B), dtype=torch.complex128)
    for n in range(cutoff_A - 1):
        for m in range(cutoff_B):
            a_dagger[n+1, m, n, m] = torch.sqrt(torch.tensor(n + 1, dtype=torch.float64))
    return a_dagger

#Define the creation operator for the second mode
def creation_operator_second_mode(cutoff_A, cutoff_B):
    a_dagger = torch.zeros((cutoff_A, cutoff_B, cutoff_A, cutoff_B), dtype=torch.complex128)
    for n in range(cutoff_A):
        for m in range(cutoff_B - 1):
            a_dagger[n, m+1, n, m] = torch.sqrt(torch.tensor(m + 1, dtype=torch.float64))
    return a_dagger


def get_torch_eigspectra_all(L_symmetry, symmetry, cutoff_A, cutoff_B):
    Lmat = L_symmetry.permute(0,3,1,2).reshape(cutoff_A*cutoff_B*cutoff_A*cutoff_B//4,cutoff_A*cutoff_B*cutoff_A*cutoff_B//4).to(device)
    eigenvalues, right_eigenvectors = torch.linalg.eig(Lmat)
    
    # Assuming eigenvalues is a complex tensor
    real_parts = torch.real(eigenvalues)
    sorted_indices = torch.argsort(real_parts)  # Indices to sort by real parts

    # Use the sorted indices to sort the eigenvalues
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = right_eigenvectors[:, sorted_indices]

    print(symmetry, "eig_val", np.round(sorted_eigenvalues[-1],8))

    return sorted_eigenvalues, sorted_eigenvectors


def get_torch_eigspectra_all_numpy(L_symmetry, symmetry, cutoff_A, cutoff_B, k=2, algo_type='LR'):
    # Compute the matrix dimension (assuming the same reshape as before)
    n = cutoff_A * cutoff_B * cutoff_A * cutoff_B // 4

    # Reshape L_symmetry into a 2D matrix (remains as a torch tensor)
    Lmat = L_symmetry.permute(0, 3, 1, 2).reshape(n, n)
    
    # Convert the torch tensor to a NumPy array (CPU only)
    Lmat_np = Lmat.cpu().numpy().astype(np.complex128)
    
    # Convert the dense NumPy array to a sparse CSR matrix
    L_sparse = scipy.sparse.csr_matrix(Lmat_np, dtype=np.complex128)
    
    # Compute k eigenpairs using SciPy's sparse eigen solver.
    # Use eigsh for symmetric matrices. If the matrix is non-symmetric, consider using eigs.
    eigenvalues, eigenvectors = sp.sparse.linalg.eigs(L_sparse, k=k, sigma=None, which=algo_type, ncv=100, return_eigenvectors=True)
    
    # Sort the eigenpairs by eigenvalue (ascending order)
    sorted_indices = np.argsort(eigenvalues)
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    
    # Report CPU memory usage using psutil
    process = psutil.Process(os.getpid())
    mem_MB = process.memory_info().rss / 1024**2
    print("CPU memory usage: {:.2f} MB".format(mem_MB))
    
    print(symmetry, "eig_val", np.round(sorted_eigenvalues[-1], 8))
    
    # Convert the results back to torch tensors
    sorted_eigenvalues_torch = torch.from_numpy(sorted_eigenvalues)
    sorted_eigenvectors_torch = torch.from_numpy(sorted_eigenvectors)
    
    return sorted_eigenvalues_torch, sorted_eigenvectors_torch



#####################
## MAIN CODE HERE ###
#####################


eig_threshold = 1e-2

cutoff_A = 6  # First mode
cutoff_Bs = [10]
Delta=0.8
J=1.1

unscaled_Fs = [1.8]
Ns = [1]
for i in range(1):
    
    print("i", i)
    N = Ns[i]
    cutoff_B = cutoff_Bs[i]
    
    # Compute the annihilation operator
    aA = annihilation_operator_first_mode(cutoff_A, cutoff_B).to(device)
    aB = annihilation_operator_second_mode(cutoff_A, cutoff_B).to(device)

    # Compute the creation operators
    adagA = creation_operator_first_mode(cutoff_A, cutoff_B).to(device)
    adagB = creation_operator_second_mode(cutoff_A, cutoff_B).to(device)

    # two modes as a single dimension
    aA = aA.reshape(cutoff_A * cutoff_B, cutoff_A * cutoff_B).to(device)
    aB = aB.reshape(cutoff_A * cutoff_B, cutoff_A * cutoff_B).to(device)
    adagA = adagA.reshape(cutoff_A * cutoff_B, cutoff_A * cutoff_B).to(device)
    adagB = adagB.reshape(cutoff_A * cutoff_B, cutoff_A * cutoff_B).to(device)

    for unscaled_F in unscaled_Fs:
       
        eig_prefix_npy = 'f'+str(cutoff_A)+'_'+str(cutoff_B)+"_N"+str(N)+"_r"+str(unscaled_F)

        F = unscaled_F * np.sqrt(N)
        U = 1/N
        
        print("---------------")
        print("---------------")

        save_eig_pp = True
        save_eig_mm = True
        save_eig_pm = True

        F = unscaled_F * np.sqrt(N)
        U = 1/N
        print("N =", N)
        print("F =", unscaled_F)


        H=(-Delta-J)*torch.matmul(adagB,aB)+(-Delta+J)*torch.matmul(adagA,aA)+np.sqrt(2)*F*(adagB+aB)+(U/4)*(
            2*torch.matmul(torch.matmul(adagA,adagA),torch.matmul(aA,aA))+
            2*torch.matmul(torch.matmul(adagB,adagB),torch.matmul(aB,aB))+
            2*torch.matmul(torch.matmul(adagB,adagB),torch.matmul(aA,aA))+
            2*torch.matmul(torch.matmul(aB,aB),torch.matmul(adagA,adagA))+
            8*torch.matmul(torch.matmul(adagB,aB),torch.matmul(adagA,aA))
            ).to(device)

        L = (- 1j*leftmult(H, cutoff_A, cutoff_B) + 1j*rightmult(H, cutoff_A, cutoff_B) + 2*bothsidemult(aB, adagB) - leftmult(adagB@aB, cutoff_A, cutoff_B) - rightmult(adagB@aB, cutoff_A, cutoff_B)).to(device)

        Lprop=L.view(cutoff_A,cutoff_B,cutoff_A,cutoff_B,cutoff_A,cutoff_B,cutoff_A,cutoff_B).to(device) #proper dimensions for the Liouvillian
        
        del L

        #Even photon numbers in mode 1 represent the positive block, odd photons negative block
        Lpp=Lprop[::2,:,::2,:,::2,:,::2,:].to(device) #extracting the positive block
        #del Lprop
        print(Lpp.shape)
        Lpp=Lpp.reshape(cutoff_A*cutoff_B//2,cutoff_A*cutoff_B//2,cutoff_A*cutoff_B//2,cutoff_A*cutoff_B//2).to(device)

        # get time to get eigenvectors
        start_time = time.time()
        eigvals_pp_npy, eigvecs_pp_npy = get_torch_eigspectra_all_numpy(Lpp, 'pp', cutoff_A, cutoff_B, algo_type='SM')
        end_time = time.time()
        print("Time to get pp eigenvectors:", np.round(end_time - start_time))

        # Check if pp and mm eigvals is nearly zero
        pp_eigval = eigvals_pp_npy[-1]
        if np.abs(pp_eigval) > eig_threshold:
            print("EIG ERROR: pp eigval is not zero")
            save_eig_pp = False
            #continue

        if save_eig_pp:
            np.save(eig_prefix_npy + "_evecs_beam_pp", eigvecs_pp_npy)
            np.save(eig_prefix_npy + "_evals_beam_pp", eigvals_pp_npy)
        
        del Lpp
        del eigvecs_pp_npy
        del eigvals_pp_npy

        Lmm=Lprop[1::2,:,1::2,:,1::2,:,1::2,:].to(device) #extracting the negative block
        #del Lprop
        print(Lmm.shape)
        Lmm=Lmm.reshape(cutoff_A*cutoff_B//2,cutoff_A*cutoff_B//2,cutoff_A*cutoff_B//2,cutoff_A*cutoff_B//2).to(device)

        start_time = time.time()
        eigvals_mm_npy, eigvecs_mm_npy = get_torch_eigspectra_all_numpy(Lmm, 'mm', cutoff_A, cutoff_B, algo_type='SM')
        end_time = time.time()
        print("Time to get mm eigenvectors:", np.round(end_time - start_time))

        mm_eigval = eigvals_mm_npy[-1]
        if np.abs(mm_eigval) > eig_threshold:
            print("EIG ERROR: mm eigval is not zero")
            save_eig_mm = False
            #continue

        if save_eig_mm:
            np.save(eig_prefix_npy + "_evecs_beam_mm", eigvecs_mm_npy)
            np.save(eig_prefix_npy + "_evals_beam_mm", eigvals_mm_npy)
        
        del Lmm
        del eigvecs_mm_npy
        del eigvals_mm_npy
        
        #Even photon numbers in mode 1 represent the positive block, odd photons negative block
        Lpm=Lprop[::2,:,::2,:,1::2,:,1::2,:].to(device) #extracting the negative block
        #del Lprop
        print(Lpm.shape)
        Lpm=Lpm.reshape(cutoff_A*cutoff_B//2,cutoff_A*cutoff_B//2,cutoff_A*cutoff_B//2,cutoff_A*cutoff_B//2).to(device)

        start_time = time.time()
        eigvals_pm_npy, eigvecs_pm_npy = get_torch_eigspectra_all_numpy(Lpm, 'pm', cutoff_A, cutoff_B, k=6)
        end_time = time.time()
        print("Time to get pm eigenvectors:", np.round(end_time - start_time))

        if save_eig_pm:
            np.save(eig_prefix_npy + "_evecs_beam_pm", eigvecs_pm_npy)
            np.save(eig_prefix_npy + "_evals_beam_pm", eigvals_pm_npy)
        
        del Lpm
        del eigvecs_pm_npy
        del eigvals_pm_npy

        del Lprop
