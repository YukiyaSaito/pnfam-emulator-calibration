# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import numpy as np
import pandas as pd
from scipy.linalg import eigh, eig
import matplotlib.pyplot as plt
import time
import tensorflow as tf
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import sys
import random as rn
from scipy.interpolate import lagrange
from numpy.polynomial.polynomial import Polynomial
import scipy.integrate as integrate
from scipy.special import gamma
import scipy.integrate as integrate
import matplotlib.ticker as ticker
from scipy.optimize import least_squares, nnls

emass = 0.511 # MeV
alpha_c = 1/137
hbarc = 197.33 # MeV fm
compton = hbarc/emass # fm
kappa = 6147 #s
del_np = 1.293 # MeV
del_nH = 0.782 # MeV


# =====================================================================
#  New data loader: sample index + data directory based
# =====================================================================

def data_table_from_files(sample_indices, data_dir, e_min, e_max):
    """
    Load strength function data from sample_*.dat files.

    Parameters
    ----------
    sample_indices : array-like of int
        Which sample files to load (e.g. [1, 5, 12,...]).
    data_dir : str
        Directory containing sample_*.dat files.
    e_min, e_max : float
        Energy window for truncation (from config.py).

    Returns
    -------
    Lors : list of np.ndarray
        Each element is an (N, 2) array with columns [energy, strength],
        truncated to [e_min, e_max].
    """
    Lors = []
    for idx in sample_indices:
        filepath = os.path.join(data_dir, f'sample_{idx}.dat')
        # skip comment lines starting with #
        file = np.loadtxt(filepath, comments='#')
        # truncate to energy window
        mask = (file[:, 0] >= e_min) & (file[:, 0] <= e_max)
        file = file[mask]
        Lors.append(file)
    return Lors


# =====================================================================
#  Everything below is the original helper.py, with only hardcoded
#  absolute paths replaced by relative paths.
# =====================================================================


def encode_initial_guess(random_initial_guess, E, B, n, retain):
    """
    Centered retain: place k=round(retain*n) fitted (E,B) in the middle of the n-diagonal.
    D outside the kept block is filled by +/-2 stepping:
      left side:  min(E_sel) - 2, -4,...
      right side: max(E_sel) + 2, +4,...
    v0 outside kept block is zero.

    Parameters
    ----------
    random_initial_guess : np.ndarray (float64)
        Base packed parameter vector.
    E, B : array-like
        Fitted energies and strengths (length >= k_keep).
    n : int
        Matrix dimension.
    retain : float in (0,1]
        Fraction to retain.

    Returns
    -------
    params : np.ndarray (float64)
    """
    n = int(n)
    params = np.asarray(random_initial_guess, dtype=np.float64).copy()

    # how many to keep (centered)
    k_keep = int(round(float(retain) * n))
    k_keep = max(1, min(k_keep, n))
    left  = (n - k_keep) // 2
    right = left + k_keep

    # sort by energy and take centered slice of fitted set
    E = np.asarray(E, dtype=np.float64).reshape(-1)
    B = np.asarray(B, dtype=np.float64).reshape(-1)
    order = np.argsort(E)
    E, B = E[order], B[order]

    if len(E) < k_keep:
        raise ValueError(f"E,B need at least k_keep={k_keep} entries (got {len(E)}).")

    m = len(E)
    start = (m - k_keep) // 2
    E_sel = E[start:start + k_keep]
    B_sel = B[start:start + k_keep]

    # ---- build full D: kept center + +/-2 stepping on ends ----
    D_full = np.empty(n, dtype=np.float64)
    D_full[left:right] = E_sel

    step = 2.0
    # left side: decreasing by step from min(E_sel)
    cur = E_sel[0]
    for i in range(left - 1, -1, -1):
        cur -= step
        D_full[i] = cur
    # right side: increasing by step from max(E_sel)
    cur = E_sel[-1]
    for i in range(right, n):
        cur += step
        D_full[i] = cur

    # ---- v0: sqrt(B) in kept block, zeros elsewhere ----
    v0_full = np.zeros(n, dtype=np.float64)
    v0_full[left:right] = np.sqrt(np.maximum(B_sel, 0.0)).astype(np.float64)

    # ---- write back into packed vector (your layout) ----
    num_upper = n * (n + 1) // 2
    idx = 0
    idx += 1                                   # eta (keep as-is)
    params[idx:idx + n] = v0_full; idx += n    # v0
    params[idx:idx + n] = D_full; idx += n     # D diagonal
    idx += num_upper                           # S1 (leave)
    idx += num_upper                           # S2 (leave)
    # x1..x4 unchanged

    return params




'''
List of helper functions for the central point
'''
# ---------- small helpers ----------
def _softplus(x):            # numpy-friendly for init
    x = np.asarray(x, dtype=float)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)

def _inv_softplus(y):
    y = np.maximum(np.asarray(y, dtype=float), 1e-12)
    return np.log(np.expm1(y))

# Enforce E strictly increasing and B >= 0 via reparameterization (done in TF)
def _unpack_params_tf(z, n, wmin, min_spacing):
    z = tf.convert_to_tensor(z, tf.float64)
    zE, zB = z[:n], z[n:]

    e0 = wmin + tf.nn.softplus(zE[0])                       # E1 >= wmin
    gaps = tf.nn.softplus(zE[1:]) + min_spacing             # positive gaps
    E = tf.concat([e0[None], e0 + tf.cumsum(gaps)], axis=0) # (n,)

    B = tf.nn.softplus(zB) ** 2                             # >=0 (extra margin)
    return E, B

def _pack_init_np(E0, B0, wmin, min_spacing):
    zE = np.empty_like(E0, dtype=float)
    zE[0] = _inv_softplus(E0[0] - wmin)
    gaps = np.diff(E0)
    zE[1:] = _inv_softplus(np.maximum(gaps - min_spacing, 1e-6))
    zB = _inv_softplus(np.sqrt(np.maximum(B0, 1e-12)))
    return np.concatenate([zE, zB])

# ---------- main fit (uses TF inside the residual) ----------
def fit_strength_with_tf_lorentzian(omega, y, n, eta,
                                    grid_M=None, min_spacing=0.2, l2=0.0):
    """
    omega, y : (G,) arrays (measured strength)
    n        : number of Lorentzians
    eta      : fixed width (float)
    grid_M   : number of NNLS grid centers (default=len(omega))
    min_spacing : minimal spacing between centers during refine (MeV)
    l2       : small Tikhonov on raw z (0 disables)

    Returns: E_hat (n,), B_hat (n,), y_hat (G,)
    """
    omega_np = np.asarray(omega, float)
    y_np = np.asarray(y, float)
    wmin, wmax = float(omega_np.min()), float(omega_np.max())
    if grid_M is None:
        grid_M = len(omega_np)

    # ---- Stage 1: NNLS seed on a dense center grid ----
    E_grid = np.linspace(wmin + 1e-6, wmax - 1e-6, grid_M)
    A = 1.0 / ((omega[:, None] - E_grid[None, :])**2 + (eta**2)/4.0) * (eta/(2*np.pi))
    coeff, _ = nnls(A, y)
    idx = np.argsort(coeff)[-n:]
    E0 = np.sort(E_grid[idx])
    B0 = coeff[idx][np.argsort(E_grid[idx])]

    # ensure minimal spacing in seed
    for k in range(1, n):
        if E0[k] - E0[k-1] < min_spacing:
            E0[k] = E0[k-1] + min_spacing
    z0 = _pack_init_np(E0, B0, wmin, min_spacing)

    # ---- Stage 2: nonlinear refine using TF lorentzian in the residual ----
    def residuals(z):
        # unpack in TF, evaluate model with your TF lorentzian, return numpy residuals
        E_tf, B_tf = _unpack_params_tf(z, n, tf.constant(wmin, tf.float64),
                                        tf.constant(min_spacing, tf.float64))
        yhat_tf = give_me_Lorentzian(omega, E_tf, B_tf, tf.constant(eta, tf.float64))
        r = (yhat_tf.numpy() - y)
        if l2 > 0:
            r = np.concatenate([r, np.sqrt(l2) * np.asarray(z, dtype=float)])
        return r

    res = least_squares(residuals, z0, method="trf",
                        max_nfev=5000, xtol=1e-10, ftol=1e-10, gtol=1e-10)

    # unpack final parameters and compute final curve (all through TF)
    E_tf, B_tf = _unpack_params_tf(res.x, n, tf.constant(wmin, tf.float64),
                                    tf.constant(min_spacing, tf.float64))
    yhat_tf = give_me_Lorentzian(omega, E_tf, B_tf, tf.constant(eta, tf.float64))

    E_hat = E_tf.numpy()
    B_hat = B_tf.numpy()
    y_hat = yhat_tf.numpy()
    return E_hat, B_hat, y_hat



# define the Fermi function as a function of electron energy
def Fermi(Z,A,W):
    
    R = 1.2*A**(1/3)
    R = R / compton
    
    gamma_1 = np.sqrt(1 - (alpha_c*Z)**2)
    p = np.sqrt(W**2 - 1)
    y = alpha_c*Z*W/p
    
    gamma_part = np.abs(gamma(gamma_1 + 1j*y))**2/gamma(2*gamma_1+1)**2
    part_1 = 4*(2*p*R)**(-2*(1-gamma_1))
    part_2 = np.exp(np.pi*y)
    L_0 = 0.5*(1+ gamma_1)
    
    return part_1*part_2*gamma_part*L_0



# define the phase factor integral
# p W ( W0 - W)**2 F(Z,W)

def phase_factor_integrand(kind,Z,A,W_0,W):
    '''
    Based on the kind argument I can have different types
    of phase-space factors

    kind = 0: pW(W_0-W)^2F(Z,W)


    '''
    
    if (W**2 - 1 < 0): return 0
    p = np.sqrt(W**2 - 1)

    if (kind == 0):
        return p*W*(W_0-W)**2*Fermi(Z+1,A,W)
    elif (kind == 1): # ka
        return p*W**2*(W_0-W)**2*Fermi(Z+1,A,W)
    elif (kind == 2): # kb
        return p*(W_0-W)**2*Fermi(Z+1,A,W)
    elif (kind == 3): # kc
        return p*W**3*(W_0-W)**2*Fermi(Z+1,A,W)
    else:
        print('Wrong kind in phase_factor_integrand!')
        exit(-1)


def phase_factor(kind,Z,A,W_0):

    '''
    Performs integration of the phase_factor_integrand function
    '''
    
    integrand = lambda W: phase_factor_integrand(kind,Z,A,W_0,W)
    
    res, err = integrate.quad(integrand, 1, W_0)
    

    return res

def theta2(K_val):
    if (K_val == 0):
        return 1
    else:
        return 2
    
    
    
def fit_phase_space(kind, Z, A, ala_np):

    ''' 
    
    Fit the phase-space factor to a N_fit-degree polynomial

    returns:

    popt - list of polynomial parameters for the fit
    
    '''

    # Number of Chebyshev grid points
    n_points = 20

    # Generate the Chebyshev grid points
    chebyshev_grid = np.cos(np.pi * (np.arange(n_points) + 0.5) / n_points)

    # Define the range for your x-coordinates (e.g., between -1 and 1)
    x_min, x_max = emass, ala_np+del_np

    # Transform the Chebyshev grid to a regular grid
    regular_grid = 0.5 * (x_min + x_max) + 0.5 * (x_max - x_min) * chebyshev_grid

    # calculate phase-space factor on chebyshev grid
    ydata = []
    for w0 in regular_grid:

        ydata.append(phase_factor(kind, Z+1,A,w0/emass))

    xdata = regular_grid
    ydata = np.array(ydata)


    # on the same figure perform a Lagrange fit
    poly = lagrange(xdata, ydata)



    return poly


@tf.function
def evaluate_polynomial_tf(coeffs, x):
    """
    TensorFlow function to evaluate a polynomial at a given point.
    
    Parameters:
        coeffs (tf.Tensor): Coefficients [a0, a1,..., aN] 
        x (float or tf.Tensor): The x values to evaluate the polynomial at
    
    Returns:
        tf.Tensor: Polynomial values at x
    """
    N = tf.shape(coeffs)[0]
    powers = tf.range(N, dtype=coeffs.dtype)

    # Ensure x is properly broadcasted (even if a single value)
    x_tensor = tf.reshape(x, [-1, 1])  # Reshape to 2D tensor
    x_powers = tf.pow(x_tensor, powers)

    return tf.reduce_sum(coeffs * x_powers, axis=-1)
    



# TensorFlow routine
@tf.function
def half_life_loss(eigenvalues, B,coeffs,g_A):
    '''
        Routine which calculates the half-lives
    '''
    
    W_0 = (tf.constant(del_np, dtype=tf.float64) - eigenvalues) / tf.constant(emass, dtype=tf.float64)
    
    # Apply phase factor only if W_0 > 1
    valid = tf.greater(W_0, 1.0)
    
    #phase = tf.constant(1.0, dtype=tf.float64)
    
    # Compute phase factor and contributions
    coefficients = tf.reverse(coeffs, axis=[0])
    #coefficients = tf.constant(coeffs[::-1], dtype=tf.float64)
    
    
    x_tensor = W_0*emass

    # Evaluate the polynomial at all x values
    phase = evaluate_polynomial_tf(coefficients, x_tensor)
    
    # Add contributions only where W_0 > 1
    contributions = tf.where(valid, phase * B * tf.constant(g_A**2, dtype=tf.float64) \
* tf.math.log(tf.constant(2.0, dtype=tf.float64)) / tf.constant(kappa, dtype=tf.float64), tf.constant(0.0, dtype=tf.float64))
    
    # Sum the contributions
    suma = tf.math.log(tf.constant(2.0, dtype=tf.float64))/tf.reduce_sum(contributions)
    
    return suma

@tf.function
def give_me_Lorentzian(energy, poles, strength, width):
    if isinstance(energy, np.ndarray):
        energy = tf.convert_to_tensor(energy, dtype=tf.float64)

    poles = tf.convert_to_tensor(poles, dtype=tf.float64)
    strength = tf.convert_to_tensor(strength, dtype=tf.float64)

    energy_expanded = tf.expand_dims(energy, axis=-1)

    numerator = strength * (width / 2 / np.pi)
    denominator = ((energy_expanded - poles) ** 2 + (width ** 2 / 4))
    
    lorentzian = numerator / denominator

    value = tf.reduce_sum(lorentzian, axis=-1)
    
    
    return value



# nec_mat for M_true(a) = D + a * S1 + b * S2
def nec_mat(n):
    D = np.diag(np.random.uniform(1, 10, n))
    A = np.random.uniform(1, 10, (n, n))
    S1 = np.abs(A + A.T) / 2
    S2 = np.abs(A + A.T) / 2
    return D, S1, S2




def data_table(fmt_data, coeffs, g_A, nucnam):
    '''
    
    Here split the dataset from "beta_decay_data" folder
    into: training set, validation set and test set
    use any ratio you like (e.g. 0.8 0.1 0.1)
    
    For the optimizatio use only training set, and after you finish
    test it on validation set
    
    returns also number of QRPA poles n_QRPA
    
    '''

    Lors = []
    # HLs = []
    

    for frmt in fmt_data:
        
        alpha = frmt[0]
        beta = frmt[1]

        # first open the file with the data
        file = np.loadtxt('beta_decay_data_'+nucnam+'/lorm_'+nucnam+'_'+beta+'_'+alpha+'.out')
        
        # normalize the Lorentzians
        #norm = np.sum(file[:,1])
        #file[:,1] = file[:,1]/norm
        
        #print(alpha, beta, norm**2)

        # file = file[file[:,0]<del_nH]
        #file = file[file[:,0]<10]
        # file = file[file[:,0]>-10]
        file = file[file[:,0] < 30.0] 
        file = file[file[:,0] > -5.0]

        Lors.append(file)  
        
        # # now calculate half-lives the old way
        # file = np.loadtxt('../beta_decay_data_'+nucnam+'/excm_'+nucnam+'_'+beta+'_'+alpha+'.out')
        # file = file[file[:,0]<del_nH]
        # file = file[file[:,0]>-10]
        # HLs.append(half_life_loss(file[:,0], file[:,1],coeffs, g_A))

     
    # return Lors, HLs
    return Lors



'''
    data table is now constructed for alpha & beta parameters
'''

def data_table_4d(fmt_data, coeffs, g_A, nucnam):
    '''
    Loads 4-parameter strength data.
    Expected format: lorm_nucnam_p1_p2_p3_p4.out
    '''
    Lors = []

    for frmt in fmt_data:
        p1 = frmt[0]
        p2 = frmt[1]
        p3 = frmt[2]
        p4 = frmt[3]

        # Construct the 4D filename
        filename = f'beta_decay_data_{nucnam}/lorm_{nucnam}_{p1}_{p2}_{p3}_{p4}.out'
        
        file = np.loadtxt(filename)
        
        # Truncate to the physical window
        file = file[file[:,0] < 30.0] 
        file = file[file[:,0] > -5.0]

        Lors.append(file)  

    return Lors



def modified_DS(params, n): ## Original
    ''''
    Build PMM matrices:
    M = D + (alpha-alpha_0)*exp(-(beta-beta_0)*x1)*S1 + (alpha-alpha_0)*exp(-(beta-beta_0)*x2)*S2
    and external field:
    v(alpha,beta) = v0 + (alpha-alpha_0)*v1 + (beta-beta_0)*v2
    '''
    D_shape = (n, n)
    S1_shape = (n, n)
    S2_shape = (n, n)

    # Indices in param vector
    idx = 0
    eta = tf.convert_to_tensor(params[idx])
    idx += 1

    v0_mod = tf.convert_to_tensor(params[idx:idx+n])
    idx += n
    v1_mod = tf.convert_to_tensor(params[idx:idx+n])
    idx += n
    v2_mod = tf.convert_to_tensor(params[idx:idx+n])
    idx += n

    D_mod = tf.linalg.diag(params[idx:idx+n])
    idx += n

    # S1 upper triangle
    S1_mod = tf.zeros(S1_shape, dtype=tf.float64)
    upper_tri_indices1 = np.triu_indices(S1_shape[0])
    indices1 = tf.constant(list(zip(upper_tri_indices1[0], upper_tri_indices1[1])))

    num_upper1 = len(upper_tri_indices1[0])
    S1_mod = tf.tensor_scatter_nd_update(S1_mod, indices1,
                                         params[idx:idx+num_upper1])
    idx += num_upper1

    S1_mod = S1_mod + tf.linalg.band_part(tf.transpose(S1_mod), -1, 0) - tf.linalg.diag(tf.linalg.diag_part(S1_mod))

    # S2 upper triangle
    S2_mod = tf.zeros(S2_shape, dtype=tf.float64)
    upper_tri_indices2 = np.triu_indices(S2_shape[0])
    indices2 = tf.constant(list(zip(upper_tri_indices2[0], upper_tri_indices2[1])))

    num_upper2 = len(upper_tri_indices2[0])
    S2_mod = tf.tensor_scatter_nd_update(S2_mod, indices2,
                                         params[idx:idx+num_upper2])
    idx += num_upper2

    S2_mod = S2_mod + tf.linalg.band_part(tf.transpose(S2_mod), -1, 0) - tf.linalg.diag(tf.linalg.diag_part(S2_mod))
    
    
    # Add new learned params x1 and x2 and x3
    x1 = tf.convert_to_tensor(params[idx])
    idx += 1
    x2 = tf.convert_to_tensor(params[idx])
    idx += 1
    x3 = tf.convert_to_tensor(params[idx])
    idx += 1


    # return D_mod, S1_mod, S2_mod, v0_mod, eta, x1, x2, x3
    return D_mod, S1_mod, S2_mod, v0_mod, v1_mod, v2_mod, eta, x1, x2, x3

# def modified_DS(params, n):
#     '''
#     Build 2nd-Order PMM matrix with STATIC v0:
#     M = D + d_a*S1 + d_b*S2 + (d_a^2)*S11 + (d_b^2)*S22 + (d_a*d_b)*S12
#     v = v0
#     '''
#     idx = 0
    
#     # 1. Width parameter
#     eta = tf.convert_to_tensor(params[idx]); idx += 1

#     # 2. Static transition vector
#     v0_mod = tf.convert_to_tensor(params[idx:idx+n]); idx += n

#     # 3. Diagonal matrix (Poles)
#     D_mod = tf.linalg.diag(params[idx:idx+n]); idx += n

#     # 4. Unpack 5 Symmetric Matrices (S1, S2, S11, S22, S12)
#     upper_tri_indices = np.triu_indices(n)
#     indices = tf.constant(list(zip(upper_tri_indices[0], upper_tri_indices[1])))
#     num_upper = len(upper_tri_indices[0])
    
#     matrices = []
#     for _ in range(5):
#         M = tf.zeros((n, n), dtype=tf.float64)
#         M = tf.tensor_scatter_nd_update(M, indices, params[idx:idx+num_upper])
#         # Mirror across diagonal to make it symmetric
#         M = M + tf.linalg.band_part(tf.transpose(M), -1, 0) - tf.linalg.diag(tf.linalg.diag_part(M))
#         matrices.append(M)
#         idx += num_upper
        
#     S1_mod, S2_mod, S11_mod, S22_mod, S12_mod = matrices
    
#     # 5. Extra params
#     x1 = tf.convert_to_tensor(params[idx]); idx += 1
#     x2 = tf.convert_to_tensor(params[idx]); idx += 1
#     x3 = tf.convert_to_tensor(params[idx]); idx += 1

#     return D_mod, S1_mod, S2_mod, S11_mod, S22_mod, S12_mod, v0_mod, eta, x1, x2, x3


def modified_DS_4d(params, n):
    '''
    Build 4D PMM matrices:
    M = D + dt1*S1 + dt2*S2 + dt3*S3 + dt4*S4
    v = v0 + dt1*v1 + dt2*v2 + dt3*v3 + dt4*v4
    '''
    idx = 0
    eta = tf.convert_to_tensor(params[idx]); idx += 1

    # Transition vectors
    v0_mod = tf.convert_to_tensor(params[idx:idx+n]); idx += n
    v1_mod = tf.convert_to_tensor(params[idx:idx+n]); idx += n
    v2_mod = tf.convert_to_tensor(params[idx:idx+n]); idx += n
    v3_mod = tf.convert_to_tensor(params[idx:idx+n]); idx += n
    v4_mod = tf.convert_to_tensor(params[idx:idx+n]); idx += n

    # Diagonal (Poles)
    D_mod = tf.linalg.diag(params[idx:idx+n]); idx += n

    # 4 Symmetric Matrices
    S_shape = (n, n)
    upper_tri_indices = np.triu_indices(n)
    indices = tf.constant(list(zip(upper_tri_indices[0], upper_tri_indices[1])))
    num_upper = len(upper_tri_indices[0])

    matrices = []
    for _ in range(4):
        S_mod = tf.zeros(S_shape, dtype=tf.float64)
        S_mod = tf.tensor_scatter_nd_update(S_mod, indices, params[idx:idx+num_upper])
        S_mod = S_mod + tf.linalg.band_part(tf.transpose(S_mod), -1, 0) - tf.linalg.diag(tf.linalg.diag_part(S_mod))
        matrices.append(S_mod)
        idx += num_upper
        
    S1_mod, S2_mod, S3_mod, S4_mod = matrices

    # Width dependence params (x0, x1, x2, x3, x4)
    x_width = [tf.convert_to_tensor(params[idx+i]) for i in range(5)]

    return D_mod, S1_mod, S2_mod, S3_mod, S4_mod, v0_mod, v1_mod, v2_mod, v3_mod, v4_mod, eta, x_width


# cost_function
# def cost_function(params, n, fmt_data, Lors_true, HLs_true,coeffs,g_A, weight, central_point, retain):
    
#     '''
#     params: tf.Variable
#     D_shape, S1_shape, S2_shape : int
#     alpha_values, beta_values: list
#     data_table: pd.DataFrame
    
#     calculates the cost function by subtracting two Lorentzians
    
#     '''
    
#     D_mod, S1_mod, S2_mod, v0_mod, eta, x1, x2, x3 = modified_DS(params, n)
    
    
#     total_cost = 0
    
#     count = 0
#     HLs_calc = []
#     for idx, alpha in enumerate(fmt_data):

#         M_true = D_mod + (float(alpha[0])-float(central_point[0])) * S1_mod \
#             + (float(alpha[1]) - float(central_point[1])) * S2_mod
        

#         eigenvalues, eigenvectors = tf.linalg.eigh(M_true)
        
#         n_i = eigenvalues.shape[0]
#         k_keep = int(round(retain * n_i))         # how many eigenvalues to keep
#         k_keep = max(1, min(k_keep, n_i))         # safety: clamp between 1 and n
        
#         left  = (n_i - k_keep) // 2               # starting index of the centered block
#         right = left + k_keep                     # ending index (exclusive)
        
#         eigenvalues  = eigenvalues[left:right]
#         eigenvectors = eigenvectors[:, left:right]
        
#         projections = tf.linalg.matvec(tf.transpose(eigenvectors), v0_mod)
        
#         # Square each projection
#         B = tf.square(projections)
        
#         mask = tf.cast((eigenvalues > -10) & (eigenvalues < 15), dtype=tf.float64)

#         # Apply the mask to zero out B where eigenvalue is negative
#         B = B * mask
        

#         #B = [tf.square(tf.tensordot(eigenvectors[:, i], v0_mod, axes=1)) for i in range(eigenvectors.shape[1])]
#         Lor_true = tf.constant(Lors_true[count][:,1], dtype=tf.float64)

#         #Generate the x values
#         x = tf.constant(Lors_true[count][:,0], dtype=tf.float64)
        
#         width = tf.sqrt(tf.square(eta) + tf.square(x1 + x2*float(alpha[0]) + x3*float(alpha[1])))
        

#         # Use tf.map_fn to apply the give_me_Lorentzian function over the x values
#         Lor = give_me_Lorentzian(x, eigenvalues, B, width)
        
        
        

#         total_cost += tf.reduce_sum((Lor - Lor_true) ** 2)
#         ''' Total cost modified to match previous definitions'''
        
        
#         ''' Add half-lives to optimization as well'''

#         hls = half_life_loss(eigenvalues, B, coeffs, g_A)
#         HLs_calc.append(hls)
        
#         total_cost += tf.constant(weight,dtype=tf.float64)*tf.reduce_sum((tf.math.log(hls) - tf.math.log(HLs_true[idx])) ** 2)

#         count+=1
            
#     return total_cost, Lor, Lor_true, x, HLs_calc, B, eigenvalues


'''
Changed definition of cost function for strength
'''
# helper: trapezoidal integral in TF (float64)
def tf_trapz(y, x):
    dx  = x[1:] - x[:-1]                       # (N-1,)
    avg = 0.5 * (y[:-1] + y[1:])               # (N-1,)
    return tf.reduce_sum(avg * dx)             # scalar (float64)


# =====================================================================
# Windowed-moment helpers (used by cost_function_windowed and parity plots)
# =====================================================================

def soft_window_mask_tf(x, E_lo, E_hi, tau):
    """Smooth sigmoid window over energy x. Output in (0,1) with width-tau edges."""
    x  = tf.cast(x,    tf.float64)
    lo = tf.cast(E_lo, tf.float64)
    hi = tf.cast(E_hi, tf.float64)
    t  = tf.cast(tau,  tf.float64)
    return tf.sigmoid((x - lo) / t) * tf.sigmoid((hi - x) / t)


def windowed_moments_spectrum_tf(x, y, E_lo, E_hi, tau):
    """(S_in, centroid, variance) of a continuous spectrum y(x) under a sigmoid window."""
    EPS  = tf.constant(1e-16, dtype=tf.float64)
    mask = soft_window_mask_tf(x, E_lo, E_hi, tau)
    ym   = y * mask
    S_in = tf_trapz(ym, x)
    mu   = tf_trapz(x * ym, x) / (S_in + EPS)
    var  = tf_trapz(tf.square(x - mu) * ym, x) / (S_in + EPS)
    return S_in, mu, var


def windowed_moments_poles_tf(E_poles, B, E_lo, E_hi, tau):
    """(S_in, centroid, variance) of discrete poles under a sigmoid window."""
    EPS = tf.constant(1e-16, dtype=tf.float64)
    w   = soft_window_mask_tf(E_poles, E_lo, E_hi, tau)
    Bw  = B * w
    S_in = tf.reduce_sum(Bw)
    mu   = tf.reduce_sum(E_poles * Bw) / (S_in + EPS)
    var  = tf.reduce_sum(tf.square(E_poles - mu) * Bw) / (S_in + EPS)
    return S_in, mu, var

# cost_function # Original
def cost_function(params, n, fmt_data, Lors_true, coeffs, g_A, weight, central_point, retain):

    # D_mod, S1_mod, S2_mod, v0_mod, eta, x1, x2, x3 = modified_DS(params, n)
    D_mod, S1_mod, S2_mod, v0_mod, v1_mod, v2_mod, eta, x1, x2, x3 = modified_DS(params, n)

    total_cost = tf.constant(0.0, dtype=tf.float64)
    # HLs_calc   = []
    count      = 0

    EPS_DEN = tf.constant(1e-16, dtype=tf.float64)  # protect against /0

    for idx, alpha in enumerate(fmt_data):
        delta_alpha = float(alpha[0]) - float(central_point[0])
        delta_beta  = float(alpha[1]) - float(central_point[1])

        M_true = D_mod + delta_alpha * S1_mod + delta_beta * S2_mod
        eigenvalues, eigenvectors = tf.linalg.eigh(M_true)

        # keep centered fraction of spectrum
        n_i    = eigenvalues.shape[0]
        k_keep = int(round(retain * n_i))
        k_keep = max(1, min(k_keep, n_i))
        left   = (n_i - k_keep) // 2
        right  = left + k_keep
        eigenvalues  = eigenvalues[left:right]
        eigenvectors = eigenvectors[:, left:right]

        # Build the dynamic transition vector
        v_dynamic = v0_mod + delta_alpha * v1_mod + delta_beta * v2_mod

        # projections and strengths (B >= 0)
        # projections = tf.linalg.matvec(tf.transpose(eigenvectors), v0_mod)
        projections = tf.linalg.matvec(tf.transpose(eigenvectors), v_dynamic)
        B = tf.square(projections)

        # mask eigenvalues outside window
        # mask = tf.cast((eigenvalues > -10) & (eigenvalues < 15), dtype=tf.float64)
        mask = tf.cast((eigenvalues > -5.0) & (eigenvalues < 30.0), dtype=tf.float64)
        B = B * mask

        # true spectrum (E grid x, values S_true)
        Lor_true = tf.constant(Lors_true[count][:, 1], dtype=tf.float64)
        x        = tf.constant(Lors_true[count][:, 0], dtype=tf.float64)

        # width(E; alpha) and predicted spectrum
        width = tf.sqrt(tf.square(eta) + tf.square(x1 + x2*float(alpha[0]) + x3*float(alpha[1])))
        Lor   = give_me_Lorentzian(x, eigenvalues, B, width)


        # # ---------- 1. Normalized Spectral L2 Loss ----------
        # diff = Lor - Lor_true

        # # Create an energy penalty weight that is massive near 0 MeV, and small at 30 MeV
        # # Adding a small epsilon like 1.0 to prevent division by zero
        # # energy_weight = 1.0 / (x + 1.0)
        # # numer_spec = tf_trapz(tf.square(diff)*energy_weight, x)          
        # # denom_spec = tf_trapz(tf.square(Lor_true)*energy_weight, x)      

        # numer_spec = tf_trapz(tf.square(diff), x)          
        # denom_spec = tf_trapz(tf.square(Lor_true), x)      
        # spec_loss = numer_spec / (denom_spec + EPS_DEN)
        
        # # ---------- 2. Energy-Weighted Sum Rules (Moments) ----------
        # # Zeroth moment (Total Strength)
        # m0_true = tf_trapz(Lor_true, x) + EPS_DEN
        # m0_pred = tf_trapz(Lor, x) + EPS_DEN
        
        # # Normalized First moment (Energy Centroid in MeV)
        # m1_true = tf_trapz(x * Lor_true, x) / m0_true
        # m1_pred = tf_trapz(x * Lor, x) / m0_pred
        
        # # Relative squared error for total strength
        # m0_penalty = tf.square((m0_pred - m0_true) / m0_true)
        
        # # Squared error for centroid location
        # m1_penalty = tf.square(m1_pred - m1_true)
        
        # # Combine losses: moments guide the placement, L2 shapes the peak
        # lambda_moments = tf.constant(1, dtype=tf.float64)        

        # total_cost += spec_loss + lambda_moments * (m0_penalty + m1_penalty)
        # total_cost += spec_loss

        # ---------- normalized L2(E) loss ----------
        diff = Lor - Lor_true
        numer = tf_trapz(tf.square(diff), x)          # ∫ (Ŝ - S)^2 dE
        denom = tf_trapz(tf.square(Lor_true), x)      # ∫ S^2 dE
        spec_loss = numer / (denom + EPS_DEN)
        # total_cost += spec_loss
        # ------------------------------------------

        # ---------- Cumulative (Wasserstein-like) Loss ----------
        # 1. Calculate the grid spacing (assuming uniform energy grid from pnFAM)
        dx = x[1] - x[0]
        
        # 2. Compute the cumulative strength functions (running integrals)
        cum_Lor = tf.math.cumsum(Lor) * dx
        cum_Lor_true = tf.math.cumsum(Lor_true) * dx
        
        # 3. Calculate the normalized L2 error of the CUMULATIVE distributions
        diff_cum = cum_Lor - cum_Lor_true
        numer = tf_trapz(tf.square(diff_cum), x)          # ∫ (I_pred - I_true)^2 dE
        denom = tf_trapz(tf.square(cum_Lor_true), x)      # ∫ (I_true)^2 dE
        
        cum_loss = numer / (denom + EPS_DEN)


        # # ----------- Testing new loss term: Total Strength Anchor -----------
        # lambda_cum = 1.0
        # lambda_m0  = 0.1

        # m0_true = cum_Lor_true[-1]
        # m0_pred = tf.reduce_sum(B)
        # m0_penalty = tf.square(m0_pred - m0_true) / (tf.square(m0_true) + EPS_DEN)

        # total_cost += spec_loss + lambda_cum * cum_loss + lambda_m0 * m0_penalty



        # total_cost += cum_loss

        # ---------- Combine the Losses ---------- It was working but testing other options as well. To go back to this, just uncomment and comment out the above.
        # Adjust the weight of the cumulative loss if needed.
        # A 1:1 ratio is a standard starting point.
        lambda_cum = 1.0 
        # total_cost += (lambda_cum * cum_loss)
        total_cost += (spec_loss + lambda_cum * cum_loss)
        # tf.print("Spectral Loss:", spec_loss, "Cumulative Loss:", cum_loss)
        # --------------------------------------------------------

        # # ---------- Updated Cumulative (Wasserstein-like) Loss ----------
        # # 1. Calculate the grid spacing
        # dx = x[1] - x[0]
        
        # # 2. Compute the cumulative strength functions
        # cum_Lor = tf.math.cumsum(Lor) * dx
        # cum_Lor_true = tf.math.cumsum(Lor_true) * dx
        
        # # 3. Calculate the normalized L2 error of the CUMULATIVE distributions
        # diff_cum = cum_Lor - cum_Lor_true
        # numer = tf_trapz(tf.square(diff_cum), x)          
        # denom = tf_trapz(tf.square(cum_Lor_true), x)      
        # spec_loss = numer / (denom + EPS_DEN)
        
        # # --- NEW: TOTAL STRENGTH (SUM RULE) ANCHOR ---
        # # Get the total strength of the true data (the last element of its integral)
        # true_total_strength = cum_Lor_true[-1]
        
        # # Get the total predicted strength (sum of the squared projections)
        # pred_total_strength = tf.reduce_sum(B)
        
        # # Create a massive penalty if the total strengths do not match
        # strength_penalty = tf.square(pred_total_strength - true_total_strength) / (tf.square(true_total_strength) + EPS_DEN)
        
        # # Add both to the total cost. Weight the strength penalty heavily.
        # # total_cost += (spec_loss + 10.0 * strength_penalty)
        # total_cost += strength_penalty
        # # --------------------------------------------------------

        # # ---------- half-life term (unchanged) ----------
        # hls = half_life_loss(eigenvalues, B, coeffs, g_A)
        # HLs_calc.append(hls)
        # total_cost += tf.constant(weight, dtype=tf.float64) * \
        #               tf.reduce_sum((tf.math.log(hls) - tf.math.log(HLs_true[idx])) ** 2)
        # # -----------------------------------------------

        count += 1

    return total_cost, Lor, Lor_true, x, B, eigenvalues


# =====================================================================
# Windowed-moment cost function (parallel to cost_function; not a replacement)
# Aligns emulator training with Bayesian-inference observables:
#   - in-window centroid  (matches gt_e_window centroid likelihood)
#   - in-window strength  (optional)
#   - in-window width     (optional)
# By default still includes spec_loss + lambda_cum * cum_loss (multi-task).
# Set include_spec_cum=False to train against moments alone.
#
# One emulator = one nucleus, so `window` is a single (E_lo, E_hi) tuple.
# =====================================================================
def cost_function_windowed(params, n, fmt_data, Lors_true, coeffs, g_A, weight,
                           central_point, retain,
                           window, tau=0.5,
                           lambda_cent=1.0, lambda_sfrac=0.0, lambda_width=0.0,
                           include_spec_cum=True, lambda_cum=1.0):
    """
    Windowed-moment loss. `window` is a single (E_lo, E_hi) tuple (MeV) applied
    to every sample since all samples correspond to the same nucleus.

    Unused args (coeffs, g_A, weight) are kept in the signature for drop-in
    compatibility with existing call sites.
    """
    D_mod, S1_mod, S2_mod, v0_mod, v1_mod, v2_mod, eta, x1, x2, x3 = modified_DS(params, n)

    E_lo, E_hi  = window
    window_size = tf.constant(E_hi - E_lo, dtype=tf.float64)

    total_cost = tf.constant(0.0, dtype=tf.float64)
    EPS_DEN    = tf.constant(1e-16, dtype=tf.float64)
    lam_cent   = tf.constant(lambda_cent,  dtype=tf.float64)
    lam_sfrac  = tf.constant(lambda_sfrac, dtype=tf.float64)
    lam_width  = tf.constant(lambda_width, dtype=tf.float64)
    lam_cum_tf = tf.constant(lambda_cum,   dtype=tf.float64)

    Lor = Lor_true = x = B = eigenvalues = None  # last-iteration copies returned for plotting

    for idx, alpha in enumerate(fmt_data):
        delta_alpha = float(alpha[0]) - float(central_point[0])
        delta_beta  = float(alpha[1]) - float(central_point[1])

        M_true = D_mod + delta_alpha * S1_mod + delta_beta * S2_mod
        eigenvalues, eigenvectors = tf.linalg.eigh(M_true)

        n_i    = eigenvalues.shape[0]
        k_keep = int(round(retain * n_i))
        k_keep = max(1, min(k_keep, n_i))
        left   = (n_i - k_keep) // 2
        right  = left + k_keep
        eigenvalues  = eigenvalues[left:right]
        eigenvectors = eigenvectors[:, left:right]

        v_dynamic   = v0_mod + delta_alpha * v1_mod + delta_beta * v2_mod
        projections = tf.linalg.matvec(tf.transpose(eigenvectors), v_dynamic)
        B = tf.square(projections)

        hard_mask = tf.cast((eigenvalues > -5.0) & (eigenvalues < 30.0), dtype=tf.float64)
        B = B * hard_mask

        Lor_true = tf.constant(Lors_true[idx][:, 1], dtype=tf.float64)
        x        = tf.constant(Lors_true[idx][:, 0], dtype=tf.float64)
        width    = tf.sqrt(tf.square(eta) + tf.square(x1 + x2*float(alpha[0]) + x3*float(alpha[1])))
        Lor      = give_me_Lorentzian(x, eigenvalues, B, width)

        if include_spec_cum:
            diff       = Lor - Lor_true
            numer      = tf_trapz(tf.square(diff), x)
            denom      = tf_trapz(tf.square(Lor_true), x)
            spec_loss  = numer / (denom + EPS_DEN)

            dx           = x[1] - x[0]
            cum_Lor      = tf.math.cumsum(Lor) * dx
            cum_Lor_true = tf.math.cumsum(Lor_true) * dx
            diff_cum     = cum_Lor - cum_Lor_true
            numer_cum    = tf_trapz(tf.square(diff_cum), x)
            denom_cum    = tf_trapz(tf.square(cum_Lor_true), x)
            cum_loss     = numer_cum / (denom_cum + EPS_DEN)

            total_cost += spec_loss + lam_cum_tf * cum_loss

        S_true, mu_true, var_true = windowed_moments_spectrum_tf(x, Lor_true, E_lo, E_hi, tau)
        S_pred, mu_pred, var_pred = windowed_moments_poles_tf(eigenvalues, B, E_lo, E_hi, tau)

        # Centroid match (normalized by window size so the penalty is O(1))
        cent_loss = tf.square((mu_pred - mu_true) / window_size)
        total_cost += lam_cent * cent_loss

        if lambda_sfrac > 0.0:
            S_tot_true = tf_trapz(Lor_true, x) + EPS_DEN
            S_tot_pred = tf.reduce_sum(B)      + EPS_DEN
            frac_true  = S_true / S_tot_true
            frac_pred  = S_pred / S_tot_pred
            total_cost += lam_sfrac * tf.square(frac_pred - frac_true)

        if lambda_width > 0.0:
            sig_true = tf.sqrt(tf.maximum(var_true, EPS_DEN))
            sig_pred = tf.sqrt(tf.maximum(var_pred, EPS_DEN))
            total_cost += lam_width * tf.square((sig_pred - sig_true) / window_size)

    return total_cost, Lor, Lor_true, x, B, eigenvalues


# # cost_function # Modified for 2nd-Order PMM
# def cost_function(params, n, fmt_data, Lors_true, coeffs, g_A, weight, central_point, retain): # Quadratic PMM

#     # Catch the 5 matrices and the single v0 vector
#     D_mod, S1_mod, S2_mod, S11_mod, S22_mod, S12_mod, v0_mod, eta, x1, x2, x3 = modified_DS(params, n)
    
#     total_cost = tf.constant(0.0, dtype=tf.float64)
#     # HLs_calc   = []
#     count      = 0

#     EPS_DEN = tf.constant(1e-16, dtype=tf.float64)  # protect against /0

#     for idx, alpha in enumerate(fmt_data):
#         delta_alpha = float(alpha[0]) - float(central_point[0])
#         delta_beta  = float(alpha[1]) - float(central_point[1])

#         # Apply the Second-Order Matrix Expansion
#         M_true = (D_mod 
#                   + delta_alpha * S1_mod 
#                   + delta_beta * S2_mod 
#                   + (delta_alpha**2) * S11_mod 
#                   + (delta_beta**2) * S22_mod 
#                   + (delta_alpha * delta_beta) * S12_mod)
        
#         eigenvalues, eigenvectors = tf.linalg.eigh(M_true)

#         # keep centered fraction of spectrum
#         n_i    = eigenvalues.shape[0]
#         k_keep = int(round(retain * n_i))
#         k_keep = max(1, min(k_keep, n_i))
#         left   = (n_i - k_keep) // 2
#         right  = left + k_keep
#         eigenvalues  = eigenvalues[left:right]
#         eigenvectors = eigenvectors[:, left:right]

#         # Build the dynamic transition vector
#         # v_dynamic = v0_mod + delta_alpha * v1_mod + delta_beta * v2_mod

#         # projections and strengths (B >= 0)
#         projections = tf.linalg.matvec(tf.transpose(eigenvectors), v0_mod)
#         # projections = tf.linalg.matvec(tf.transpose(eigenvectors), v_dynamic)
#         B = tf.square(projections)

#         # mask eigenvalues outside window
#         # mask = tf.cast((eigenvalues > -10) & (eigenvalues < 15), dtype=tf.float64)
#         mask = tf.cast((eigenvalues > -5.0) & (eigenvalues < 30.0), dtype=tf.float64)
#         B = B * mask

#         # true spectrum (E grid x, values S_true)
#         Lor_true = tf.constant(Lors_true[count][:, 1], dtype=tf.float64)
#         x        = tf.constant(Lors_true[count][:, 0], dtype=tf.float64)

#         # width(E; alpha) and predicted spectrum
#         width = tf.sqrt(tf.square(eta) + tf.square(x1 + x2*float(alpha[0]) + x3*float(alpha[1])))
#         Lor   = give_me_Lorentzian(x, eigenvalues, B, width)

#         # ---------- normalized L2(E) loss ----------
#         diff = Lor - Lor_true
#         numer = tf_trapz(tf.square(diff), x)          # ∫ (Ŝ - S)^2 dE
#         denom = tf_trapz(tf.square(Lor_true), x)      # ∫ S^2 dE
#         spec_loss = numer / (denom + EPS_DEN)
#         total_cost += spec_loss
#         # ------------------------------------------

#         # # ---------- half-life term (unchanged) ----------
#         # hls = half_life_loss(eigenvalues, B, coeffs, g_A)
#         # HLs_calc.append(hls)
#         # total_cost += tf.constant(weight, dtype=tf.float64) * \
#         #               tf.reduce_sum((tf.math.log(hls) - tf.math.log(HLs_true[idx])) ** 2)
#         # # -----------------------------------------------

#         count += 1

#     return total_cost, Lor, Lor_true, x, B, eigenvalues



# def cost_function_only_HL(params, n, fmt_data, HLs_true, central_point):
    
#     '''
#     params: tf.Variable
#     D_shape, S1_shape, S2_shape : int
#     alpha_values, beta_values: list
#     data_table: pd.DataFrame
    
#     calculates the cost function by subtracting only the half-lives !
    
#     '''


    
#     D_mod, S1_mod, S2_mod = modified_DS_only_HL(params, n)

    
#     total_cost = 0
#     HLs_calc = []
    

#     for idx, alpha in enumerate(fmt_data):

#         M_true = D_mod + (float(alpha[0]) - float(central_point[0])) * S1_mod \
#                        + (float(alpha[1]) - float(central_point[1])) * S2_mod
        
        

#         eigenvalues, eigenvectors = tf.linalg.eigh(M_true)


        
#         ''' Add half-lives to optimization as well'''
#         log_hls = eigenvalues[int(n/2)] #tf.reduce_sum((eigenvalues))
        
#         total_cost += (log_hls - np.log10(HLs_true[idx])) ** 2

        
#         # save the half lives for CV check
#         HLs_calc.append(10**log_hls)
            
#     return total_cost, HLs_calc

def cost_function_4d(params, n, fmt_data, Lors_true, coeffs, g_A, weight, central_point, retain):

    D_mod, S1_mod, S2_mod, S3_mod, S4_mod, v0_mod, v1_mod, v2_mod, v3_mod, v4_mod, eta, x_width = modified_DS_4d(params, n)

    total_cost = tf.constant(0.0, dtype=tf.float64)
    count = 0
    EPS_DEN = tf.constant(1e-16, dtype=tf.float64)

    for idx, theta in enumerate(fmt_data):
        # Deltas from centroid
        dt1 = float(theta[0]) - float(central_point[0])
        dt2 = float(theta[1]) - float(central_point[1])
        dt3 = float(theta[2]) - float(central_point[2])
        dt4 = float(theta[3]) - float(central_point[3])

        # Assemble M and diagonalize
        M_true = D_mod + dt1 * S1_mod + dt2 * S2_mod + dt3 * S3_mod + dt4 * S4_mod
        eigenvalues, eigenvectors = tf.linalg.eigh(M_true)

        # Truncate basis
        n_i = eigenvalues.shape[0]
        k_keep = int(round(retain * n_i))
        k_keep = max(1, min(k_keep, n_i))
        left = (n_i - k_keep) // 2
        right = left + k_keep
        eigenvalues = eigenvalues[left:right]
        eigenvectors = eigenvectors[:, left:right]

        # Dynamic transition vector and strengths
        v_dynamic = v0_mod + dt1 * v1_mod + dt2 * v2_mod + dt3 * v3_mod + dt4 * v4_mod
        projections = tf.linalg.matvec(tf.transpose(eigenvectors), v_dynamic)
        B = tf.square(projections)

        # Apply physical energy mask
        mask = tf.cast((eigenvalues > -5.0) & (eigenvalues < 30.0), dtype=tf.float64)
        B = B * mask

        Lor_true = tf.constant(Lors_true[count][:, 1], dtype=tf.float64)
        x = tf.constant(Lors_true[count][:, 0], dtype=tf.float64)

        # Dynamic width
        width_arg = x_width[0] + x_width[1]*float(theta[0]) + x_width[2]*float(theta[1]) + x_width[3]*float(theta[2]) + x_width[4]*float(theta[3])
        width = tf.sqrt(tf.square(eta) + tf.square(width_arg))
        
        Lor = give_me_Lorentzian(x, eigenvalues, B, width)

        # Spectral L2 Loss
        diff = Lor - Lor_true
        numer = tf_trapz(tf.square(diff), x)
        denom = tf_trapz(tf.square(Lor_true), x)
        spec_loss = numer / (denom + EPS_DEN)

        # Cumulative Loss
        dx = x[1] - x[0]
        cum_Lor = tf.math.cumsum(Lor) * dx
        cum_Lor_true = tf.math.cumsum(Lor_true) * dx
        diff_cum = cum_Lor - cum_Lor_true
        numer_cum = tf_trapz(tf.square(diff_cum), x)
        denom_cum = tf_trapz(tf.square(cum_Lor_true), x)
        cum_loss = numer_cum / (denom_cum + EPS_DEN)

        lambda_cum = 1.0 
        total_cost += (spec_loss + lambda_cum * cum_loss)

        count += 1

    return total_cost, Lor, Lor_true, x, B, eigenvalues


def cost_function_only_HL(params, n, fmt_data, HLs_true, central_point):
    '''
    calculates the cost function by subtracting only the half-lives!
    '''
    D_mod, S1_mod, S2_mod = modified_DS_only_HL(params, n)
    
    total_cost = tf.constant(0.0, dtype=tf.float64)
    HLs_calc = []
    
    # Explicitly cast central point to float64 tensors
    cp_0 = tf.constant(central_point[0], dtype=tf.float64)
    cp_1 = tf.constant(central_point[1], dtype=tf.float64)

    for idx, alpha in enumerate(fmt_data):
        # Explicitly cast incoming parameters to float64 to prevent Type Confusion in M_true
        alpha_0 = tf.constant(alpha[0], dtype=tf.float64)
        alpha_1 = tf.constant(alpha[1], dtype=tf.float64)
        
        M_true = D_mod + (alpha_0 - cp_0) * S1_mod + (alpha_1 - cp_1) * S2_mod
        
        # Matrix is now purely tf.float64, safe for eigh
        eigenvalues, eigenvectors = tf.linalg.eigh(M_true)

        # The model uses the middle eigenvalue as the predicted log10(Half-life)
        log_hls = eigenvalues[int(n/2)] 
        
        # Explicit cast of the true target to float64
        true_log_hl = tf.constant(HLs_true[idx], dtype=tf.float64)
        
        # Add the squared error to the total cost
        total_cost += tf.square(log_hls - true_log_hl)
        
        # Save the calculated half-life (convert from log10 back to linear)
        HLs_calc.append(tf.pow(tf.constant(10.0, dtype=tf.float64), log_hls))

    return total_cost, HLs_calc


def ao_centroid_param_count(n, r):
    """Total parameter count for the AO-PMM centroid emulator.

    Layout: D diag (n) + S1 upper-tri (n(n+1)/2) + S2 upper-tri (n(n+1)/2)
    + r(r+1)/2 secondary real-symmetric matrices (each n(n+1)/2 entries)
    + 1 scalar bias g.
    """
    base = n + n * (n + 1)                     # D + S1 + S2
    num_sec = r * (r + 1) // 2
    mat_size = n * (n + 1) // 2
    return base + num_sec * mat_size + 1


def ao_pmm_param_count(n, r):
    """Total parameter count for scalar AO-PMM emulators."""
    return ao_centroid_param_count(n, r)


def _sym_from_upper_vec(vec, n):
    """Build a real-symmetric n×n matrix from n(n+1)/2 upper-triangle entries."""
    M = tf.zeros((n, n), dtype=tf.float64)
    upper_idx = np.triu_indices(n)
    idx_const = tf.constant(list(zip(upper_idx[0], upper_idx[1])))
    M = tf.tensor_scatter_nd_update(M, idx_const, vec)
    M = M + tf.linalg.band_part(tf.transpose(M), -1, 0) - tf.linalg.diag(tf.linalg.diag_part(M))
    return M


def cost_function_only_HL_ao(params, n, fmt_data, HLs_true, central_point, r=3):
    '''
    AO-PMM half-life emulator.

    The AO output is interpreted as log10(HL), matching cost_function_only_HL.
    The second return value is converted back to half-life seconds for parity
    plots and CV residual files.
    '''
    D_mod, S1_mod, S2_mod = modified_DS_only_HL(params, n)

    base = n + n * (n + 1)
    mat_size = n * (n + 1) // 2
    num_sec = r * (r + 1) // 2

    deltas = []
    offset = base
    for _ in range(num_sec):
        deltas.append(_sym_from_upper_vec(params[offset:offset + mat_size], n))
        offset += mat_size
    g_bias = params[offset]

    pairs = [(i, j) for i in range(r) for j in range(i, r)]

    fixed_bias = tf.constant(0.0, dtype=tf.float64)
    for D_ij in deltas:
        eigs = tf.linalg.eigvalsh(D_ij)
        fixed_bias += 0.5 * tf.square(tf.reduce_max(tf.abs(eigs)))

    total_cost = tf.constant(0.0, dtype=tf.float64)
    HLs_calc = []

    cp_0 = tf.constant(central_point[0], dtype=tf.float64)
    cp_1 = tf.constant(central_point[1], dtype=tf.float64)

    lo = n // 2 - r // 2

    for idx, alpha in enumerate(fmt_data):
        alpha_0 = tf.constant(alpha[0], dtype=tf.float64)
        alpha_1 = tf.constant(alpha[1], dtype=tf.float64)

        M_true = D_mod + (alpha_0 - cp_0) * S1_mod + (alpha_1 - cp_1) * S2_mod
        _eigvals, eigvecs = tf.linalg.eigh(M_true)
        v = [eigvecs[:, lo + k] for k in range(r)]

        bilinear_sum = tf.constant(0.0, dtype=tf.float64)
        for k_idx, (i, j) in enumerate(pairs):
            amp = tf.tensordot(v[i], tf.linalg.matvec(deltas[k_idx], v[j]), axes=1)
            bilinear_sum += tf.square(amp)

        log_hl_pred = g_bias + bilinear_sum - fixed_bias
        true_log_hl = tf.constant(HLs_true[idx], dtype=tf.float64)

        total_cost += tf.square(log_hl_pred - true_log_hl)
        HLs_calc.append(tf.pow(tf.constant(10.0, dtype=tf.float64), log_hl_pred))

    return total_cost, HLs_calc


def cost_function_only_centroid_ao(params, n, fmt_data, centroids_true, central_point, r=3):
    '''
    AO-PMM centroid emulator (paper Eq. (10), real-valued).

    Primary matrix: M = D + (alpha-cp_0) S1 + (beta-cp_1) S2 (same as middle-eig form).
    Output:  z = g + sum_{i<=j} |v_i^T Delta_ij v_j|^2  -  0.5 sum_{i<=j} ||Delta_ij||_2^2
    where {v_0,...,v_{r-1}} are r eigenvectors of M centered around index n//2 and
    ||.||_2 is the spectral norm (= max|eigenvalue| for real-symmetric Delta).

    The r(r+1)/2 Delta matrices are real-symmetric; an additional scalar bias g
    is trained. See ao_centroid_param_count(n, r) for the required param length.
    '''
    D_mod, S1_mod, S2_mod = modified_DS_only_HL(params, n)

    base = n + n * (n + 1)
    mat_size = n * (n + 1) // 2
    num_sec = r * (r + 1) // 2

    deltas = []
    offset = base
    for _ in range(num_sec):
        deltas.append(_sym_from_upper_vec(params[offset:offset + mat_size], n))
        offset += mat_size
    g_bias = params[offset]

    # (i, j) pairs with i <= j, in the same order used to unpack deltas.
    pairs = [(i, j) for i in range(r) for j in range(i, r)]

    # Fixed bias: 0.5 * sum ||Delta||_2^2 (spectral norm squared).
    fixed_bias = tf.constant(0.0, dtype=tf.float64)
    for D_ij in deltas:
        eigs = tf.linalg.eigvalsh(D_ij)
        fixed_bias += 0.5 * tf.square(tf.reduce_max(tf.abs(eigs)))

    total_cost = tf.constant(0.0, dtype=tf.float64)
    centroids_calc = []

    cp_0 = tf.constant(central_point[0], dtype=tf.float64)
    cp_1 = tf.constant(central_point[1], dtype=tf.float64)

    lo = n // 2 - r // 2  # starting index of the central r eigenvectors

    for idx, alpha in enumerate(fmt_data):
        alpha_0 = tf.constant(alpha[0], dtype=tf.float64)
        alpha_1 = tf.constant(alpha[1], dtype=tf.float64)

        M_true = D_mod + (alpha_0 - cp_0) * S1_mod + (alpha_1 - cp_1) * S2_mod
        _eigvals, eigvecs = tf.linalg.eigh(M_true)
        v = [eigvecs[:, lo + k] for k in range(r)]

        bilinear_sum = tf.constant(0.0, dtype=tf.float64)
        for k_idx, (i, j) in enumerate(pairs):
            amp = tf.tensordot(v[i], tf.linalg.matvec(deltas[k_idx], v[j]), axes=1)
            bilinear_sum += tf.square(amp)

        centroid_pred = g_bias + bilinear_sum - fixed_bias

        true_centroid = tf.constant(centroids_true[idx], dtype=tf.float64)
        total_cost += tf.square(centroid_pred - true_centroid)
        centroids_calc.append(centroid_pred)

    return total_cost, centroids_calc


def cost_function_only_peak_ao(params, n, fmt_data, peaks_true, central_point, r=3):
    '''
    AO-PMM peak-location emulator (paper Eq. (10), real-valued). Structurally
    identical to cost_function_only_centroid_ao; the target is the GT-resonance
    peak energy (MeV). Parameter layout and count match ao_centroid_param_count.
    '''
    D_mod, S1_mod, S2_mod = modified_DS_only_HL(params, n)

    base = n + n * (n + 1)
    mat_size = n * (n + 1) // 2
    num_sec = r * (r + 1) // 2

    deltas = []
    offset = base
    for _ in range(num_sec):
        deltas.append(_sym_from_upper_vec(params[offset:offset + mat_size], n))
        offset += mat_size
    g_bias = params[offset]

    pairs = [(i, j) for i in range(r) for j in range(i, r)]

    fixed_bias = tf.constant(0.0, dtype=tf.float64)
    for D_ij in deltas:
        eigs = tf.linalg.eigvalsh(D_ij)
        fixed_bias += 0.5 * tf.square(tf.reduce_max(tf.abs(eigs)))

    total_cost = tf.constant(0.0, dtype=tf.float64)
    peaks_calc = []

    cp_0 = tf.constant(central_point[0], dtype=tf.float64)
    cp_1 = tf.constant(central_point[1], dtype=tf.float64)

    lo = n // 2 - r // 2

    for idx, alpha in enumerate(fmt_data):
        alpha_0 = tf.constant(alpha[0], dtype=tf.float64)
        alpha_1 = tf.constant(alpha[1], dtype=tf.float64)

        M_true = D_mod + (alpha_0 - cp_0) * S1_mod + (alpha_1 - cp_1) * S2_mod
        _eigvals, eigvecs = tf.linalg.eigh(M_true)
        v = [eigvecs[:, lo + k] for k in range(r)]

        bilinear_sum = tf.constant(0.0, dtype=tf.float64)
        for k_idx, (i, j) in enumerate(pairs):
            amp = tf.tensordot(v[i], tf.linalg.matvec(deltas[k_idx], v[j]), axes=1)
            bilinear_sum += tf.square(amp)

        peak_pred = g_bias + bilinear_sum - fixed_bias

        true_peak = tf.constant(peaks_true[idx], dtype=tf.float64)
        total_cost += tf.square(peak_pred - true_peak)
        peaks_calc.append(peak_pred)

    return total_cost, peaks_calc


def cost_function_only_peak(params, n, fmt_data, peaks_true, central_point):
    '''
    Peak-location emulator (AE form): middle eigenvalue of the PMM matrix is
    interpreted directly as the GT-resonance peak energy (MeV). Mirrors
    cost_function_only_centroid.
    '''
    D_mod, S1_mod, S2_mod = modified_DS_only_HL(params, n)

    total_cost = tf.constant(0.0, dtype=tf.float64)
    peaks_calc = []

    cp_0 = tf.constant(central_point[0], dtype=tf.float64)
    cp_1 = tf.constant(central_point[1], dtype=tf.float64)

    for idx, alpha in enumerate(fmt_data):
        alpha_0 = tf.constant(alpha[0], dtype=tf.float64)
        alpha_1 = tf.constant(alpha[1], dtype=tf.float64)

        M_true = D_mod + (alpha_0 - cp_0) * S1_mod + (alpha_1 - cp_1) * S2_mod
        eigenvalues, _ = tf.linalg.eigh(M_true)

        peak_pred = eigenvalues[int(n/2)]

        true_peak = tf.constant(peaks_true[idx], dtype=tf.float64)
        total_cost += tf.square(peak_pred - true_peak)
        peaks_calc.append(peak_pred)

    return total_cost, peaks_calc


def cost_function_only_centroid(params, n, fmt_data, centroids_true, central_point):
    '''
    Centroid-only emulator: mirrors cost_function_only_HL, but the middle
    eigenvalue is interpreted directly as the GT centroid (MeV) — no log10.
    Returns (total_cost, centroids_calc) where centroids_calc is a list of
    emulated centroids in MeV.
    '''
    D_mod, S1_mod, S2_mod = modified_DS_only_HL(params, n)

    total_cost = tf.constant(0.0, dtype=tf.float64)
    centroids_calc = []

    cp_0 = tf.constant(central_point[0], dtype=tf.float64)
    cp_1 = tf.constant(central_point[1], dtype=tf.float64)

    for idx, alpha in enumerate(fmt_data):
        alpha_0 = tf.constant(alpha[0], dtype=tf.float64)
        alpha_1 = tf.constant(alpha[1], dtype=tf.float64)

        M_true = D_mod + (alpha_0 - cp_0) * S1_mod + (alpha_1 - cp_1) * S2_mod

        eigenvalues, _ = tf.linalg.eigh(M_true)

        # Middle eigenvalue is the predicted centroid (MeV).
        centroid_pred = eigenvalues[int(n/2)]

        true_centroid = tf.constant(centroids_true[idx], dtype=tf.float64)

        total_cost += tf.square(centroid_pred - true_centroid)
        centroids_calc.append(centroid_pred)

    return total_cost, centroids_calc


def data_table_only_HL(fmt_data,coeffs, g_A, nucnam):
    '''
    
    Here split the dataset from "beta_decay_data" folder
    into: training set, validation set and test set
    use any ratio you like (e.g. 0.8 0.1 0.1)
    
    For the optimizatio use only training set, and after you finish
    test it on validation set
    
    returns also number of QRPA poles n_QRPA
    
    '''

    HLs = []
    

    for frmt in fmt_data:
        
        alpha = frmt[0]
        beta = frmt[1]

        # now calculate half-lives the old way
        file = np.loadtxt('beta_decay_data_'+nucnam+'/excm_'+nucnam+'_'+beta+'_'+alpha+'.out')
        file = file[file[:,0]<del_nH]
        file = file[file[:,0]>-10]
        HLs.append(half_life_loss(file[:,0], file[:,1],coeffs, g_A))

     
    return HLs


def modified_DS_only_HL(params, n):
    '''

     
     added S1_shape & S2_shape 
     
     params: tf.Variable
     D_shape: int, shape of diagonal matrix
     S1_shape : int
     S2_shape : int
     
     given params, construct D, S1 and S2 matrices , 

    
    '''
    D_shape = (n,n)
    S1_shape = (n,n)
    S2_shape = (n,n)
    
    # initialize D, S1 and S2
    D_mod = tf.linalg.diag(params[:D_shape[0]])
    S1_mod = tf.zeros(S1_shape, dtype=tf.float64)
    S2_mod = tf.zeros(S2_shape, dtype=tf.float64)
    
    # construct S1 and S2 matrices
    upper_tri_indices1 = np.triu_indices(S1_shape[0])
    indices1 = tf.constant(list(zip(upper_tri_indices1[0], upper_tri_indices1[1])))
    S1_mod = tf.tensor_scatter_nd_update(S1_mod, indices1,\
            params[D_shape[0]:D_shape[0] + len(upper_tri_indices1[0])])
    
    S1_mod = S1_mod + tf.linalg.band_part(tf.transpose(S1_mod), -1, 0) - tf.linalg.diag(tf.linalg.diag_part(S1_mod))
    
    
    upper_tri_indices2 = np.triu_indices(S2_shape[0])
    indices2 = tf.constant(list(zip(upper_tri_indices2[0], upper_tri_indices2[1])))
    S2_mod = tf.tensor_scatter_nd_update(S2_mod, indices2 \
    , params[D_shape[0] + len(upper_tri_indices1[0]):D_shape[0] \
        + len(upper_tri_indices1[0])+len(upper_tri_indices2[0])])
    
    S2_mod = S2_mod + tf.linalg.band_part(tf.transpose(S2_mod), -1, 0) - tf.linalg.diag(tf.linalg.diag_part(S2_mod))
    
    
    return D_mod, S1_mod, S2_mod


       # generalized_eigen for M_true(a)
def generalized_eigen(D, S1, S2, alpha):
    M_true = D + float(alpha[0]) * S1 + float(alpha[1]) * S2
    eigenvalues, eigenvectors = eigh(M_true)
    return np.real(eigenvalues), np.real(eigenvectors) 




def plot_Lorentzian_for_idx(idx, test_set,n,params, coeffs, g_A):

    alpha = float(test_set[idx][0])
    beta = float(test_set[idx][1])
    
    
    Lors_test, HLs_test = data_table(test_set, coeffs, g_A)
    Lors_orig = Lors_test[idx]
    
    opt_D, opt_S1, opt_S2, opt_v0 = modified_DS(params, n)
    opt_eigenvalues, opt_eigenvectors = generalized_eigen(opt_D.numpy(), opt_S1.numpy(), opt_S2.numpy(), test_set[idx])
    opt_dot_products = [np.square(np.dot(opt_eigenvectors[:, i], opt_v0.numpy())) for i in range(opt_eigenvectors.shape[1])]
    
    
    
    fig, ax = plt.subplots()
    
    
    
    # plot the Lorentzian for the original data
    x = Lors_orig[:,0]
    opt_Lor = []
    for en in x:
        opt_Lor.append(give_me_Lorentzian(en,opt_eigenvalues,opt_dot_products,0.5))
    
    plt.plot(x, Lors_orig[:,1], 'b--',label='QRPA calculation')    
    plt.plot(x, opt_Lor, 'r-',label='emulated Lorentzian')
        
    
    ax.set_title(r'$V_0^{is}$ = '+str(round(alpha,1))+r', $g_0 = $'+str(round(beta,1)), size = 18)
    ax.legend(frameon = False)
    
    
    plt.xlabel(r'$\omega$ (MeV)', size = 18)
    plt.ylabel('$S$ (1/MeV)', size = 18)
    
    plt.annotate('${}^{74}$Ni', (0.2,0.7), xycoords='axes fraction', size = 22)

    plt.gca().tick_params(axis="y",direction="in", which = 'both', labelsize = 12)
    plt.gca().tick_params(axis="x",direction="in", which = 'both', labelsize = 12)
    
    plt.gca().yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    plt.gca().xaxis.set_minor_locator(ticker.MultipleLocator(0.25))
    
    plt.ylim(0)
    plt.xlim(-6,0.782)
    
    plt.savefig('gamow_teller_strength_emulator.pdf', bbox_inches='tight')
    
    
def data_Lorentzian_for_idx(idx, test_set,n,params, coeffs, g_A):

    
    
    Lors_test, HLs_test = data_table(test_set, coeffs, g_A)
    Lors_orig = Lors_test[idx]
    
    opt_D, opt_S1, opt_S2, opt_v0 = modified_DS(params, n)
    opt_eigenvalues, opt_eigenvectors = generalized_eigen(opt_D.numpy(), opt_S1.numpy(), opt_S2.numpy(), test_set[idx])
    opt_dot_products = [np.square(np.dot(opt_eigenvectors[:, i], opt_v0.numpy())) for i in range(opt_eigenvectors.shape[1])]
    
    
    
    fig, ax = plt.subplots()
    
    
    
    # plot the Lorentzian for the original data
    x = Lors_orig[:,0]
    opt_Lor = []
    for en in x:
        opt_Lor.append(give_me_Lorentzian(en,opt_eigenvalues,opt_dot_products,0.5))
    
    
    return x, Lors_orig[:,1], opt_Lor
    
 
# def plot_half_lives(test_set,params,n, coeffs, g_A, central_point, nucnam, retain):
#     '''
#     Calculate half-lives for type 1 alg
#     '''
#     hl_guess = []
#     times = []
    
#     Lors_test, HLs_test = data_table(test_set, coeffs, g_A, nucnam)
    
#     D_mod, S1_mod, S2_mod, v0_mod, eta, x1, x2, x3 = modified_DS(params, n)
    
    
#     for idx in range(len(test_set)):
        
#         start = time.time()  # Start time
        
#         M_true = D_mod + (float(test_set[idx][0])-float(central_point[0])) * S1_mod \
#             + (float(test_set[idx][1]) - float(central_point[1])) * S2_mod
        

#         eigenvalues, eigenvectors = tf.linalg.eigh(M_true)
        
#         n_i = eigenvalues.shape[0]
#         k_keep = int(round(retain * n_i))         # how many eigenvalues to keep
#         k_keep = max(1, min(k_keep, n_i))         # safety: clamp between 1 and n
        
#         left  = (n_i - k_keep) // 2               # starting index of the centered block
#         right = left + k_keep                     # ending index (exclusive)
        
#         eigenvalues  = eigenvalues[left:right]
#         eigenvectors = eigenvectors[:, left:right]
        
#         projections = tf.linalg.matvec(tf.transpose(eigenvectors), v0_mod)
        
#         # Square each projection
#         B = tf.square(projections)
        
#         mask = tf.cast((eigenvalues > -10) & (eigenvalues < 15), dtype=tf.float64)

#         # Apply the mask to zero out B where eigenvalue is negative
#         B = B * mask
        

#         hls = half_life_loss(eigenvalues, B, coeffs, g_A)
        
        
#         end = time.time()  # Start time
        
#         times.append(end-start)
#         hl_guess.append(hls)
        
        
#     return hl_guess, HLs_test, times


# def plot_half_lives_only_HL(test_set,params,n, coeffs, g_A, central_point, nucnam):
#     '''
#     Note that this function is for type 2 Alg in the paper
#     '''
#     hl_guess = []
#     times = []
    
#     HLs_test = data_table_only_HL(test_set,coeffs,g_A, nucnam)
    
#     D_mod, S1_mod, S2_mod = modified_DS_only_HL(params,n)
    
#     for idx in range(len(test_set)):
        
#         start = time.time()  # Start time
        
        
        
#         M_true = D_mod + (float(test_set[idx][0]) - float(central_point[0])) * S1_mod \
#                        + (float(test_set[idx][1]) - float(central_point[1])) * S2_mod
        
        

#         eigenvalues, eigenvectors = tf.linalg.eigh(M_true)


        
#         ''' Add half-lives to optimization as well'''
#         log_hls = eigenvalues[int(n/2)]#tf.reduce_sum(tf.square(eigenvalues))
        

        
#         end = time.time()  # Start time
        
#         times.append(end-start)
#         hl_guess.append(10**log_hls)
        
        
#     return hl_guess, HLs_test, times

def plot_half_lives(test_set, params, n, coeffs, g_A, central_point, nucnam, retain, *, reps=5, warmup=1):
    """
    Calculate half-lives for type 1 algorithm, with robust per-point timing.
    Returns: hl_guess (list[float]), HLs_test (as returned by data_table), times (list[float])
    """
    import time
    import numpy as np
    import tensorflow as tf

    def robust_time(fn, reps=20, warmup=1):
        # Warm-up (excluded)
        for _ in range(warmup):
            out = fn()
            try:
                _ = float(out.numpy())
            except Exception:
                _ = float(out)
        # Timed reps
        ts = []
        last_out = None
        for _ in range(reps):
            t0 = time.perf_counter()
            out = fn()
            try:
                last_out = float(out.numpy())
            except Exception:
                last_out = float(out)
            t1 = time.perf_counter()
            ts.append(t1 - t0)
        return float(np.median(ts)), last_out

    hl_guess = []
    times = []

    # Ground truth (unchanged)
    Lors_test, HLs_test = data_table(test_set, coeffs, g_A, nucnam)

    # Precompute constants once
    a0 = float(central_point[0])
    b0 = float(central_point[1])

    # Build model parts once (move inside the loop if you want to include build cost per point)
    D_mod, S1_mod, S2_mod, v0_mod, eta, x1, x2, x3 = modified_DS(params, n)

    for idx in range(len(test_set)):
        a = float(test_set[idx][0])
        b = float(test_set[idx][1])

        def eval_point():
            # Build matrix
            M_true = (D_mod
                      + (a - a0) * S1_mod
                      + (b - b0) * S2_mod)

            # Eigendecomposition
            eigenvalues, eigenvectors = tf.linalg.eigh(M_true)

            # Keep centered block (retain fraction)
            n_i = eigenvalues.shape[0]
            k_keep = int(round(retain * n_i))
            k_keep = max(1, min(k_keep, n_i))
            left  = (n_i - k_keep) // 2
            right = left + k_keep

            eigenvalues  = eigenvalues[left:right]
            eigenvectors = eigenvectors[:, left:right]

            # Projections and strengths
            projections = tf.linalg.matvec(tf.transpose(eigenvectors), v0_mod)
            B = tf.square(projections)

            # Mask eigenvalues (dtype-safe)
            mask = tf.cast((eigenvalues > -10) & (eigenvalues < 15), dtype=eigenvalues.dtype)
            B = B * mask

            # Half-life (may return tf.Tensor)
            return half_life_loss(eigenvalues, B, coeffs, g_A)

        med_time, hls_val = robust_time(eval_point, reps=reps, warmup=warmup)
        hl_guess.append(hls_val)   # numeric float
        times.append(med_time)

    return hl_guess, HLs_test, times


# =====================================================================
# Evaluation + parity-plot helpers (used by main_mod.py and eval notebooks)
# =====================================================================

def predict_strength_spectrum(params, n, alpha_beta, x_grid, central_point, retain):
    """
    Run the PMM emulator at one (alpha,beta) point.

    Returns (E_poles, B, Lor_on_grid) as numpy arrays.
    """
    D_mod, S1_mod, S2_mod, v0_mod, v1_mod, v2_mod, eta, x1, x2, x3 = modified_DS(params, n)
    delta_alpha = float(alpha_beta[0]) - float(central_point[0])
    delta_beta  = float(alpha_beta[1]) - float(central_point[1])

    M = D_mod + delta_alpha * S1_mod + delta_beta * S2_mod
    eigenvalues, eigenvectors = tf.linalg.eigh(M)
    n_i    = eigenvalues.shape[0]
    k_keep = max(1, min(int(round(retain * n_i)), n_i))
    left   = (n_i - k_keep) // 2
    right  = left + k_keep
    eigenvalues  = eigenvalues[left:right]
    eigenvectors = eigenvectors[:, left:right]

    v_dynamic   = v0_mod + delta_alpha * v1_mod + delta_beta * v2_mod
    projections = tf.linalg.matvec(tf.transpose(eigenvectors), v_dynamic)
    B = tf.square(projections)
    hard_mask = tf.cast((eigenvalues > -5.0) & (eigenvalues < 30.0), dtype=tf.float64)
    B = B * hard_mask

    x_tf  = tf.constant(x_grid, dtype=tf.float64)
    width = tf.sqrt(tf.square(eta) + tf.square(x1 + x2*float(alpha_beta[0]) + x3*float(alpha_beta[1])))
    Lor   = give_me_Lorentzian(x_tf, eigenvalues, B, width)
    return eigenvalues.numpy(), B.numpy(), Lor.numpy()


def peak_from_spectrum(E, S):
    """
    Global peak location of a broadened spectrum, with local parabolic
    interpolation around the grid maximum.
    """
    E = np.asarray(E, dtype=float)
    S = np.asarray(S, dtype=float)
    if E.size == 0 or S.size == 0:
        return float("nan")
    k = int(np.nanargmax(S))
    if k <= 0 or k >= len(S) - 1:
        return float(E[k])

    s_m, s_0, s_p = float(S[k - 1]), float(S[k]), float(S[k + 1])
    denom = s_m - 2.0 * s_0 + s_p
    if denom >= 0.0 or not np.isfinite(denom):
        return float(E[k])

    delta = 0.5 * (s_m - s_p) / denom
    dE = float(E[k + 1] - E[k])
    return float(E[k]) + delta * dE


def evaluate_peak_locations(params, n, fmt_data, Lors_true, central_point, retain):
    """
    Per-sample global peak locations (MeV) for emulator and true spectra.

    Both peaks are taken from broadened spectra on the same energy grid.
    """
    peak_pred, peak_true = [], []
    for idx, alpha in enumerate(fmt_data):
        x_grid = Lors_true[idx][:, 0]
        y_true = Lors_true[idx][:, 1]
        _E_poles, _B_vec, y_pred = predict_strength_spectrum(
            params, n, alpha, x_grid, central_point, retain)

        peak_pred.append(peak_from_spectrum(x_grid, y_pred))
        peak_true.append(peak_from_spectrum(x_grid, y_true))

    return np.array(peak_pred), np.array(peak_true)


def evaluate_windowed_moments(params, n, fmt_data, Lors_true, central_point, retain,
                              window, tau=0.5):
    """
    For each sample, compute predicted and true windowed moments.

    window : (E_lo, E_hi) tuple (one per nucleus / training run).

    Returns dict of numpy arrays (each length len(fmt_data)):
        S_pred, S_true         - in-window strength
        mu_pred, mu_true       - in-window centroid (MeV)
        sig_pred, sig_true     - in-window sqrt(variance) (MeV)
        S_tot_pred, S_tot_true - total strength (sum(B) vs trapz(S,x))
    """
    E_lo, E_hi = window

    S_p, S_t   = [], []
    mu_p, mu_t = [], []
    sg_p, sg_t = [], []
    Stot_p, Stot_t = [], []

    for idx, alpha in enumerate(fmt_data):
        x_grid = Lors_true[idx][:, 0]
        y_true = Lors_true[idx][:, 1]
        E_poles, B_vec, _ = predict_strength_spectrum(params, n, alpha, x_grid, central_point, retain)

        Sp, mup, varp = windowed_moments_poles_tf(
            tf.constant(E_poles, dtype=tf.float64),
            tf.constant(B_vec,   dtype=tf.float64),
            E_lo, E_hi, tau)
        St, mut, vart = windowed_moments_spectrum_tf(
            tf.constant(x_grid, dtype=tf.float64),
            tf.constant(y_true, dtype=tf.float64),
            E_lo, E_hi, tau)

        S_p.append(float(Sp.numpy()));   S_t.append(float(St.numpy()))
        mu_p.append(float(mup.numpy())); mu_t.append(float(mut.numpy()))
        sg_p.append(float(np.sqrt(max(float(varp.numpy()), 0.0))))
        sg_t.append(float(np.sqrt(max(float(vart.numpy()), 0.0))))
        Stot_p.append(float(np.sum(B_vec)))
        Stot_t.append(float(np.trapezoid(y_true, x_grid)))

    return {
        "S_pred":     np.array(S_p),   "S_true":     np.array(S_t),
        "mu_pred":    np.array(mu_p),  "mu_true":    np.array(mu_t),
        "sig_pred":   np.array(sg_p),  "sig_true":   np.array(sg_t),
        "S_tot_pred": np.array(Stot_p),"S_tot_true": np.array(Stot_t),
    }


def evaluate_half_lives_from_emulator(params, n, fmt_data, Lors_true,
                                      coeffs, g_A, central_point, retain):
    """
    Per-sample predicted and true half-lives (seconds).

    Predicted: half_life_loss applied to emulator poles (E, B).
    True:      half_life_loss applied to the true spectrum treated as discrete
               poles with strength S(E) * dE (trapezoidal equivalent on a
               uniform grid).
    """
    HL_pred, HL_true = [], []
    for idx, alpha in enumerate(fmt_data):
        x_grid = Lors_true[idx][:, 0]
        y_true = Lors_true[idx][:, 1]
        E_poles, B_vec, _ = predict_strength_spectrum(params, n, alpha, x_grid, central_point, retain)

        hl_p = half_life_loss(
            tf.constant(E_poles, dtype=tf.float64),
            tf.constant(B_vec,   dtype=tf.float64),
            coeffs, g_A)

        dE = float(x_grid[1] - x_grid[0])
        hl_t = half_life_loss(
            tf.constant(x_grid,      dtype=tf.float64),
            tf.constant(y_true * dE, dtype=tf.float64),
            coeffs, g_A)

        HL_pred.append(float(hl_p.numpy()))
        HL_true.append(float(hl_t.numpy()))

    return np.array(HL_pred), np.array(HL_true)


def parity_metrics(pred, true):
    """Mean relative error and max factor. NaN-safe; skips zero-true entries."""
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    nz   = np.abs(true) > 0
    rel  = np.full_like(pred, np.nan)
    rel[nz] = np.abs(pred[nz] - true[nz]) / np.abs(true[nz])
    mean_rel = float(np.nanmean(rel))
    both_pos = (pred > 0) & (true > 0)
    if np.any(both_pos):
        fac = np.maximum(pred[both_pos] / true[both_pos], true[both_pos] / pred[both_pos])
        max_fac = float(np.max(fac))
    else:
        max_fac = float('nan')
    return mean_rel, max_fac


def plot_parity(train_pred, train_true, val_pred, val_true,
                *, quantity, units, save_path,
                log_scale=False, title_suffix=""):
    """Parity plot for train + val with mean-relative-error / max-factor annotation."""
    train_pred = np.asarray(train_pred, dtype=float)
    train_true = np.asarray(train_true, dtype=float)
    val_pred   = np.asarray(val_pred,   dtype=float)
    val_true   = np.asarray(val_true,   dtype=float)

    tr_mrel, tr_fac = parity_metrics(train_pred, train_true)
    va_mrel, va_fac = parity_metrics(val_pred,   val_true)

    fig = plt.figure(figsize=(8, 7))
    ax  = fig.add_subplot(111)
    ax.scatter(train_pred, train_true, color='#1f77b4', marker='o', s=60,
               alpha=0.7, edgecolor='k', label='Training',   zorder=5)
    ax.scatter(val_pred,   val_true,   color='#ff7f0e', marker='s', s=60,
               alpha=0.9, edgecolor='k', label='Validation', zorder=6)

    all_vals = np.concatenate([train_true, val_true, train_pred, val_pred])
    finite_vals = all_vals[np.isfinite(all_vals)]
    n_bad = all_vals.size - finite_vals.size
    if n_bad > 0:
        print(f"[plot_parity:{quantity}] WARNING: {n_bad}/{all_vals.size} "
              f"values are non-finite and will be ignored for axis limits.")
    if finite_vals.size == 0:
        # Nothing plottable; emit empty figure with a note and return.
        ax.text(0.5, 0.5, f"No finite {quantity} values to plot",
                transform=ax.transAxes, ha='center', va='center', fontsize=12)
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        return {"train_mean_rel": float('nan'), "train_max_fac": float('nan'),
                "val_mean_rel":   float('nan'), "val_max_fac":   float('nan')}
    if log_scale:
        positive = finite_vals[finite_vals > 0]
        lo = float(np.min(positive)) * 0.5 if positive.size else 1e-3
        hi = float(np.max(positive)) * 2.0 if positive.size else 1.0
        ax.set_xscale('log'); ax.set_yscale('log')
    else:
        span = float(np.max(finite_vals) - np.min(finite_vals))
        lo = float(np.min(finite_vals)) - 0.05*span
        hi = float(np.max(finite_vals)) + 0.05*span

    ax.plot([lo, hi], [lo, hi], 'r--', label='Ideal', zorder=1)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(f'Emulated {quantity} ({units})', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'True QRPA {quantity} ({units})', fontsize=12, fontweight='bold')
    ax.set_title(f'SMLR Parity - {quantity}{title_suffix}', fontsize=13)
    ax.grid(True, which='major', ls='-',  alpha=0.4)
    ax.grid(True, which='minor', ls='--', alpha=0.2)
    ax.legend(loc='upper left', fontsize=11)

    textstr = '\n'.join((
        r'$\bf{Training}$',
        f'Mean rel err: {tr_mrel*100:.2f}%',
        f'Max factor:   {tr_fac:.2f}x',
        '',
        r'$\bf{Validation}$',
        f'Mean rel err: {va_mrel*100:.2f}%',
        f'Max factor:   {va_fac:.2f}x',
    ))
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray')
    ax.text(0.65, 0.25, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props, zorder=10)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)

    return {"train_mean_rel": tr_mrel, "train_max_fac": tr_fac,
            "val_mean_rel":   va_mrel, "val_max_fac":   va_fac}


def plot_half_lives_only_HL(test_set, params, n, coeffs, g_A, central_point, nucnam, *, reps=5, warmup=1):
    """
    Type 2 algorithm (only HL) with robust per-point timing.
    Returns: hl_guess (list[float]), HLs_test, times (list[float])
    """
    import time
    import numpy as np
    import tensorflow as tf

    def robust_time(fn, reps=20, warmup=1):
        # Warm-up (excluded)
        for _ in range(warmup):
            out = fn()
            try:
                _ = float(out.numpy())
            except Exception:
                _ = float(out)
        # Timed reps
        ts = []
        last_out = None
        for _ in range(reps):
            t0 = time.perf_counter()
            out = fn()
            try:
                last_out = float(out.numpy())
            except Exception:
                last_out = float(out)
            t1 = time.perf_counter()
            ts.append(t1 - t0)
        return float(np.median(ts)), last_out

    hl_guess = []
    times = []

    # Ground truth (unchanged)
    # HLs_test = data_table_only_HL(test_set, coeffs, g_A, nucnam)
    HLs_test = []

    # Precompute constants once
    a0 = float(central_point[0])
    b0 = float(central_point[1])

    # Build model parts once (move inside the loop if you want to include build cost per point)
    D_mod, S1_mod, S2_mod = modified_DS_only_HL(params, n)

    for idx in range(len(test_set)):
        a = float(test_set[idx][0])
        b = float(test_set[idx][1])

        def eval_point():
            M_true = (D_mod
                      + (a - a0) * S1_mod
                      + (b - b0) * S2_mod)

            eigenvalues, eigenvectors = tf.linalg.eigh(M_true)

            # Original behavior: take the middle eigenvalue (as log(T1/2))
            mid_idx = int(n/2)
            mid_idx = max(0, min(mid_idx, int(eigenvalues.shape[0]) - 1))
            log_hls = eigenvalues[mid_idx]

            # Return 10**log_hls (half-life)
            # Ensure we return a Tensor/number suitable for robust_time materialization
            try:
                return tf.pow(tf.constant(10.0, dtype=log_hls.dtype), log_hls)
            except Exception:
                # Fallback if dtype mismatch (shouldn't happen)
                return 10.0 ** log_hls

        med_time, hls_val = robust_time(eval_point, reps=reps, warmup=warmup)
        hl_guess.append(hls_val)   # numeric float
        times.append(med_time)

    return hl_guess, HLs_test, times


# =====================================================================
# Exponential-PMM HL emulator (additive variant; does not modify the AE
# or AO code paths above).
#
# Matrix ansatz, modelled after Jin et al., Phys. Rev. Research 7, 043347
# (2025), Eq. (22):
#
#   M(a, b) = D
#           + (a - a0) * S1
#           + (b - b0) * S2
#           + (a - a0) * exp( -(b - b0) * x1 ) * S3
#
# log10(HL) is predicted as the middle eigenvalue of M(a, b) -- same
# readout as cost_function_only_HL.
#
# Parameter packing (length = exp_pmm_param_count(n)):
#   [0 .. n)                            : D diagonal
#   [n .. n + n(n+1)/2)                 : S1 upper triangle
#   [n + n(n+1)/2 .. n + 2*n(n+1)/2)    : S2 upper triangle
#   [n + 2*n(n+1)/2 .. n + 3*n(n+1)/2)  : S3 upper triangle
#   [n + 3*n(n+1)/2]                    : x1 (scalar)
# =====================================================================


def exp_pmm_param_count(n):
    """Total trainable parameters for the exponential-PMM HL emulator."""
    return n + 3 * (n * (n + 1) // 2) + 1


def modified_DS_only_HL_exp(params, n):
    """Build (D, S1, S2, S3, x1) for the exponential-PMM HL emulator.

    Returns tf.float64 tensors.
    """
    mat_size = n * (n + 1) // 2

    D_mod = tf.linalg.diag(params[:n])

    upper = np.triu_indices(n)
    idx_const = tf.constant(list(zip(upper[0], upper[1])))

    offset = n
    sym_mats = []
    for _ in range(3):
        M = tf.zeros((n, n), dtype=tf.float64)
        M = tf.tensor_scatter_nd_update(M, idx_const,
                                        params[offset:offset + mat_size])
        M = M + tf.linalg.band_part(tf.transpose(M), -1, 0) - tf.linalg.diag(tf.linalg.diag_part(M))
        sym_mats.append(M)
        offset += mat_size

    S1_mod, S2_mod, S3_mod = sym_mats
    x1 = params[offset]

    return D_mod, S1_mod, S2_mod, S3_mod, x1


def cost_function_only_HL_exp(params, n, fmt_data, HLs_true, central_point):
    """Exponential-PMM half-life emulator.

        M_true   = D + (a-a0)*S1 + (b-b0)*S2 + (a-a0)*exp(-(b-b0)*x1)*S3
        log10HL  = eigenvalues(M_true)[n // 2]

    Returns (total_cost, [HL_predictions_in_seconds]). HLs_true is
    assumed to be log10(half-life), matching cost_function_only_HL.
    """
    D_mod, S1_mod, S2_mod, S3_mod, x1 = modified_DS_only_HL_exp(params, n)

    total_cost = tf.constant(0.0, dtype=tf.float64)
    HLs_calc = []

    cp_0 = tf.constant(central_point[0], dtype=tf.float64)
    cp_1 = tf.constant(central_point[1], dtype=tf.float64)
    mid_idx = int(n / 2)

    for idx, alpha in enumerate(fmt_data):
        alpha_0 = tf.constant(alpha[0], dtype=tf.float64)
        alpha_1 = tf.constant(alpha[1], dtype=tf.float64)

        da = alpha_0 - cp_0
        db = alpha_1 - cp_1

        M_true = (D_mod
                  + da * S1_mod
                  + db * S2_mod
                  + da * tf.exp(-db * x1) * S3_mod)

        eigenvalues, _ = tf.linalg.eigh(M_true)
        log_hl_pred = eigenvalues[mid_idx]

        true_log_hl = tf.constant(HLs_true[idx], dtype=tf.float64)
        total_cost += tf.square(log_hl_pred - true_log_hl)

        HLs_calc.append(tf.pow(tf.constant(10.0, dtype=tf.float64), log_hl_pred))

    return total_cost, HLs_calc


# =====================================================================
# Generalised symmetric exponential-PMM HL emulator (additive variant).
#
# Each parametric direction carries one extra exponentially-modulated
# symmetric matrix whose exponent is a general linear form in both
# (a - a0) and (b - b0):
#
#   M(a, b) = D
#           + (a - a0) * S1
#           + (b - b0) * S2
#           + (a - a0) * exp( -x1a*(a-a0) - x1b*(b-b0) ) * S3
#           + (b - b0) * exp( -x2a*(a-a0) - x2b*(b-b0) ) * S4
#
# Subsumes:
#  - Paper Eq. (22) form (x1a = x2b = 0)
#  - Self-modulated form (x1b = x2a = 0)
# Captures both inter-parameter coupling and per-direction saturation
# simultaneously.
#
# At the central point, (a-a0) = (b-b0) = 0, so all extra terms vanish
# and M(a0, b0) = D, preserving the self-consistency invariant of the
# AE / exp variants.
#
# Parameter packing (length = exp_gen_pmm_param_count(n)):
#   [0 .. n)                : D diagonal
#   next n(n+1)/2 entries   : S1 upper triangle
#   next n(n+1)/2 entries   : S2 upper triangle
#   next n(n+1)/2 entries   : S3 upper triangle
#   next n(n+1)/2 entries   : S4 upper triangle
#   then 4 scalars          : x1a, x1b, x2a, x2b
# =====================================================================


def exp_gen_pmm_param_count(n):
    """Total trainable parameters for the generalised exponential-PMM HL emulator."""
    return n + 4 * (n * (n + 1) // 2) + 4


def modified_DS_only_HL_exp_gen(params, n):
    """Build (D, S1, S2, S3, S4, x1a, x1b, x2a, x2b) for the generalised
    exponential-PMM HL emulator. Returns tf.float64 tensors.
    """
    mat_size = n * (n + 1) // 2

    D_mod = tf.linalg.diag(params[:n])

    upper = np.triu_indices(n)
    idx_const = tf.constant(list(zip(upper[0], upper[1])))

    offset = n
    sym_mats = []
    for _ in range(4):
        M = tf.zeros((n, n), dtype=tf.float64)
        M = tf.tensor_scatter_nd_update(M, idx_const,
                                        params[offset:offset + mat_size])
        M = M + tf.linalg.band_part(tf.transpose(M), -1, 0) - tf.linalg.diag(tf.linalg.diag_part(M))
        sym_mats.append(M)
        offset += mat_size

    S1_mod, S2_mod, S3_mod, S4_mod = sym_mats
    x1a = params[offset]
    x1b = params[offset + 1]
    x2a = params[offset + 2]
    x2b = params[offset + 3]

    return D_mod, S1_mod, S2_mod, S3_mod, S4_mod, x1a, x1b, x2a, x2b


def cost_function_only_HL_exp_gen(params, n, fmt_data, HLs_true, central_point):
    """Generalised exponential-PMM half-life emulator.

        M_true   = D + (a-a0)*S1 + (b-b0)*S2
                   + (a-a0)*exp(-x1a*(a-a0) - x1b*(b-b0))*S3
                   + (b-b0)*exp(-x2a*(a-a0) - x2b*(b-b0))*S4
        log10HL  = eigenvalues(M_true)[n // 2]

    Returns (total_cost, [HL_predictions_in_seconds]). HLs_true is
    assumed to be log10(half-life), matching cost_function_only_HL.
    """
    (D_mod, S1_mod, S2_mod, S3_mod, S4_mod,
     x1a, x1b, x2a, x2b) = modified_DS_only_HL_exp_gen(params, n)

    total_cost = tf.constant(0.0, dtype=tf.float64)
    HLs_calc = []

    cp_0 = tf.constant(central_point[0], dtype=tf.float64)
    cp_1 = tf.constant(central_point[1], dtype=tf.float64)
    mid_idx = int(n / 2)

    for idx, alpha in enumerate(fmt_data):
        alpha_0 = tf.constant(alpha[0], dtype=tf.float64)
        alpha_1 = tf.constant(alpha[1], dtype=tf.float64)

        da = alpha_0 - cp_0
        db = alpha_1 - cp_1

        M_true = (D_mod
                  + da * S1_mod
                  + db * S2_mod
                  + da * tf.exp(-x1a * da - x1b * db) * S3_mod
                  + db * tf.exp(-x2a * da - x2b * db) * S4_mod)

        eigenvalues, _ = tf.linalg.eigh(M_true)
        log_hl_pred = eigenvalues[mid_idx]

        true_log_hl = tf.constant(HLs_true[idx], dtype=tf.float64)
        total_cost += tf.square(log_hl_pred - true_log_hl)

        HLs_calc.append(tf.pow(tf.constant(10.0, dtype=tf.float64), log_hl_pred))

    return total_cost, HLs_calc


# =====================================================================
# Second-order (quadratic-in-(da, db)) PMM HL emulator. Additive
# variant; does not modify existing AE / AO / exp / exp_gen paths.
#
# Matrix ansatz:
#
#   M(a, b) = D
#           + (a - a0) * S1
#           + (b - b0) * S2
#           + (a - a0)**2     * S11
#           + (a - a0)*(b-b0) * S12
#           + (b - b0)**2     * S22
#
# Pure polynomial in (a, b) up to second order; matrix coefficients are
# all real-symmetric n x n. At the central point (a0, b0) every
# parametric term vanishes, so M(a0, b0) = D, preserving the
# self-consistency invariant of the linear / exp variants.
#
# Parameter packing (length = quad_pmm_param_count(n)):
#   [0 .. n)                : D diagonal
#   next n(n+1)/2 entries   : S1   upper triangle
#   next n(n+1)/2 entries   : S2   upper triangle
#   next n(n+1)/2 entries   : S11  upper triangle
#   next n(n+1)/2 entries   : S12  upper triangle
#   next n(n+1)/2 entries   : S22  upper triangle
# =====================================================================


def quad_pmm_param_count(n):
    """Total trainable parameters for the second-order PMM HL emulator."""
    return n + 5 * (n * (n + 1) // 2)


def modified_DS_only_HL_quad(params, n):
    """Build (D, S1, S2, S11, S12, S22) for the second-order PMM HL emulator.

    Returns tf.float64 tensors.
    """
    mat_size = n * (n + 1) // 2

    D_mod = tf.linalg.diag(params[:n])

    upper = np.triu_indices(n)
    idx_const = tf.constant(list(zip(upper[0], upper[1])))

    offset = n
    sym_mats = []
    for _ in range(5):
        M = tf.zeros((n, n), dtype=tf.float64)
        M = tf.tensor_scatter_nd_update(M, idx_const,
                                        params[offset:offset + mat_size])
        M = M + tf.linalg.band_part(tf.transpose(M), -1, 0) - tf.linalg.diag(tf.linalg.diag_part(M))
        sym_mats.append(M)
        offset += mat_size

    S1_mod, S2_mod, S11_mod, S12_mod, S22_mod = sym_mats
    return D_mod, S1_mod, S2_mod, S11_mod, S12_mod, S22_mod


def cost_function_only_HL_quad(params, n, fmt_data, HLs_true, central_point):
    """Second-order PMM half-life emulator.

        M_true   = D + (a-a0)*S1 + (b-b0)*S2
                   + (a-a0)**2 * S11
                   + (a-a0)*(b-b0) * S12
                   + (b-b0)**2 * S22
        log10HL  = eigenvalues(M_true)[n // 2]

    Returns (total_cost, [HL_predictions_in_seconds]). HLs_true is
    assumed to be log10(half-life), matching cost_function_only_HL.
    """
    (D_mod, S1_mod, S2_mod,
     S11_mod, S12_mod, S22_mod) = modified_DS_only_HL_quad(params, n)

    total_cost = tf.constant(0.0, dtype=tf.float64)
    HLs_calc = []

    cp_0 = tf.constant(central_point[0], dtype=tf.float64)
    cp_1 = tf.constant(central_point[1], dtype=tf.float64)
    mid_idx = int(n / 2)

    for idx, alpha in enumerate(fmt_data):
        alpha_0 = tf.constant(alpha[0], dtype=tf.float64)
        alpha_1 = tf.constant(alpha[1], dtype=tf.float64)

        da = alpha_0 - cp_0
        db = alpha_1 - cp_1

        M_true = (D_mod
                  + da * S1_mod
                  + db * S2_mod
                  + (da * da) * S11_mod
                  + (da * db) * S12_mod
                  + (db * db) * S22_mod)

        eigenvalues, _ = tf.linalg.eigh(M_true)
        log_hl_pred = eigenvalues[mid_idx]

        true_log_hl = tf.constant(HLs_true[idx], dtype=tf.float64)
        total_cost += tf.square(log_hl_pred - true_log_hl)

        HLs_calc.append(tf.pow(tf.constant(10.0, dtype=tf.float64), log_hl_pred))

    return total_cost, HLs_calc


# =====================================================================
# Weighted-selector PMM ("wsel") for HL — option L3
# ---------------------------------------------------------------------
# Same AE-linear matrix M(a, b) = D + (a-a0)*S1 + (b-b0)*S2, but instead
# of hard-selecting eigenvalue index n//2 the predictor is a region-
# dependent softmax-weighted sum of all n eigenvalues:
#
#   logits(a, b) = s + s_a*(a-a0) + s_b*(b-b0)        # length n
#   w(a, b)      = softmax(logits)                    # nonneg, sums to 1
#   log10HL_hat  = sum_i w_i(a, b) * lambda_i(M(a,b))
#
# Lets the emulator slide its eigenvalue selection across (a, b), which
# is the right inductive bias if the QRPA response has a level crossing
# / cusp inside the parameter box.
#
# Reduces to near-AE when s_a = s_b = 0 and s is sharply peaked at n//2.
#
# Parameter packing (length = wsel_pmm_param_count(n)):
#   [0 .. n)                : D diagonal
#   next n(n+1)/2 entries   : S1   upper triangle
#   next n(n+1)/2 entries   : S2   upper triangle
#   next n entries          : s     (base logits)
#   next n entries          : s_a   (alpha slope of logits)
#   next n entries          : s_b   (beta slope of logits)
# =====================================================================


def wsel_pmm_param_count(n):
    """Total trainable parameters for the weighted-selector PMM HL emulator."""
    return n + 2 * (n * (n + 1) // 2) + 3 * n


def modified_DS_only_HL_wsel(params, n):
    """Build (D, S1, S2, s, s_a, s_b) for the wsel PMM HL emulator.

    Returns tf.float64 tensors. D, S1, S2 are real-symmetric n x n.
    s, s_a, s_b are length-n vectors of softmax logits and slopes.
    """
    mat_size = n * (n + 1) // 2

    D_mod = tf.linalg.diag(params[:n])

    upper = np.triu_indices(n)
    idx_const = tf.constant(list(zip(upper[0], upper[1])))

    offset = n
    sym_mats = []
    for _ in range(2):
        M = tf.zeros((n, n), dtype=tf.float64)
        M = tf.tensor_scatter_nd_update(M, idx_const,
                                        params[offset:offset + mat_size])
        M = M + tf.linalg.band_part(tf.transpose(M), -1, 0) - tf.linalg.diag(tf.linalg.diag_part(M))
        sym_mats.append(M)
        offset += mat_size

    S1_mod, S2_mod = sym_mats

    s_base = params[offset:offset + n]
    offset += n
    s_a = params[offset:offset + n]
    offset += n
    s_b = params[offset:offset + n]
    offset += n

    return D_mod, S1_mod, S2_mod, s_base, s_a, s_b


def cost_function_only_HL_wsel(params, n, fmt_data, HLs_true, central_point):
    """Weighted-selector PMM half-life emulator.

        M_true       = D + (a-a0)*S1 + (b-b0)*S2
        logits(a,b)  = s + s_a*(a-a0) + s_b*(b-b0)
        w(a,b)       = softmax(logits)
        log10HL_hat  = sum_i w_i * eigenvalues(M_true)[i]

    Returns (total_cost, [HL_predictions_in_seconds]). HLs_true is
    assumed to be log10(half-life), matching cost_function_only_HL.
    """
    (D_mod, S1_mod, S2_mod,
     s_base, s_a, s_b) = modified_DS_only_HL_wsel(params, n)

    total_cost = tf.constant(0.0, dtype=tf.float64)
    HLs_calc = []

    cp_0 = tf.constant(central_point[0], dtype=tf.float64)
    cp_1 = tf.constant(central_point[1], dtype=tf.float64)

    for idx, alpha in enumerate(fmt_data):
        alpha_0 = tf.constant(alpha[0], dtype=tf.float64)
        alpha_1 = tf.constant(alpha[1], dtype=tf.float64)

        da = alpha_0 - cp_0
        db = alpha_1 - cp_1

        M_true = D_mod + da * S1_mod + db * S2_mod

        eigenvalues = tf.linalg.eigvalsh(M_true)

        logits = s_base + s_a * da + s_b * db
        weights = tf.nn.softmax(logits)

        log_hl_pred = tf.reduce_sum(weights * eigenvalues)

        true_log_hl = tf.constant(HLs_true[idx], dtype=tf.float64)
        total_cost += tf.square(log_hl_pred - true_log_hl)

        HLs_calc.append(tf.pow(tf.constant(10.0, dtype=tf.float64), log_hl_pred))

    return total_cost, HLs_calc
