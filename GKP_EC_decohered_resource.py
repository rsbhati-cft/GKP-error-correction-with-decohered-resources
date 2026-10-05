#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 16:20:21 2026

@author: rajendrabhati
"""

import numpy as np
import qutip as qt
from qutip import *
from scipy.special import eval_hermite, factorial
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib import cm
import time

# ============================================================
# Parameters
# ============================================================

N = 50                  # Fock cutoff
r = 1.15                # peak squeezing         # envelope parameter
beta = 0.1             # damping 
Smax = 8                # number of lattice peaks
alpha = np.sqrt(np.pi)  # GKP displacement unit
xvec = np.linspace(-3, 3, 200)


# ============================================================
# Basic operators
# ============================================================

a = qt.destroy(N)
I = qt.qeye(N)


def D(alpha):
    """Displacement operator."""
    return qt.displace(N, alpha)


def S(r):
    """Single-mode squeezing operator."""
    return qt.squeeze(N, r)


# ============================================================
# Finite-energy GKP states
# ============================================================

def gkp_state(mu):
    """
    Approximate finite-energy GKP logical state |mu_L>, mu=0,1.

    Constructed as a Gaussian-weighted lattice of displaced
    squeezed vacua.
    """
    Delta = np.sqrt(np.tanh(beta))
    r = 0.5 * np.log(1 / np.tanh(beta))
    
    psi = 0 * qt.basis(N, 0)

    for s in range(-Smax, Smax + 1):
        n = 2*s + mu

        # Gaussian envelope
        w = np.exp(-2*np.pi*Delta**2 * (s + mu/2)**2)

        # Position displacement: alpha = q/sqrt(2)
        d = n * np.sqrt(np.pi)

        psi += w * D(d / np.sqrt(2)) * S(r) * qt.basis(N, 0)

    return psi.unit()




def gkp_basis():
    """Return |0_L>, |1_L>."""
    return gkp_state(0), gkp_state(1)

def gkp_6states():
    g0, g1 = gkp_state(0), gkp_state(1)
    g_plus = (g0 + g1).unit()
    g_minus = (g0 - g1).unit()
    g_up = (g0 + 1j * g1).unit()
    g_down = (g0 - 1j * g1).unit()
    return g0, g1, g_plus, g_minus, g_up, g_down
# ============================================================
# Gaussian pdf
# ============================================================

def gaussian_pdf(r, sigma):
    """Zero-mean Gaussian probability density."""
    return np.exp(-r**2 / (2*sigma**2)) / (np.sqrt(2*np.pi)*sigma)


# ============================================================
# GKP Bell state
# ============================================================

def gkp_bell():
    """
    Finite-energy GKP Bell state

        |Phi_GKP> = (|00> + |11>)/sqrt(2).
    """
    g0, g1 = gkp_basis()

    bell = (qt.tensor(g0, g0) +
            qt.tensor(g1, g1)).unit()

    return bell

# ============================================================
# Gaussian-random displacement channel
# ============================================================


def gaussian_displacement_channel_two_mode(
        rho,
        sigma,
        rmax=5.0,
        nr=101):
    """
    Apply the same Gaussian-random displacement D(r) to
    both modes:

        rho -> ∫dr G(r) [D(r)⊗D(r)] rho [D†(r)⊗D†(r)]

    Parameters
    ----------
    rho : Qobj
        Two-mode density matrix.
    sigma : float
        Standard deviation of the displacement.
    rmax : float
        Integration range [-rmax*sigma, rmax*sigma].
    nr : int
        Number of integration points.

    Returns
    -------
    rho_out : Qobj
        Two-mode output density matrix.
    """

    r_values = np.linspace(-rmax*sigma, rmax*sigma, nr)

    rho_out = qt.Qobj(
        np.zeros((N**2, N**2), dtype=complex),
        dims=rho.dims
    )

    for r in r_values:

        # Same displacement on both modes
        U = qt.tensor(D(r), D(r))

        rho_r = U * rho * U.dag()

        rho_out += gaussian_pdf(r, sigma) * rho_r

    # Numerical integration
    rho_out = rho_out * (r_values[1] - r_values[0])

    # Numerical cleanup
    rho_out = (rho_out + rho_out.dag()) / 2
    rho_out = rho_out / rho_out.tr()

    return rho_out


# ============================================================
# Two-mode EPR state
# ============================================================

def epr_state(r_epr=2.0):
    """
    Finite-squeezing EPR state.

    Generated directly as a two-mode squeezed vacuum;
    no beam splitter/homodyne simulation is required.
    """
    a1 = qt.tensor(a, I)
    a2 = qt.tensor(I, a)

    S2 = (r_epr * (a1 * a2 - a1.dag() * a2.dag())).expm()

    return S2 * qt.tensor(qt.basis(N, 0),
                          qt.basis(N, 0))


def epr_projector(q, p, r_epr=2.0):
    """
    Direct EPR-measurement projector

        Pi(q,p) = |EPR(q,p)><EPR(q,p)|

    where the measurement outcome is encoded as a
    two-mode displacement.
    """
    epr = epr_state(r_epr)

    # D(alpha) on first mode and D(alpha*) on second mode
    # gives the appropriate phase-space translation.
    alpha_m = (q + 1j*p) / np.sqrt(2)

    U = qt.tensor(D(alpha_m), D(-alpha_m.conjugate()))

    epr_m = U * epr

    return epr_m * epr_m.dag()


# ============================================================
# Logical Pauli operators
# ============================================================

def logical_paulis(**kwargs):
    g0, g1 = gkp_basis(**kwargs)

    X = g0 * g1.dag() + g1 * g0.dag()
    Z = g0 * g0.dag() - g1 * g1.dag()
    Y = -1j * (g0 * g1.dag() - g1 * g0.dag())

    return X, Y, Z


# ============================================================
# GKP teleportation Kraus operator
# ============================================================

def gkp_projector():
    """
    Finite-energy GKP code-space projector.
    """
    g0, g1 = gkp_basis()
    return g0*g0.dag() + g1*g1.dag()

def gkp_projector_GS():
    """
    Finite-energy GKP code-space projector.
    """
    g0, g1 = gkp_basis()
    a = g0.dag() * g1
    g1_gs = (g1 - a * g0).unit()
    return g0*g0.dag() + g1_gs*g1_gs.dag()

#=============================
#=============================

Pi_GS = gkp_projector_GS()
Pi = gkp_projector()

#=============================
#=============================


def teleportation_kraus(q, p):
    """
    Direct representation of the GKP teleportation Kraus operator

        K(q,p) = Pi_GKP D(-alpha)

    without explicitly simulating the EPR measurement circuit.
    """
    # Pi = gkp_projector()

    alpha_m = (q + 1j*p) / np.sqrt(2)

    return Pi * D(-alpha_m)


# ============================================================
# Teleportation
# ============================================================

def teleport(state, q, p):
    """
    Apply the GKP teleportation Kraus operator to an input state.
    """
    K = teleportation_kraus(q, p)

    out = K * state

    prob = (out.dag() * out).full()[0, 0].real

    if prob > 0:
        out = out / np.sqrt(prob)

    return out, prob


# ============================================================
# Leakage
# ============================================================

def gkp_leakage(rho):
    # P = gkp_projector()
    psi = Pi_GS * rho
    return np.real(1-psi.tr())

# ============================================================
# Noisy teleportation channel one mode
# ============================================================

def noisy_teleportation_channel(
        q, p,
        rho,
        tau,
        rmax=3.0,
        nr=101):
    """
    Apply the same Gaussian-random displacement D(r) to
    both modes:

        rho -> ∫dr G(r) D(-ir) rho D†(-ir)

    Parameters
    ----------
    rho : Qobj
        single-mode density matrix.
    sigma : float
        Standard deviation of the displacement.
    rmax : float
        Integration range [-rmax*sigma, rmax*sigma].
    nr : int
        Number of integration points.

    Returns
    -------
    rho_out : Qobj
        Two-mode output density matrix.
    """

    r_values = np.linspace(-rmax*tau, rmax*tau, nr)

    rho_out = qt.Qobj(
        np.zeros((N, N), dtype=complex),
        dims=rho.dims
    )

    for r in r_values:

        # Same displacement on both modes
        K = teleportation_kraus(q, p)
        disp = D(-1j*r)
        U = disp * K * disp

        rho_r = U * rho * U.dag()

        rho_out += gaussian_pdf(r, tau) * rho_r

    # Numerical integration
    rho_out = rho_out * (r_values[1] - r_values[0])

    # Numerical cleanup
    rho_out = (rho_out + rho_out.dag()) / 2
    rho_out = rho_out / rho_out.tr()

    return rho_out

# ============================================================
# N round Noisy teleportation channel one mode
# ============================================================

def noisy_tele_seq(n, rho, sigma):
    arr = []
    q, p = 0, 0
    for k in range(n):
        rho = noisy_teleportation_channel(
                q, p,
                rho,
                sigma,
                rmax=3.0,
                nr=101)
        l = gkp_leakage(rho)
        arr = np.append(arr, l)
    return arr, rho

# ============================================================
# N round Noisy teleportation channel one mode === Uhlmann fid
# ============================================================

def noisy_tele_seq_fid(n, psi, sigma):
    rho = psi * psi.dag()
    arr = []
    q, p = 0, 0
    for k in range(n):
        rho = noisy_teleportation_channel(
                q, p,
                rho,
                sigma,
                rmax=3.0,
                nr=101)
        l = np.real(psi.dag() * rho * psi)
        arr = np.append(arr, l)
    return arr

# ============================================================
# fock to position
# ============================================================

def gkp_wavefunction_x(psi, x):
    """
    Position-space wavefunction <x|psi> for a Fock-space state.
    Convention: [q,p] = i and vacuum variance = 1/2.
    """
    coeffs = psi.full().ravel()

    out = np.zeros_like(x, dtype=complex)

    for n, c in enumerate(coeffs):
        phi_n = (
            np.exp(-x**2 / 2)
            * eval_hermite(n, x)
            / np.sqrt(2**n * factorial(n) * np.sqrt(np.pi))
        )
        out += c * phi_n

    return out

# ============================================================
# Gamma
# ============================================================


def decoherence_exponant_LD(heating_rate, t, osc_freq, cut_off_scale):
    # SI_unit_factor = 0.2 * 10**11 # boltzman / plank
    omega_cut = cut_off_scale * osc_freq
    # A = (eff_coupling * temp)/osc_freq
    B = (np.exp(-omega_cut * t) + omega_cut * t -1) / omega_cut
    return 2 * heating_rate * B

def decoherence_exponant_exp_cutoff(heating_rate, t):
    B = 2 * t * np.arctan(t) - np.log2(1 + t**2)
    return heating_rate * B

#=======================================

def tau_LD(temp, t):
    return temp * (np.exp(-t) + t - 1)

def tau_EC(temp, t):
    return temp * (2 * t * np.arctan(t) - np.log2(1 + t**2))


# ============================================================
# Teleportation Uhlmann fidelity
# ============================================================

def tele_uhlman_fid(psi, t, heating_rate, osc_freq, cut_off_scale):
    q, p = 0, 0
    rho = psi * psi.dag()
    tau = decoherence_exponant_LD(heating_rate, t, osc_freq, cut_off_scale)
    psi_out = noisy_teleportation_channel(
            q, p,
            rho,
            tau,
            rmax=3.0,
            nr=101)
    f = psi.dag() * psi_out * psi
    
    return np.real(f)

#=============================================================

def tele_uhlman_fid_qualitative(psi, t, temp):
    q, p = 0, 0
    rho = psi * psi.dag()
    tau = temp * (np.exp(-t) + t - 1)
    psi_out = noisy_teleportation_channel(
            q, p,
            rho,
            tau,
            rmax=3.0,
            nr=101)
    f = psi.dag() * psi_out * psi
    
    return np.real(f)

def tele_uhlman_fid_qualitative_exp_cutoff(psi, t, temp):
    q, p = 0, 0
    rho = psi * psi.dag()
    tau = temp * (2 * t * np.arctan(t) - np.log2(1 + t**2))
    psi_out = noisy_teleportation_channel(
            q, p,
            rho,
            tau,
            rmax=3.0,
            nr=101)
    f = psi.dag() * psi_out * psi
    
    return np.real(f)

#=============================================================


def avg_tele_uhlman_fid(tau):
    six_states = gkp_6states()
    q, p = 0, 0
    fid = 0
    for psi in six_states:
        rho = psi * psi.dag()
        # tau = decoherence_exponant_LD(heating_rate, t, osc_freq, cut_off_scale)
        psi_out = noisy_teleportation_channel(
                q, p,
                rho,
                tau,
                rmax=3.0,
                nr=101)
        f = psi.dag() * psi_out * psi
        fid = fid + f
        
    return np.real(fid/6)


import numpy as np



# ============================================================
# Example 11 Wigner function plots 
# ============================================================
"""Generates the plot FIG.2(d)"""


# # -------------------------------------------------------
# # Parameters
# # -------------------------------------------------------
# # N = 50
# # xvec = np.linspace(-3, 3, 300)
# tau = 0.5
# # Example parameter values
# params = [0,1,2,6,10]
# val = [1,2,6,10]

# # gkp_states = gkp_6states()

# # Your states should be generated here
# # states = [gkp_states[0], gkp_states[1], gkp_states[2], gkp_states[3], gkp_states[4]]
# # -------------------------------------------------------

# psi = gkp_6states()[4]
# rho = psi * psi.dag()

# states = [rho]
# for n in val:
#     rho_out = noisy_tele_seq(n, rho, tau)[1]
#     states.append(rho_out)
        

# # -------------------------------------------------------
# # Plot
# # -------------------------------------------------------

# # plt.figure(figsize=(10, 8), dpi=900)
# fig, axes = plt.subplots(
#     1, 5,
#     figsize=(13, 3),
#     sharex=True,
#     sharey=True,
#     dpi=900
# )

# # Calculate all Wigner functions
# W = [wigner(state, xvec, xvec) for state in states]

# # Common color scale
# vmax = max(np.max(np.abs(w)) for w in W)
# vmin = -vmax

# for ax, w, param in zip(axes, W, params):

#     cf = ax.contourf(
#         xvec,
#         xvec,
#         w,
#         levels=100,
#         vmin=vmin,
#         vmax=vmax,
#         cmap='RdBu_r'
#     )

#     ax.set_aspect('equal')

#     ax.set_title(rf'$n={param}$', fontsize=18)

#     ax.set_xlabel(r'$q$', fontsize=18)

# # Only first subplot gets y-label
# axes[0].set_ylabel(r'$p$', fontsize=18)

# # -------------------------------------------------------
# # Common colorbar
# # -------------------------------------------------------
# # cbar = fig.colorbar(
# #     cf,
# #     ax=axes,
# #     location='right',
# #     fraction=0.025,
# #     pad=0.03
# # )

# # cbar.set_label(r'$W(q,p)$', fontsize=11)

# # cbar_ax = fig.add_axes([0.90, 0.18, 0.015, 0.67])

# cbar_ax = fig.add_axes([0.925, 0.18, 0.015, 0.67])

# cbar = fig.colorbar(cf, cax=cbar_ax)
# cbar.set_label(r'$W(q,p)$', fontsize=14)


# # -------------------------------------------------------
# # Layout
# # -------------------------------------------------------
# plt.subplots_adjust(
#     wspace=0.05,
#     left=0.06,
#     right=0.92,
#     bottom=0.18,
#     top=0.85
# )

# plt.savefig("wigner_plots_tau_0.5.pdf", format='pdf')


# plt.show()


# ============================================================
# Example 10 
# ============================================================

# psi = gkp_basis()[0]
# rho = psi * psi.dag()
# q,p = 0,0
# tau = 100000

# # rho_out = noisy_teleportation_channel(
# #         q, p,
# #         rho,
# #         tau,
# #         rmax=3.0,
# #         nr=101)


# l =  avg_tele_uhlman_fid(tau)

# print(tau, l)

# ============================================================
# Example 9 Leakage vs time different temp
# ============================================================

"""Following script generates the data for FIG.1(c)"""

# t1 = time.time()

# temp = 0.1

# t_val = np.linspace(0.01, 1, 100)

# psi = gkp_basis()[0]
# rho = psi * psi.dag()
# q,p = 0,0

# my_list = [0]

# for t in t_val:
#     tau = temp * (t - 1 + np.exp(-t))
#     psi_out = noisy_teleportation_channel(
#             q, p,
#             rho,
#             tau,
#             rmax=3.0,
#             nr=101)
#     l = gkp_leakage(psi_out)
#     my_list.append(l)

# # fid_val = [tele_uhlman_fid(psi, t, heating_rate, osc_freq, cut_off_scale) for t in t_val]

# np.save("leakage_Temp_0.1.npy", my_list)

# print(my_list)

# t2 = time.time()

# # print(fid_val)
# print(t2-t1)

#=============================================================
"""Following script generates the plot FIG.1(c)"""

# t_val = np.linspace(0, 1.01, 101)

# fid_1 = np.load("leakage_Temp_0.1.npy")
# fid_2 = np.load("leakage_Temp_1.npy")
# fid_3 = np.load("leakage_Temp_2.npy")
# fid_4 = np.load("leakage_Temp_5.npy")


# plt.figure(figsize=(10, 8), dpi=900)

# plt.plot(t_val, fid_1, linewidth=2, linestyle="--", label = r"$\gamma T/{\omega}_c = 0.1$")
# plt.plot(t_val, fid_2, linewidth=2, linestyle=":", label = r"$\gamma T/{\omega}_c = 1$")
# plt.plot(t_val, fid_3, linewidth=2, linestyle="-.", label = r"$\gamma T/{\omega}_c = 2$")
# plt.plot(t_val, fid_4, linewidth=2, linestyle="-", label = r"$\gamma T/{\omega}_c = 5$")
# # plt.plot(t_val, fid_2, linewidth=2, linestyle="--", label = r"$T/\omega_c = 10^{-7}$")
# # plt.plot(t_val, fid_3, linewidth=2, linestyle="-.", label = r"$T/\omega_c = 3\times 10^{-7}$")
# # plt.plot(t_val, fid_4, linewidth=2, linestyle="-", label = r"$T/\omega_c = 10\times 10^{-7}$")

# plt.xlabel(r"Interaction time $\omega_c t$", fontsize=30)
# plt.ylabel(r"Leakage $\mathcal{L}$", fontsize=30)
# plt.legend(loc='upper left', fontsize=24)
# plt.grid(False)

# # Start x-axis from x = 0
# plt.xlim(left=0, right =1)

# # Scientific notation
# plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))

# ax = plt.gca()

# # Keep all x-ticks except x = 0
# xticks = ax.get_xticks()
# ax.set_xticks(xticks[xticks != 0])

# ax.tick_params(
#     axis="both",
#     which="major",
#     direction="in",
#     top=True,
#     right=True,
#     labelsize=22
# )

# plt.savefig("leakage_LD_vs_time.pdf", format='pdf')

# plt.show()

# ============================================================
# Example 8 LD vs exp cutoff  Tau
# ============================================================
# temp = 5
# t_val = np.linspace(0, 2, 100)

# tau_val_LD = [tau_LD(temp, t) for t in t_val]
# tau_val_EC = [tau_EC(temp, t) for t in t_val]

# plt.figure(figsize=(10, 8), dpi=900)

# plt.plot(t_val, tau_val_LD, linewidth=2, linestyle="--", label = r"Lorentz-Drude")
# plt.plot(t_val, tau_val_EC, linewidth=2, linestyle=":", label = r"exp. cutoff")

# # plt.plot(t_val, fid_2, linewidth=2, linestyle="--", label = r"$T/\omega_c = 10^{-7}$")
# # plt.plot(t_val, fid_3, linewidth=2, linestyle="-.", label = r"$T/\omega_c = 3\times 10^{-7}$")
# # plt.plot(t_val, fid_4, linewidth=2, linestyle="-", label = r"$T/\omega_c = 10\times 10^{-7}$")

# plt.xlabel(r"$\omega_c t$", fontsize=18)
# plt.ylabel(r"$\tau$", fontsize=18)
# plt.legend(loc='upper left', fontsize=15)
# plt.grid(False)

# # Start x-axis from x = 0
# plt.xlim(left=0, right =0.6)
# plt.ylim(bottom=0, top =1)

# # Scientific notation
# plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))

# ax = plt.gca()

# # Keep all x-ticks except x = 0
# xticks = ax.get_xticks()
# ax.set_xticks(xticks[xticks != 0])

# ax.tick_params(
#     axis="both",
#     which="major",
#     direction="in",
#     top=True,
#     right=True,
#     labelsize=14
# )

# plt.savefig("tau_LD_vs_exp_T_5.pdf", format='pdf')

# plt.show()


# ============================================================
# Example 7
# ============================================================

# t1 = time.time()

# temp = 0.1

# t_val = np.linspace(0, 2, 100)

# psi = gkp_basis()[0]

# my_list = []

# for t in t_val:
#     fid_val = tele_uhlman_fid_qualitative_exp_cutoff(psi, t, temp)
#     my_list.append(fid_val)

# # fid_val = [tele_uhlman_fid(psi, t, heating_rate, osc_freq, cut_off_scale) for t in t_val]

# np.save("uhlman_fid_exp_cut_T_0.1.npy", my_list)

# print(my_list)

# t2 = time.time()

# print(fid_val)
# print(t2-t1)

# #=======================

# t1 = time.time()

# temp = 1

# t_val = np.linspace(0, 2, 100)

# psi = gkp_basis()[0]

# my_list = []

# for t in t_val:
#     fid_val = tele_uhlman_fid_qualitative_exp_cutoff(psi, t, temp)
#     my_list.append(fid_val)

# # fid_val = [tele_uhlman_fid(psi, t, heating_rate, osc_freq, cut_off_scale) for t in t_val]

# np.save("uhlman_fid_exp_cut_T_1.npy", my_list)

# print(my_list)

# t2 = time.time()

# print(fid_val)
# print(t2-t1)

# #===========================

# t1 = time.time()

# temp = 2

# t_val = np.linspace(0, 2, 100)

# psi = gkp_basis()[0]

# my_list = []

# for t in t_val:
#     fid_val = tele_uhlman_fid_qualitative_exp_cutoff(psi, t, temp)
#     my_list.append(fid_val)

# # fid_val = [tele_uhlman_fid(psi, t, heating_rate, osc_freq, cut_off_scale) for t in t_val]

# np.save("uhlman_fid_exp_cut_T_2.npy", my_list)

# print(my_list)

# t2 = time.time()

# print(fid_val)
# print(t2-t1)

# #==========================

# t1 = time.time()

# temp = 5

# t_val = np.linspace(0, 2, 100)

# psi = gkp_basis()[0]

# my_list = []

# for t in t_val:
#     fid_val = tele_uhlman_fid_qualitative_exp_cutoff(psi, t, temp)
#     my_list.append(fid_val)

# # fid_val = [tele_uhlman_fid(psi, t, heating_rate, osc_freq, cut_off_scale) for t in t_val]

# np.save("uhlman_fid_exp_cut_T_5.npy", my_list)

# print(my_list)

# t2 = time.time()

# print(fid_val)
# print(t2-t1)

#=============================================================

# t_val = np.linspace(0, 2, 100) 


# fid_1 = np.load("uhlman_fid_exp_cut_T_1.npy")
# fid_2 = np.load("uhlman_fid_exp_cut_T_2.npy")
# fid_3 = np.load("uhlman_fid_exp_cut_T_0.1.npy")
# fid_4 = np.load("uhlman_fid_exp_cut_T_5.npy")
# # fid_2 = np.load("uhlman_fid_nbar_10_Omega_10_MHz.npy")
# # fid_3 = np.load("uhlman_fid_nbar_30_Omega_3Pi_MHz.npy")
# # fid_4 = np.load("uhlman_fid_nbar_100_Omega_10_MHz.npy")

# plt.figure(figsize=(10, 8), dpi=900)

# plt.plot(t_val, fid_3, linewidth=2, linestyle="--", label = r"$\gamma T/\tilde{\omega}_c = 0.1$")
# plt.plot(t_val, fid_1, linewidth=2, linestyle=":", label = r"$\gamma T/\tilde{\omega}_c = 1$")
# plt.plot(t_val, fid_2, linewidth=2, linestyle="-.", label = r"$\gamma T/\tilde{\omega}_c = 2$")
# plt.plot(t_val, fid_4, linewidth=2, linestyle="-", label = r"$\gamma T/\tilde{\omega}_c = 5$")
# # plt.plot(t_val, fid_2, linewidth=2, linestyle="--", label = r"$T/\omega_c = 10^{-7}$")
# # plt.plot(t_val, fid_3, linewidth=2, linestyle="-.", label = r"$T/\omega_c = 3\times 10^{-7}$")
# # plt.plot(t_val, fid_4, linewidth=2, linestyle="-", label = r"$T/\omega_c = 10\times 10^{-7}$")

# plt.xlabel(r"Interaction time $\omega_c t$", fontsize=18)
# plt.ylabel(r"Uhlmann fidelity $f$", fontsize=18)
# plt.legend(loc='lower left', fontsize=15)
# plt.grid(False)

# # Start x-axis from x = 0
# plt.xlim(left=0, right =1)

# # Scientific notation
# plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))

# ax = plt.gca()

# # Keep all x-ticks except x = 0
# xticks = ax.get_xticks()
# ax.set_xticks(xticks[xticks != 0])

# ax.tick_params(
#     axis="both",
#     which="major",
#     direction="in",
#     top=True,
#     right=True,
#     labelsize=14
# )

# plt.savefig("uhlmann_exp_cut_qualitative.pdf", format='pdf')

# plt.show()



# ============================================================
# Example 6 uhlmann fidelity for lorentz-drude
# ============================================================
"""Following script gnerates the data for FIG.2(b)"""

# t1 = time.time()

# temp = 5

# t_val = np.linspace(0, 2, 100)

# psi = gkp_basis()[0]

# my_list = []

# for t in t_val:
#     fid_val = tele_uhlman_fid_qualitative(psi, t, temp)
#     my_list.append(fid_val)

# # fid_val = [tele_uhlman_fid(psi, t, heating_rate, osc_freq, cut_off_scale) for t in t_val]

# np.save("uhlman_fid_T_5.npy", my_list)

# print(my_list)

# t2 = time.time()

# print(fid_val)
# print(t2-t1)

#=============================================================

"""Following script gnerates the plot FIG.2(b)"""

# t_val = np.linspace(0, 2, 100) 


# fid_1 = np.load("uhlman_fid_T_1.npy")
# fid_2 = np.load("uhlman_fid_T_2.npy")
# fid_3 = np.load("uhlman_fid_T_0.1.npy")
# fid_4 = np.load("uhlman_fid_T_5.npy")
# # fid_2 = np.load("uhlman_fid_nbar_10_Omega_10_MHz.npy")
# # fid_3 = np.load("uhlman_fid_nbar_30_Omega_3Pi_MHz.npy")
# # fid_4 = np.load("uhlman_fid_nbar_100_Omega_10_MHz.npy")

# plt.figure(figsize=(10, 8), dpi=900)

# plt.plot(t_val, fid_3, linewidth=2, linestyle="--", label = r"$\gamma T/\omega_c = 0.1$")
# plt.plot(t_val, fid_1, linewidth=2, linestyle=":", label = r"$\gamma T/\omega_c = 1$")
# plt.plot(t_val, fid_2, linewidth=2, linestyle="-.", label = r"$\gamma T/\omega_c = 2$")
# plt.plot(t_val, fid_4, linewidth=2, linestyle="-", label = r"$\gamma T/\omega_c = 5$")
# # plt.plot(t_val, fid_2, linewidth=2, linestyle="--", label = r"$T/\omega_c = 10^{-7}$")
# # plt.plot(t_val, fid_3, linewidth=2, linestyle="-.", label = r"$T/\omega_c = 3\times 10^{-7}$")
# # plt.plot(t_val, fid_4, linewidth=2, linestyle="-", label = r"$T/\omega_c = 10\times 10^{-7}$")

# plt.xlabel(r"Interaction time $\omega_c t$", fontsize=30)
# plt.ylabel(r"Uhlmann fidelity $\mathcal{F}$", fontsize=30)
# plt.legend(loc='lower left', fontsize=24)
# plt.grid(False)

# # Start x-axis from x = 0
# plt.xlim(left=0, right =1)

# # Scientific notation
# plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))

# ax = plt.gca()

# # Keep all x-ticks except x = 0
# xticks = ax.get_xticks()
# ax.set_xticks(xticks[xticks != 0])

# ax.tick_params(
#     axis="both",
#     which="major",
#     direction="in",
#     top=True,
#     right=True,
#     labelsize=22
# )

# plt.savefig("uhlmann_lorentz_drude.pdf", format='pdf')

# plt.show()


# ============================================================
# Example 5
# ============================================================

# t1 = time.time()

# heating_rate = 100
# osc_freq = 10* 10**6
# cut_off_scale = 10
# cut_freq = cut_off_scale * osc_freq

# t_val = np.linspace(10**(-5), 10**(-2), 100)

# psi = gkp_basis()[0]

# my_list = []

# for t in t_val:
#     fid_val = tele_uhlman_fid(psi, t, heating_rate, osc_freq, cut_off_scale)
#     my_list.append(fid_val)

# # fid_val = [tele_uhlman_fid(psi, t, heating_rate, osc_freq, cut_off_scale) for t in t_val]

# np.save("uhlman_fid_nbar_100_Omega_10_MHz.npy", my_list)

# print(my_list)

# t2 = time.time()

# print(fid_val)
# print(t2-t1)

#=============================================================

# heating_rate = 30
# osc_freq = 2* np.pi * 1.5 * 10**6
# cut_off_scale = 10
# cut_freq = cut_off_scale * osc_freq

# t_val = np.linspace(10**(-5), 10**(-2), 100) 

# fid_val = np.load("uhlman_fid_nbar_1_Omega_10_MHz.npy")

# fid_1 = np.load("uhlman_fid_nbar_1_Omega_10_MHz.npy")
# fid_2 = np.load("uhlman_fid_nbar_10_Omega_10_MHz.npy")
# fid_3 = np.load("uhlman_fid_nbar_30_Omega_3Pi_MHz.npy")
# fid_4 = np.load("uhlman_fid_nbar_100_Omega_10_MHz.npy")

# plt.figure(figsize=(10, 8), dpi=900)

# plt.plot(t_val, fid_1, linewidth=2, linestyle=":", label = r"$T/\omega_c = 0.1\times 10^{-7}$")
# plt.plot(t_val, fid_2, linewidth=2, linestyle="--", label = r"$T/\omega_c = 10^{-7}$")
# plt.plot(t_val, fid_3, linewidth=2, linestyle="-.", label = r"$T/\omega_c = 3\times 10^{-7}$")
# plt.plot(t_val, fid_4, linewidth=2, linestyle="-", label = r"$T/\omega_c = 10\times 10^{-7}$")


# plt.xlabel(r"Interaction time $\omega_c t$", fontsize=18)
# plt.ylabel(r"Uhlmann fidelity $f$", fontsize=18)
# plt.legend(loc='lower left', fontsize=15)
# plt.grid(False)

# # Start x-axis from x = 0
# plt.xlim(left=0, right = max(t_val))

# # Scientific notation
# plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))

# ax = plt.gca()

# # Keep all x-ticks except x = 0
# xticks = ax.get_xticks()
# ax.set_xticks(xticks[xticks != 0])

# ax.tick_params(
#     axis="both",
#     which="major",
#     direction="in",
#     top=True,
#     right=True,
#     labelsize=14
# )

# plt.savefig("uhlmann_Omega_10_MHz.pdf", format='pdf')

# plt.show()


# ============================================================
# Example 4
# ============================================================

# heating_rate = 30
# osc_freq = 2* np.pi * 1.5 * 10**6
# cut_off_scale = 10
# cut_freq = cut_off_scale * osc_freq

# t_val = np.linspace(0, 10**(-1), 10)

# fid_val = np.array([avg_tele_uhlman_fid(t, heating_rate, osc_freq, cut_off_scale) for t in t_val])

# plt.figure(figsize=(10, 8), dpi=900)

# plt.plot(t_val/(cut_freq ), fid_val, label="", linewidth=2)
# # plt.plot(x_val, g1_val, label=r'GKP $|1_L\rangle$', linewidth=2)

# # plt.xlim(-8, 8)
# # plt.ylim(np.min([g0_val.min(), g1_val.min()]),
# #           np.max([g0_val.max(), g1_val.max()]))

# plt.xlabel(r"$x$", fontsize=20)
# plt.ylabel(r"$\psi(x)$", fontsize=20)
# plt.legend(fontsize=16)
# plt.grid(False)

# plt.show()
# ============================================================
# Example 3
# ============================================================

# temp = 4
# osc_freq = 2* np.pi * 1.5 * 10**6
# cut_off_scale = 10

# def Gamma(g, t):
#     return decoherence_exponant(g, t, osc_freq, cut_off_scale)

# g_vec = np.linspace(10**(-2), 0.5*10**(2), 200)
# t_vec = np.linspace(10**(-6), 10**(-3), 200)


# gamma_val = [[Gamma(g,t) for t in t_vec] for g in g_vec]

# #======

# plt.figure(figsize=(6, 5))

# ax = plt.contourf(
#     t_vec,
#     g_vec,
#     gamma_val,
#     levels=200,
#     cmap="RdBu_r"
# )

# plt.xlabel(r"Time $t$ (s)")
# plt.ylabel(r"Heating rate $\bar{n}$ ($s^{-1}$)")
# plt.title("")

# # Scientific notation
# plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))

# plt.colorbar(label="")
# plt.tight_layout()

# plt.savefig("Gamma_plot_1.pdf", format='pdf')

# plt.show()

# ============================================================
# Example 2 Wigner excercise
# ============================================================
# n = 10
# tau = 10
# q, p = 0, 0
# g0, g1 = gkp_basis()

# rho_in = ket2dm(g0)

# # rho_out = noisy_teleportation_channel(
# #         q, p,
# #         rho_in,
# #         tau,
# #         rmax=3.0,
# #         nr=101)

# rho_out = noisy_tele_seq(n, rho_in, tau)[1]


# W = wigner(rho_out, xvec, xvec)

# # --------------------------------------------------
# # Plot
# # --------------------------------------------------

# plt.figure(figsize=(6, 5))

# plt.contourf(xvec, xvec, W, levels=100)

# plt.xlabel(r"$q$")
# plt.ylabel(r"$p$")
# plt.title("")

# plt.colorbar(label=r"$W(q,p)$")
# plt.tight_layout()
# plt.show()

# # Symmetric color scale around zero
# Wmax = np.max(np.abs(W))
# norm = TwoSlopeNorm(vmin=-Wmax, vcenter=0, vmax=Wmax)

# # --------------------------------------------------
# # Plot
# # --------------------------------------------------
# plt.figure(figsize=(6, 5))

# plt.contourf(
#     xvec,
#     xvec,
#     W,
#     levels=200,
#     cmap="RdBu_r",
#     norm=norm
# )

# plt.xlabel(r"$q$")
# plt.ylabel(r"$p$")
# plt.title("")

# plt.colorbar(label=r"$W(q,p)$")
# plt.tight_layout()
# plt.show()


# ============================================================
# Example 1 leakage accumulation
# ============================================================

# time0 = time.time()

# g0, g1 = gkp_basis()
# q, p = 0, 0
# n = 20

# Pure-state density matrix
# rho = g0 * g0.dag()

# Gaussian displacement noise

# chi = noisy_tele_seq(n, rho, sigma)

# print(chi)

# chi0 = noisy_tele_seq(n, rho, 0.01)

# np.save("seq_tele_leakage_0.01.npy", chi0)
# print(chi0)

# chi1 = noisy_tele_seq(n, rho, 0.05)

# np.save("seq_tele_leakage_0.05.npy", chi1)

# chi2 = noisy_tele_seq(n, rho, 0.1)

# np.save("seq_tele_leakage_0.1.npy", chi2)

# chi3 = noisy_tele_seq(n, rho, 0.15)

# np.save("seq_tele_leakage_0.15.npy", chi3)

# chi4 = noisy_tele_seq(n, rho, 0.2)

# np.save("seq_tele_leakage_0.2.npy", chi4)

# chi5 = noisy_tele_seq(n, rho, 0.25)

# np.save("seq_tele_leakage_0.25.npy", chi5)

# chi6 = noisy_tele_seq(n, rho, 0.3)

# np.save("seq_tele_leakage_0.3.npy", chi6)

# chi7 = noisy_tele_seq(n, rho, 0.1)

# np.save("seq_tele_leakage_0.1.npy", chi7)

# chi9 = noisy_tele_seq(n, rho, 0.5)

# np.save("seq_tele_leakage_0.5.npy", chi9)

# chi8 = noisy_tele_seq(n, rho, 1)

# np.save("seq_tele_leakage_1.npy", chi8)

# time1 = time.time()

# print(time1-time0)
# ============================================================
# Example 2 N_vs_GKP_leakage
# ============================================================


# g0, g1 = gkp_basis()

# x_val = np.arange(-8, 8, 0.1)

# g0_val = np.array([gkp_wavefunction_x(g0, x) for x in x_val])
# g1_val = np.array([gkp_wavefunction_x(g1, x) for x in x_val])

# plt.figure(figsize=(10, 8), dpi=300)

# plt.plot(x_val, g0_val, label=r'GKP $|0_L\rangle$', linewidth=2)
# plt.plot(x_val, g1_val, label=r'GKP $|1_L\rangle$', linewidth=2)

# plt.xlim(-8, 8)
# plt.ylim(np.min([g0_val.min(), g1_val.min()]),
#           np.max([g0_val.max(), g1_val.max()]))

# plt.xlabel(r"$x$", fontsize=20)
# plt.ylabel(r"$\psi(x)$", fontsize=20)
# plt.legend(fontsize=16)
# plt.grid(False)

# plt.show()

# n_val = np.arange(1,n+1)

# l_01 = np.load("seq_tele_leakage_0.01.npy")
# l_05 = np.load("seq_tele_leakage_0.05.npy")
# l_1 = np.load("seq_tele_leakage_0.1.npy")
# l_2 = np.load("seq_tele_leakage_0.2.npy")
# l_3 = np.load("seq_tele_leakage_0.3.npy")
# l_5 = np.load("seq_tele_leakage_0.5.npy")
# l_10 = np.load("seq_tele_leakage_1.npy")

# plt.figure(figsize=(10, 8), dpi=300)

# plt.xlim(1, 10)

# # plt.plot(n_val, l_01, label=r'$\tau=0.01$', linewidth=2)
# # plt.plot(n_val, l_05, label=r'$\tau=0.05$', linewidth=2)
# plt.plot(n_val, l_1, 'o-', markersize=7, label=r'$\tau=0.1$', linewidth=2)
# plt.plot(n_val, l_2, 's-', markersize=7, label=r'$\tau=0.2$', linewidth=2)
# plt.plot(n_val, l_3, '^-', markersize=7, label=r'$\tau=0.3$', linewidth=2)
# plt.plot(n_val, l_5, 'd-', markersize=7, label=r'$\tau=0.5$', linewidth=2)
# plt.plot(n_val, l_10, '*-', markersize=7, label=r'$\tau=1.0$', linewidth=2)

# plt.xlabel(r"correction rounds $n$", fontsize=30)
# plt.ylabel(r"leakage $\mathcal{L}$", fontsize=30)
# plt.legend(loc='upper right', fontsize=24)
# plt.grid(False)

# ax = plt.gca()
# ax.tick_params(axis = "both", which="major", direction="in", top=True, right=True, labelsize=22)

# plt.savefig("N_vs_GKP_leakage.pdf", format='pdf')

# plt.show()

# ============================================================
# Example 0 fidelity n round teleportation
# ============================================================
""" following script generates the data for FIG.2(a)"""
# time0 = time.time()

# g0, g1 = gkp_basis()
# q, p = 0, 0
# n = 10

# # Pure-state density matrix
# # rho = g0 * g0.dag()

# # Gaussian displacement noise

# # chi = noisy_tele_seq_fid(n, g0, 1)

# # print(chi)

# chi = noisy_tele_seq_fid(n, g0, 1)

# np.save("seq_tele_fid_1.npy", chi)
# print(chi)



# chi1 = noisy_tele_seq(n, rho, 0.05)

# np.save("seq_tele_leakage_0.05.npy", chi1)

# chi2 = noisy_tele_seq(n, rho, 0.1)

# np.save("seq_tele_leakage_0.1.npy", chi2)

# chi3 = noisy_tele_seq(n, rho, 0.15)

# np.save("seq_tele_leakage_0.15.npy", chi3)

# chi4 = noisy_tele_seq(n, rho, 0.2)

# np.save("seq_tele_leakage_0.2.npy", chi4)

# chi5 = noisy_tele_seq(n, rho, 0.25)

# np.save("seq_tele_leakage_0.25.npy", chi5)

# chi6 = noisy_tele_seq(n, rho, 0.3)

# np.save("seq_tele_leakage_0.3.npy", chi6)

# chi7 = noisy_tele_seq(n, rho, 0.1)

# np.save("seq_tele_leakage_0.1.npy", chi7)

# chi9 = noisy_tele_seq(n, rho, 0.5)

# np.save("seq_tele_leakage_0.5.npy", chi9)

# chi8 = noisy_tele_seq(n, rho, 1)

# np.save("seq_tele_leakage_1.npy", chi8)

# time1 = time.time()

# print(time1-time0)
# ============================================================
# 
# ============================================================
""" following script plots the data for FIG.2(a)"""

# g0, g1 = gkp_basis(beta)

# x_val = np.arange(-8, 8, 0.1)

# g0_val = np.array([gkp_wavefunction_x(g0, x) for x in x_val]1
# g1_val = np.array([gkp_wavefunction_x(g1, x) for x in x_val])

# plt.figure(figsize=(10, 8), dpi=300)

# plt.plot(x_val, g0_val, label=r'GKP $|0_L\rangle$', linewidth=2)
# plt.plot(x_val, g1_val, label=r'GKP $|1_L\rangle$', linewidth=2)

# plt.xlim(-8, 8)
# plt.ylim(np.min([g0_val.min(), g1_val.min()]),
#          np.max([g0_val.max(), g1_val.max()]))

# plt.xlabel(r"$x$", fontsize=20)
# plt.ylabel(r"$\psi(x)$", fontsize=20)
# plt.legend(fontsize=16)
# plt.grid(False)

# plt.show()

# n_val = np.arange(1,n+1)

# l_01 = np.load("seq_tele_leakage_0.01.npy")
# l_05 = np.load("seq_tele_leakage_0.05.npy")
# l_1 = np.load("seq_tele_leakage_0.1.npy")
# l_2 = np.load("seq_tele_leakage_0.2.npy")
# l_3 = np.load("seq_tele_leakage_0.3.npy")
# l_5 = np.load("seq_tele_leakage_0.5.npy")
# l_10 = np.load("seq_tele_leakage_1.npy")

# plt.figure(figsize=(10, 8), dpi=300)

# plt.xlim(1, 10)

# # plt.plot(n_val, l_01, label=r'$\tau=0.01$', linewidth=2)
# # plt.plot(n_val, l_05, label=r'$\tau=0.05$', linewidth=2)
# plt.plot(n_val, l_1, 'o-', markersize=7, label=r'$\tau=0.1$', linewidth=2)
# plt.plot(n_val, l_2, 's-', markersize=7, label=r'$\tau=0.2$', linewidth=2)
# plt.plot(n_val, l_3, '^-', markersize=7, label=r'$\tau=0.3$', linewidth=2)
# plt.plot(n_val, l_5, 'd-', markersize=7, label=r'$\tau=0.5$', linewidth=2)
# plt.plot(n_val, l_10, '*-', markersize=7, label=r'$\tau=1.0$', linewidth=2)

# plt.xlabel(r"correction rounds $n$", fontsize=18)
# plt.ylabel(r"leakage $\mathcal{L}$", fontsize=18)
# plt.legend(loc='upper right', fontsize=16)
# plt.grid(False)

# ax = plt.gca()
# ax.tick_params(axis = "both", which="major", direction="in", top=True, right=True, labelsize=14)

# plt.savefig("N_vs_GKP_leakage.pdf", format='pdf')

# plt.show()











