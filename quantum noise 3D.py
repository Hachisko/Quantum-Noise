# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 16:17:08 2022
Hachiskooo quantum shit
@author: Hp
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 12:30:32 2022

@author: Hp
"""

## Importing packages
import numpy as np
import scipy.linalg as scl
from numpy import linalg as LA
import matplotlib.pyplot as plt
import time
from scipy.optimize import curve_fit
from scipy.special import jv 


start_time = time.time()

# Parameters
Bz = 1                  # magnetic field along Z-direction 
Gamma0 = 1              # coupling stochastic noise
hbar = 1                # Planck constant
ti = 0                  # initial time
T = 10*np.pi/(2*Bz)     # period
tf = 2*T                # final time
sigma = 0.1             # standard deviation
mu = 0                  # mean
Nsim =1000               # number of simulations
M = 50                  # number of points for autocorrelation function

# System matrices
I = np.matrix([[1,0],[0,1]])               # identity matrix
Sz = np.matrix([[1,0],[0,-1]])             # Pauli matrix sigma_z
Sx = np.matrix([[0,1],[1,0]])              # Pauli matrix sigma_x
Sy = np.matrix([[0,-1j],[1j,0]])           # Pauli matrix sigma_y

# Time
Nt = 1000                   # number of steps
dt = (tf-ti)/Nt              # step time
t = np.arange(ti,tf,dt)      # time vector

# Spin basis
up = np.matrix([[1],[0]])
down = np.matrix([[0],[1]])

# Initial condition: |Psi(0)>
uPsi_0 = up
dPsi_0 = down
udPsi_0 = up + down
duPsi_0 = up - down

uPsi_0 = uPsi_0/LA.norm(uPsi_0)
dPsi_0 = dPsi_0/LA.norm(dPsi_0)
udPsi_0 = udPsi_0/LA.norm(udPsi_0)
duPsi_0 = duPsi_0/LA.norm(duPsi_0)

# Random noise: Gauss-Markov
# Gamma X
W = np.random.normal(mu, sigma, Nt)
Gammax = np.zeros([1,Nt], dtype=np.complex_)
Gammax[0][0] = 0;
for n in range(1,Nt):
    Gammax[0][n] = Gammax[0][n-1]*(1-sigma) + W[n];
    

Gammax = Gamma0*Gammax[0][:]

# Gamma Y
Y = np.random.normal(mu, sigma, Nt)
Gammay = np.zeros([1,Nt], dtype=np.complex_)
Gammay[0][0] = 0;
for n in range(1,Nt):
    Gammay[0][n] = Gammay[0][n-1]*(1-sigma) + Y[n];
    

Gammay = Gamma0*Gammay[0][:]

# Gamma Z
Z = np.random.normal(mu, sigma, Nt)
Gammaz = np.zeros([1,Nt], dtype=np.complex_)
Gammaz[0][0] = 0;
for n in range(1,Nt):
    Gammaz[0][n] = Gammaz[0][n-1]*(1-sigma) + Z[n];

Gammaz = Gamma0*Gammaz[0][:]

# Plot noise
lineW = 2 # Line thickness
plt.figure(figsize=(10,6), tight_layout=True)
plt.plot(t/T, Gammax,'-r', linewidth=3,label=r'$\Gamma_x$'); 
plt.plot(t/T, Gammay,'-g', linewidth=3,label=r'$\Gamma_y$'); 
plt.plot(t/T, Gammaz,'-b', linewidth=3,label=r'$\Gamma_z$');  
plt.legend(prop={"size":20})
axes = plt.gca()
axes.xaxis.label.set_size(22)
axes.yaxis.label.set_size(22)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlim(0, tf/T)


# Autocorrelation function
Gz = np.zeros([1,M], dtype=np.complex_)
Gz_theo = np.zeros([1,M], dtype=np.complex_)
G = 0
for k in range(0,Nsim):
    
    # Random noise
    W = np.random.normal(mu, sigma, Nt)
    Gamma = np.zeros([1,Nt], dtype=np.complex_)
    
    for n1 in range(1,Nt):
        Gamma[0][n1] = Gamma[0][n1-1]*(1-sigma) + W[n1]    
    
    Gamma = Gamma0*Gamma;
    for m in range(0,M):
        Rz = 0
        for n2 in range(0,Nt-m):
            Rz = Rz + 1/(Nt-m)*Gamma[0][n2]*Gamma[0][n2+m]
        Gz[0][m] = Rz
        Gz_theo[0][m] = Gz[0][0]*np.exp(-m*sigma)
    
    G = G + Gz/Nsim;

# Final correlation functions
G = G[0][:]               # Average ensemble
Gz_theo = Gz_theo[0][:]   # Theoretical
# Time for correlation function
dtG = (tf-ti)/M                    # step time
tG = np.arange(ti,tf,dtG)          # time vector
# Auxiliary Time
L = 10                              # number of time points bewteen t and t+dt
dtaux = (tf-ti)/(L*Nt)              # step time
taux = np.arange(ti,tf,dtaux)       # time vector



# Plot noise
lineW = 2 # Line thickness
plt.figure(figsize=(10,6), tight_layout=True)
plt.plot(tG/T, G,'-b', linewidth=3,label=r'$numerical$');  
plt.plot(tG/T, Gz_theo,'--r', linewidth=3,label=r'$theoretical$'); 
plt.ylabel(r'$\langle \Gamma(\tau) \Gamma\rangle$');plt.xlabel(r'$\tau/T$')
#plt.legend(prop={"size":20})
axes = plt.gca()
axes.xaxis.label.set_size(22)
axes.yaxis.label.set_size(22)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlim(0, tf/T)
plt.legend(prop={"size":20})
# plt.ylim(np.min(Gz), np.max(Gz))


# Average observables
uavgSx = 0
davgSx = 0
udavgSx = 0
duavgSx = 0

uavgSy = 0
davgSy = 0
udavgSy = 0
duavgSy = 0

uavgSz = 0
davgSz = 0
udavgSz = 0
duavgSz = 0

uavgpe = 0
davgpe = 0
udavgpe = 0
duavgpe = 0

uavgpg = 0
davgpg = 0
udavgpg = 0
duavgpg = 0

uavgC = 0
davgC = 0
udavgC = 0
duavgC = 0

# Physical observables: coherence
upe = np.zeros([1,Nt], dtype=np.complex_)
dpe = np.zeros([1,Nt], dtype=np.complex_)
udpe = np.zeros([1,Nt], dtype=np.complex_)
dupe = np.zeros([1,Nt], dtype=np.complex_)

upg = np.zeros([1,Nt], dtype=np.complex_)
dpg = np.zeros([1,Nt], dtype=np.complex_)
udpg = np.zeros([1,Nt], dtype=np.complex_)
dupg = np.zeros([1,Nt], dtype=np.complex_)

uC = np.zeros([1,Nt], dtype=np.complex_)
dC = np.zeros([1,Nt], dtype=np.complex_)
udC = np.zeros([1,Nt], dtype=np.complex_)
duC = np.zeros([1,Nt], dtype=np.complex_)

uMx = np.zeros([1,Nt], dtype=np.complex_)
dMx = np.zeros([1,Nt], dtype=np.complex_)
udMx = np.zeros([1,Nt], dtype=np.complex_)
duMx = np.zeros([1,Nt], dtype=np.complex_)

uMy = np.zeros([1,Nt], dtype=np.complex_)
dMy = np.zeros([1,Nt], dtype=np.complex_)
udMy = np.zeros([1,Nt], dtype=np.complex_)
duMy = np.zeros([1,Nt], dtype=np.complex_)

uMz = np.zeros([1,Nt], dtype=np.complex_)
dMz = np.zeros([1,Nt], dtype=np.complex_)
udMz = np.zeros([1,Nt], dtype=np.complex_)
duMz = np.zeros([1,Nt], dtype=np.complex_)

# Iteration
for k in range(0,Nsim):

    # Generating Random Noise    
    W = np.random.normal(mu, sigma, Nt)
    Gammax = np.zeros([1,Nt], dtype=np.complex_)
    
    Y = np.random.normal(mu, sigma, Nt)
    Gammay = np.zeros([1,Nt], dtype=np.complex_)
    
    Z = np.random.normal(mu, sigma, Nt)
    Gammaz = np.zeros([1,Nt], dtype=np.complex_)

    for n in range(0,Nt):
        # Iteration
        Gammax[0][n] = Gammax[0][n-1]*(1-sigma) + W[n]
        Gammay[0][n] = Gammay[0][n-1]*(1-sigma) + Y[n]   
        Gammaz[0][n] = Gammaz[0][n-1]*(1-sigma) + Z[n]   
    Gammax = Gamma0*Gammax;
    Gammay = Gamma0*Gammay;
    Gammaz = Gamma0*Gammaz;
    
    for n in range(0,Nt):
                        
        if n==0:
            # Initial condition
            uPsi = uPsi_0
            dPsi = dPsi_0
            udPsi = udPsi_0
            duPsi = duPsi_0
            
            urho = uPsi * uPsi.getH()
            drho = dPsi * dPsi.getH()
            udrho = udPsi * udPsi.getH()
            durho = duPsi * duPsi.getH()
            # Observables
            upe[0][n] = np.real(urho[0,0])
            dpe[0][n] = np.real(drho[0,0])
            udpe[0][n] = np.real(udrho[0,0])
            dupe[0][n] = np.real(durho[0,0])
            
            upg[0][n] = np.real(urho[1,1])
            dpg[0][n] = np.real(drho[1,1])
            udpg[0][n] = np.real(udrho[1,1])
            dupg[0][n] = np.real(durho[1,1])
            
            uC[0][n] = 2*np.abs(urho[0,1])
            dC[0][n] = 2*np.abs(drho[0,1])
            udC[0][n] = 2*np.abs(udrho[0,1])
            duC[0][n] = 2*np.abs(durho[0,1])
                       
            uMx[0][n] = uPsi.getH() * Sx * uPsi
            dMx[0][n] = dPsi.getH() * Sx * dPsi
            udMx[0][n] = udPsi.getH() * Sx * udPsi
            duMx[0][n] = duPsi.getH() * Sx * duPsi
            
            uMy[0][n] = uPsi.getH() * Sy * uPsi
            dMy[0][n] = dPsi.getH() * Sy * dPsi
            udMy[0][n] = udPsi.getH() * Sy * udPsi
            duMy[0][n] = duPsi.getH() * Sy * duPsi
            
            uMz[0][n] = uPsi.getH() * Sz * uPsi
            dMz[0][n] = dPsi.getH() * Sz * dPsi
            udMz[0][n] = udPsi.getH() * Sz * udPsi
            duMz[0][n] = duPsi.getH() * Sz * duPsi
            
        else:

            # Hamitonian 
            H = Gammax[0][n]*Sx + Gammay[0][n]*Sy + Gammaz[0][n]*Sz
            
            # Time propagator for a discrete time with step dt
            U = scl.expm(-1j*H/hbar*dt)
            
            # Time-proagation
            uPsi = U*uPsi
            dPsi = U*dPsi
            udPsi = U*udPsi
            duPsi = U*duPsi
            
            urho = uPsi * uPsi.getH()
            drho = dPsi * dPsi.getH()
            udrho = udPsi * udPsi.getH()
            durho = duPsi * duPsi.getH()
            
            # Observables
            upe[0][n] = np.real(urho[0,0])
            dpe[0][n] = np.real(drho[0,0])
            udpe[0][n] = np.real(udrho[0,0])
            dupe[0][n] = np.real(durho[0,0])
            
            upg[0][n] = np.real(urho[1,1])
            dpg[0][n] = np.real(drho[1,1])
            udpg[0][n] = np.real(udrho[1,1])
            dupg[0][n] = np.real(durho[1,1])
            
            uC[0][n] = 2*np.abs(urho[0,1])
            dC[0][n] = 2*np.abs(drho[0,1])
            udC[0][n] = 2*np.abs(udrho[0,1])
            duC[0][n] = 2*np.abs(durho[0,1])
                       
            uMx[0][n] = uPsi.getH() * Sx * uPsi
            dMx[0][n] = dPsi.getH() * Sx * dPsi
            udMx[0][n] = udPsi.getH() * Sx * udPsi
            duMx[0][n] = duPsi.getH() * Sx * duPsi
            
            uMy[0][n] = uPsi.getH() * Sy * uPsi
            dMy[0][n] = dPsi.getH() * Sy * dPsi
            udMy[0][n] = udPsi.getH() * Sy * udPsi
            duMy[0][n] = duPsi.getH() * Sy * duPsi
            
            uMz[0][n] = uPsi.getH() * Sz * uPsi
            dMz[0][n] = dPsi.getH() * Sz * dPsi
            udMz[0][n] = udPsi.getH() * Sz * udPsi
            duMz[0][n] = duPsi.getH() * Sz * duPsi
      
    uavgpe = uavgpe + upe/Nsim
    davgpe = davgpe + dpe/Nsim
    udavgpe = udavgpe + udpe/Nsim
    duavgpe = duavgpe + dupe/Nsim
                
    uavgpg = uavgpg + upg/Nsim
    davgpg = davgpg + dpg/Nsim
    udavgpg = udavgpg + udpg/Nsim
    duavgpg = duavgpg + dupg/Nsim
        
    uavgC = uavgC + uC/Nsim
    davgC = davgC + dC/Nsim
    udavgC = udavgC + udC/Nsim
    duavgC = duavgC + duC/Nsim
    
    uavgSx = uavgSx + uMx/Nsim
    davgSx = davgSx + dMx/Nsim
    udavgSx = udavgSx + udMx/Nsim
    duavgSx = duavgSx + duMx/Nsim
    
    uavgSy = uavgSy + uMy/Nsim
    davgSy = davgSy + dMy/Nsim
    udavgSy = udavgSy + udMy/Nsim
    duavgSy = duavgSy + duMy/Nsim
    
    uavgSz = uavgSz + uMz/Nsim
    davgSz = davgSz + dMz/Nsim
    udavgSz = udavgSz + udMz/Nsim
    duavgSz = duavgSz + duMz/Nsim

# Observables
uC = uC[0][:] 
dC = dC[0][:] 
udC = udC[0][:] 
duC = duC[0][:] 

uavgC = uavgC[0][:]
davgC = davgC[0][:]
udavgC = udavgC[0][:]
duavgC = duavgC[0][:]

uavgpe = uavgpe[0][:]
davgpe = davgpe[0][:]
udavgpe = udavgpe[0][:]
duavgpe = duavgpe[0][:]

uavgpg = uavgpg[0][:]
davgpg = davgpg[0][:]
udavgpg = udavgpg[0][:]
duavgpg = duavgpg[0][:]

uavgSx = uavgSx[0][:]
davgSx = davgSx[0][:]
udavgSx = udavgSx[0][:]
duavgSx = duavgSx[0][:]

uavgSy = uavgSy[0][:]
davgSy = davgSy[0][:]
udavgSy = udavgSy[0][:]
duavgSy = duavgSy[0][:]

uavgSz = uavgSz[0][:]
davgSz = davgSz[0][:]
udavgSz = udavgSz[0][:]
duavgSz = duavgSz[0][:]

# Plot Sigma Sx (UP)
lineW = 2 # Line thickness
plt.figure(figsize=(10,6), tight_layout=True)
plt.plot(t/T, uavgSx,'-b', linewidth=3);  
plt.ylabel(r'$\langle S_x  (up) \rangle$');plt.xlabel(r'$t/T$')
plt.legend(prop={"size":20})
axes = plt.gca()
axes.xaxis.label.set_size(22)
axes.yaxis.label.set_size(22)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlim(0, tf/T)
plt.ylim(np.min(uavgSx), np.max(uavgSx))

# Plot Sigma Sx (DOWN)
lineW = 2 # Line thickness
plt.figure(figsize=(10,6), tight_layout=True)
plt.plot(t/T, davgSx,'-b', linewidth=3);  
plt.ylabel(r'$\langle S_x  (DOWN) \rangle$');plt.xlabel(r'$t/T$')
plt.legend(prop={"size":20})
axes = plt.gca()
axes.xaxis.label.set_size(22)
axes.yaxis.label.set_size(22)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlim(0, tf/T)
plt.ylim(np.min(davgSx), np.max(davgSx))

# Plot Sigma Sx (UP+DOWN)
lineW = 2 # Line thickness
plt.figure(figsize=(10,6), tight_layout=True)
plt.plot(t/T, udavgSx,'-b', linewidth=3);  
plt.ylabel(r'$\langle S_x  (UP+DOWN) \rangle$');plt.xlabel(r'$t/T$')
plt.legend(prop={"size":20})
axes = plt.gca()
axes.xaxis.label.set_size(22)
axes.yaxis.label.set_size(22)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlim(0, tf/T)
plt.ylim(np.min(udavgSx), np.max(udavgSx))

# Plot Sigma Sx (UP-DOWN)
lineW = 2 # Line thickness
plt.figure(figsize=(10,6), tight_layout=True)
plt.plot(t/T, duavgSx,'-b', linewidth=3);  
plt.ylabel(r'$\langle S_x  (UP-DOWN) \rangle$');plt.xlabel(r'$t/T$')
plt.legend(prop={"size":20})
axes = plt.gca()
axes.xaxis.label.set_size(22)
axes.yaxis.label.set_size(22)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlim(0, tf/T)
plt.ylim(np.min(duavgSx), np.max(duavgSx))

# Plot Sigma Sy (UP)
lineW = 2 # Line thickness
plt.figure(figsize=(10,6), tight_layout=True)
plt.plot(t/T, uavgSy,'-b', linewidth=3);  
plt.ylabel(r'$\langle S_y  (up) \rangle$');plt.xlabel(r'$t/T$')
plt.legend(prop={"size":20})
axes = plt.gca()
axes.xaxis.label.set_size(22)
axes.yaxis.label.set_size(22)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlim(0, tf/T)
plt.ylim(np.min(uavgSy), np.max(uavgSy))

# Plot Sigma Sy (DOWN)
lineW = 2 # Line thickness
plt.figure(figsize=(10,6), tight_layout=True)
plt.plot(t/T, davgSy,'-b', linewidth=3);  
plt.ylabel(r'$\langle S_y  (DOWN) \rangle$');plt.xlabel(r'$t/T$')
plt.legend(prop={"size":20})
axes = plt.gca()
axes.xaxis.label.set_size(22)
axes.yaxis.label.set_size(22)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlim(0, tf/T)
plt.ylim(np.min(davgSy), np.max(davgSy))

# Plot Sigma Sy (UP+DOWN)
lineW = 2 # Line thickness
plt.figure(figsize=(10,6), tight_layout=True)
plt.plot(t/T, udavgSy,'-b', linewidth=3);  
plt.ylabel(r'$\langle S_x  (UP+DOWN) \rangle$');plt.xlabel(r'$t/T$')
plt.legend(prop={"size":20})
axes = plt.gca()
axes.xaxis.label.set_size(22)
axes.yaxis.label.set_size(22)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlim(0, tf/T)
plt.ylim(np.min(udavgSy), np.max(udavgSy))

# Plot Sigma Sy (UP-DOWN)
lineW = 2 # Line thickness
plt.figure(figsize=(10,6), tight_layout=True)
plt.plot(t/T, duavgSy,'-b', linewidth=3);  
plt.ylabel(r'$\langle S_y  (UP-DOWN) \rangle$');plt.xlabel(r'$t/T$')
plt.legend(prop={"size":20})
axes = plt.gca()
axes.xaxis.label.set_size(22)
axes.yaxis.label.set_size(22)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlim(0, tf/T)
plt.ylim(np.min(duavgSy), np.max(duavgSy))

# Plot Sigma Sz (UP)
lineW = 2 # Line thickness
plt.figure(figsize=(10,6), tight_layout=True)
plt.plot(t/T, uavgSz,'-b', linewidth=3);  
plt.ylabel(r'$\langle S_z (up) \rangle$');plt.xlabel(r'$t/T$')
plt.legend(prop={"size":20})
axes = plt.gca()
axes.xaxis.label.set_size(22)
axes.yaxis.label.set_size(22)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlim(0, tf/T)
plt.ylim(np.min(uavgSz), np.max(uavgSz))

# Plot Sigma Sz (DOWN)
lineW = 2 # Line thickness
plt.figure(figsize=(10,6), tight_layout=True)
plt.plot(t/T, davgSz,'-b', linewidth=3);  
plt.ylabel(r'$\langle S_z (DOWN) \rangle$');plt.xlabel(r'$t/T$')
plt.legend(prop={"size":20})
axes = plt.gca()
axes.xaxis.label.set_size(22)
axes.yaxis.label.set_size(22)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlim(0, tf/T)
plt.ylim(np.min(davgSz), np.max(davgSz))

# Plot Sigma Sz (UP+DOWN)
lineW = 2 # Line thickness
plt.figure(figsize=(10,6), tight_layout=True)
plt.plot(t/T, udavgSz,'-b', linewidth=3);  
plt.ylabel(r'$\langle S_z  (UP+DOWN) \rangle$');plt.xlabel(r'$t/T$')
plt.legend(prop={"size":20})
axes = plt.gca()
axes.xaxis.label.set_size(22)
axes.yaxis.label.set_size(22)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlim(0, tf/T)
plt.ylim(np.min(udavgSz), np.max(udavgSz))

# Plot Sigma Sz (UP-DOWN)
lineW = 2 # Line thickness
plt.figure(figsize=(10,6), tight_layout=True)
plt.plot(t/T, duavgSz,'-b', linewidth=3);  
plt.ylabel(r'$\langle S_z  (UP-DOWN) \rangle$');plt.xlabel(r'$t/T$')
plt.legend(prop={"size":20})
axes = plt.gca()
axes.xaxis.label.set_size(22)
axes.yaxis.label.set_size(22)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlim(0, tf/T)
plt.ylim(np.min(duavgSz), np.max(duavgSz))


# Plot populations (UP)
lineW = 2 # Line thickness
plt.figure(figsize=(10,6), tight_layout=True)
plt.plot(t/T, uavgpe,'-r', linewidth=3,label=r'$p_e(t)$');  
plt.plot(t/T, uavgpg,'-b', linewidth=3,label=r'$p_g(t)$'); 
plt.ylabel(r'$Populations  (UP)$');plt.xlabel(r'$t/T$')
plt.legend(prop={"size":20})
axes = plt.gca()
axes.xaxis.label.set_size(22)
axes.yaxis.label.set_size(22)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlim(0, tf/T)

# Plot populations (DOWN)
lineW = 2 # Line thickness
plt.figure(figsize=(10,6), tight_layout=True)
plt.plot(t/T, davgpe,'-r', linewidth=3,label=r'$p_e(t)$');  
plt.plot(t/T, davgpg,'-b', linewidth=3,label=r'$p_g(t)$'); 
plt.ylabel(r'$Populations  (DOWN)$');plt.xlabel(r'$t/T$')
plt.legend(prop={"size":20})
axes = plt.gca()
axes.xaxis.label.set_size(22)
axes.yaxis.label.set_size(22)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlim(0, tf/T)

# Plot populations (UP+DOWN)
lineW = 2 # Line thickness
plt.figure(figsize=(10,6), tight_layout=True)
plt.plot(t/T, udavgpe,'-r', linewidth=3,label=r'$p_e(t)$');  
plt.plot(t/T, udavgpg,'-b', linewidth=3,label=r'$p_g(t)$'); 
plt.ylabel(r'$Populations  (UP+DOWN)$');plt.xlabel(r'$t/T$')
plt.legend(prop={"size":20})
axes = plt.gca()
axes.xaxis.label.set_size(22)
axes.yaxis.label.set_size(22)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlim(0, tf/T)

# Plot populations (UP-DOWN)
lineW = 2 # Line thickness
plt.figure(figsize=(10,6), tight_layout=True)
plt.plot(t/T, duavgpe,'-r', linewidth=3,label=r'$p_e(t)$');  
plt.plot(t/T, duavgpg,'-b', linewidth=3,label=r'$p_g(t)$'); 
plt.ylabel(r'$Populations  (UP-DOWN)$');plt.xlabel(r'$t/T$')
plt.legend(prop={"size":20})
axes = plt.gca()
axes.xaxis.label.set_size(22)
axes.yaxis.label.set_size(22)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlim(0, tf/T)

# Plot Coherence (UP)
lineW = 2 # Line thickness
plt.figure(figsize=(10,6), tight_layout=True)
plt.plot(t/T, uC,'-b', linewidth=3);  
plt.ylabel(r'$C(t)  (UP)$');plt.xlabel(r'$t/T$')
plt.legend(prop={"size":20})
axes = plt.gca()
axes.xaxis.label.set_size(22)
axes.yaxis.label.set_size(22)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlim(0, tf/T)

# Plot Coherence (DOWN)
lineW = 2 # Line thickness
plt.figure(figsize=(10,6), tight_layout=True)
plt.plot(t/T, dC,'-b', linewidth=3);  
plt.ylabel(r'$C(t)  (DOWN)$');plt.xlabel(r'$t/T$')
plt.legend(prop={"size":20})
axes = plt.gca()
axes.xaxis.label.set_size(22)
axes.yaxis.label.set_size(22)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlim(0, tf/T)

# Plot Coherence (UP+DOWN)
lineW = 2 # Line thickness
plt.figure(figsize=(10,6), tight_layout=True)
plt.plot(t/T, udC,'-b', linewidth=3);  
plt.ylabel(r'$C(t)  (UP+DOWN)$');plt.xlabel(r'$t/T$')
plt.legend(prop={"size":20})
axes = plt.gca()
axes.xaxis.label.set_size(22)
axes.yaxis.label.set_size(22)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlim(0, tf/T)

# Plot Coherence (UP-DOWN)
lineW = 2 # Line thickness
plt.figure(figsize=(10,6), tight_layout=True)
plt.plot(t/T, duC,'-b', linewidth=3);  
plt.ylabel(r'$C(t)  (UP-DOWN)$');plt.xlabel(r'$t/T$')
plt.legend(prop={"size":20})
axes = plt.gca()
axes.xaxis.label.set_size(22)
axes.yaxis.label.set_size(22)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlim(0, tf/T)

# Plot diference populations (UP)
lineW = 2 # Line thickness
plt.figure(figsize=(10,6), tight_layout=True)
plt.plot(t/T, uavgpe-uavgpg,'-r', linewidth=3,label=r'$p_e(t)-p_g(t)$');  
plt.ylabel(r'$p_e-p_g  (UP)$');plt.xlabel(r'$t/T$')
plt.legend(prop={"size":20})
axes = plt.gca()
axes.xaxis.label.set_size(22)
axes.yaxis.label.set_size(22)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlim(0, tf/T)

# Plot diference populations (DOWN)
lineW = 2 # Line thickness
plt.figure(figsize=(10,6), tight_layout=True)
plt.plot(t/T, davgpe-davgpg,'-r', linewidth=3,label=r'$p_e(t)-p_g(t)$');  
plt.ylabel(r'$p_e-p_g  (DOWN)$');plt.xlabel(r'$t/T$')
plt.legend(prop={"size":20})
axes = plt.gca()
axes.xaxis.label.set_size(22)
axes.yaxis.label.set_size(22)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlim(0, tf/T)

# Plot diference populations (UP+DOWN)
lineW = 2 # Line thickness
plt.figure(figsize=(10,6), tight_layout=True)
plt.plot(t/T, udavgpe-udavgpg,'-r', linewidth=3,label=r'$p_e(t)-p_g(t)$');  
plt.ylabel(r'$p_e-p_g  (UP+DOWN)$');plt.xlabel(r'$t/T$')
plt.legend(prop={"size":20})
axes = plt.gca()
axes.xaxis.label.set_size(22)
axes.yaxis.label.set_size(22)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlim(0, tf/T)

# Plot diference populations (UP-DOWN)
lineW = 2 # Line thickness
plt.figure(figsize=(10,6), tight_layout=True)
plt.plot(t/T, duavgpe-duavgpg,'-r', linewidth=3,label=r'$p_e(t)-p_g(t)$');  
plt.ylabel(r'$p_e-p_g  (UP-DOWN)$');plt.xlabel(r'$t/T$')
plt.legend(prop={"size":20})
axes = plt.gca()
axes.xaxis.label.set_size(22)
axes.yaxis.label.set_size(22)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlim(0, tf/T)


# Fiteo de UP diference population 

DP = uavgpe - uavgpg

def fitfunction(t, a, b, Gamma):
    return a*np.exp(-Gamma*t) + b

# Fitting
popt, _ = curve_fit(fitfunction, DP, t)

a, b, Gamma = popt


# Fit curve
# define a sequence of inputs between the smallest and largest known inputs
#t_line = np.arange(0, 20, 1)
# calculate the output for the range
DP_line = fitfunction(t, a, b, Gamma) 

# Plot diference populations (UP)
#lineW = 2 # Line thickness
#plt.figure(figsize=(10,6), tight_layout=True)
#plt.plot(t/T, uavgpe-uavgpg,'-r', linewidth=3,label=r'$p_e(t)-p_g(t)$');  
#plt.ylabel(r'$p_e-p_g  (UP)$');plt.xlabel(r'$t/T$')
#plt.legend(prop={"size":20})
#axes = plt.gca()
#axes.xaxis.label.set_size(22)
#axes.yaxis.label.set_size(22)
#plt.xticks(fontsize=20)
#plt.yticks(fontsize=20)
#plt.xlim(0, tf/T)

#DP = DP[0][:]

# Plot data
lineW = 2 # Line thickness
plt.figure(figsize=(10,6), tight_layout=True)
plt.scatter(t, DP, s= 20,label=r'$Experimental \; data$'); 
plt.plot(t, DP_line,'-r', linewidth=3,label=r'$Fit \; curve$'); 
plt.ylabel(r'$p_e-p_g  (UP)$');plt.xlabel(r'$t/T$')
axes = plt.gca()
axes.xaxis.label.set_size(22)
axes.yaxis.label.set_size(22)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(prop={"size":20})
plt.grid()


print("--- %s seconds ---" % (time.time() - start_time))
#############################################################