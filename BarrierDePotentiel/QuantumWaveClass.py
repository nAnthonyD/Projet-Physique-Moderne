import numpy as np
from scipy.sparse import diags, identity
from scipy.sparse.linalg import splu    
import matplotlib.pyplot as plt


HBAR = 1
M = 1

DT = 0.005
NT = 3000  

class QuantumWave:

    def __init__(self, x, psi0, V):
        self.x = x
        self.dx = x[1] - x[0]
        self.psi = psi0.copy()
        self.V = V
        self.nx = len(x)

    def evolve(self, dt, nt):
        coeff = -(HBAR**2) / (2 * M * self.dx**2)
        diag_princ = -2 * coeff * np.ones(self.nx) + self.V
        diag_off = coeff * np.ones(self.nx - 1)
        H = diags([diag_off, diag_princ, diag_off], [-1, 0, 1], format="csc")

        I = identity(self.nx, format="csc")
        facteur = 1j * dt / (2 * HBAR)
        A = (I + facteur * H).tocsc()
        B = (I - facteur * H).tocsc()

        solver = splu(A)
        
        psi_history = np.zeros((nt + 1, self.nx), dtype=complex)
        psi_history[0] = self.psi

        for n in range(nt):
            self.psi = solver.solve(B.dot(self.psi))
            psi_history[n + 1] = self.psi
            
        self.history = psi_history
        return psi_history

    def calculateTraversalTime(self, dt, x_start_barrier, x_end_barrier, x0, k0):
        nt = len(self.history)
        start_idx = int(nt * 0.8)
        
        times = []
        barycenters = []
        
        for n in range(start_idx, nt):
            psi_t = self.history[n]
            mask = self.x > x_end_barrier
            density = np.abs(psi_t[mask])**2
            norm = np.sum(density) * self.dx
            
            # Sécurité pour ignorer le bruit numérique si la transmission est quasi-nulle
            if norm > 1e-12:
                barycenter = np.sum(self.x[mask] * density) * self.dx / norm
                barycenters.append(barycenter)
                times.append(n * dt)
                
        if len(times) < 2:
            return None
            
        coefs = np.polyfit(times, barycenters, 1)
        v_out = coefs[0]
        p0 = coefs[1]
        
        t_out = (x_end_barrier - p0) / v_out
        
        v_g = HBAR * k0 / M
        t_in = (x_start_barrier - x0) / v_g
        
        return t_out - t_in
    
    def showQuantumWave(self):
        plt.figure(figsize=(10, 6))
        v_norm = self.V / np.max(self.V) * np.max(np.abs(self.history[0])**2)
        
        plt.plot(self.x, np.abs(self.history[0])**2, label='|ψ(x, t=0)|²', color='gray', linestyle=':')
        plt.plot(self.x, np.abs(self.psi)**2, label='|ψ(x, t=final)|²', color='blue')
        plt.plot(self.x, v_norm, label='Potentiel V(x)', color='red', linestyle='--')
        
        plt.title('Effet Tunnel : État Initial vs Final')
        plt.xlabel('Position x')
        plt.ylabel('Densité de probabilité')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def getTransmission(self, x_end_barrier):
        density = np.abs(self.psi)**2
        self.transmission = np.sum(density[self.x > x_end_barrier]) * self.dx
        return self.transmission

    def getReflection(self, x_start_barrier):
        density = np.abs(self.psi)**2
        self.reflection = np.sum(density[self.x < x_start_barrier]) * self.dx
        return self.reflection


def createGaussianPacket(x, x0, k0, sigma):
    norm = (1 / (2 * np.pi * sigma**2)) ** 0.25
    enveloppe = np.exp(-((x - x0) ** 2) / (4 * sigma**2))
    return norm * enveloppe * np.exp(1j * k0 * x)

def createBarrier(x, V0, x_start, L):
    V = np.zeros_like(x)
    V[(x >= x_start) & (x <= x_start + L)] = V0
    return V

def calculateTransmission(psi, x, x_end_barrier):
    density = np.abs(psi) ** 2
    dx = x[1] - x[0]
    return np.sum(density[x > x_end_barrier]) * dx

def getAnalyticalTransmission(E, V0, L):
    if E == V0:
        return 1 / (1 + (M * V0 * L**2) / (2 * HBAR**2))
    
    if E < V0: 
        kappa = np.sqrt(2 * M * (V0 - E)) / HBAR
        return 1 / (1 + (V0**2 * np.sinh(kappa * L)**2) / (4 * E * (V0 - E)))
    else:  
        k2 = np.sqrt(2 * M * (E - V0)) / HBAR
        return 1 / (1 + (V0**2 * np.sin(k2 * L)**2) / (4 * E * (E - V0)))
    

    
