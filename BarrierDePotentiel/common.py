import numpy as np
from scipy.sparse import diags, identity
from scipy.sparse.linalg import splu    
import matplotlib.pyplot as plt


# --- Constantes physiques ---
HBAR = 1.0
M = 1.0

class QuantumWave:

    def __init__(self, x, psi0, V):
        self.x = x
        self.dx = x[1] - x[0]
        self.psi = psi0.copy()
        self.V = V
        self.nx = len(x)

    def evolve(self, dt, nt):
        coeff = -(HBAR**2) / (2.0 * M * self.dx**2)
        diag_princ = -2.0 * coeff * np.ones(self.nx) + self.V
        diag_off = coeff * np.ones(self.nx - 1)
        H = diags([diag_off, diag_princ, diag_off], [-1, 0, 1], format="csc")

        I = identity(self.nx, format="csc")
        facteur = 1j * dt / (2.0 * HBAR)
        A = (I + facteur * H).tocsc()
        B = (I - facteur * H).tocsc()

        solver = splu(A)
        
        psi_history = np.zeros((nt + 1, self.nx), dtype=complex)
        psi_history[0] = self.psi

        for n in range(nt):
            self.psi = solver.solve(B.dot(self.psi))
            psi_history[n + 1] = self.psi
            
        return psi_history

def createGaussianPacket(x, x0, k0, sigma):
    norm = (1.0 / (2.0 * np.pi * sigma**2)) ** 0.25
    enveloppe = np.exp(-((x - x0) ** 2) / (4.0 * sigma**2))
    return norm * enveloppe * np.exp(1j * k0 * x)

def createRectangularBarrier(x, V0, x_start, L):
    V = np.zeros_like(x)
    V[(x >= x_start) & (x <= x_start + L)] = V0
    return V

def calculateTransmission(psi, x, x_end_barrier):
    densite = np.abs(psi) ** 2
    dx = x[1] - x[0]
    return np.sum(densite[x > x_end_barrier]) * dx

def getAnalyticalTransmission(E, V0, L):
    if E == V0:
        return 1.0 / (1.0 + (M * V0 * L**2) / (2.0 * HBAR**2))
    
    if E < V0: 
        kappa = np.sqrt(2 * M * (V0 - E)) / HBAR
        return 1.0 / (1.0 + (V0**2 * np.sinh(kappa * L)**2) / (4.0 * E * (V0 - E)))
    else:  
        k2 = np.sqrt(2 * M * (E - V0)) / HBAR
        return 1.0 / (1.0 + (V0**2 * np.sin(k2 * L)**2) / (4.0 * E * (E - V0)))
    
def showQuantumWave(x, psi_initial, psi_final, V):
    plt.figure(figsize=(10, 6))
    v_norm = V / np.max(V) * np.max(np.abs(psi_initial)**2)
    
    plt.plot(x, np.abs(psi_initial)**2, label='|ψ(x, t=0)|²', color='gray', linestyle=':')
    plt.plot(x, np.abs(psi_final)**2, label='|ψ(x, t=final)|²', color='blue')
    plt.plot(x, v_norm, label='Potentiel V(x)', color='red', linestyle='--')
    
    plt.title('Effet Tunnel : État Initial vs Final')
    plt.xlabel('Position x')
    plt.ylabel('Densité de probabilité')
    plt.legend()
    plt.grid(True)
    plt.show()
    
x = np.linspace(-30, 30, 2000) 

V = createRectangularBarrier(x, V0=3.0, x_start=0.0, L=1.0)

psi0 = createGaussianPacket(x, x0=-15.0, k0=2.0, sigma=2.0) 

wave_system = QuantumWave(x, psi0, V)

wave_history = wave_system.evolve(dt=0.01, nt=1250) 

psi_final = wave_history[-1] 
showQuantumWave(x, psi0, psi_final, V)