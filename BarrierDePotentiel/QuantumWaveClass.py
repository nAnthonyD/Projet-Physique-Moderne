import numpy as np
from scipy.sparse import diags, identity
from scipy.sparse.linalg import splu    
import matplotlib.pyplot as plt


HBAR = 1
M = 1

DT = 0.005
NT = 3000  

class QuantumWave:

    def __init__(self, X, Psi0, V):
        self.X = X
        self.Dx = X[1] - X[0]
        self.Psi = Psi0.copy()
        self.V = V
        self.Nx = len(X)

    def evolve(self, Dt, Nt):
        Coeff = -(HBAR**2) / (2 * M * self.Dx**2)
        DiagPrinc = -2 * Coeff * np.ones(self.Nx) + self.V
        DiagOff = Coeff * np.ones(self.Nx - 1)
        H = diags([DiagOff, DiagPrinc, DiagOff], [-1, 0, 1], format="csc")

        I = identity(self.Nx, format="csc")
        Facteur = 1j * Dt / (2 * HBAR)
        A = (I + Facteur * H).tocsc()
        B = (I - Facteur * H).tocsc()

        Solver = splu(A)
        
        PsiHistory = np.zeros((Nt + 1, self.Nx), dtype=complex)
        PsiHistory[0] = self.Psi

        for N in range(Nt):
            self.Psi = Solver.solve(B.dot(self.Psi))
            PsiHistory[N + 1] = self.Psi
            
        self.History = PsiHistory
        return PsiHistory

    def calculateTraversalTime(self, Dt, XStartBarrier, XEndBarrier, X0, K0):
        Nt = len(self.History)
        StartIdx = int(Nt * 0.8)
        
        Times = []
        Barycenters = []
        
        for N in range(StartIdx, Nt):
            PsiT = self.History[N]
            Mask = self.X > XEndBarrier
            Density = np.abs(PsiT[Mask])**2
            Norm = np.sum(Density) * self.Dx
            
            if Norm > 1e-12:
                Barycenter = np.sum(self.X[Mask] * Density) * self.Dx / Norm
                Barycenters.append(Barycenter)
                Times.append(N * Dt)
                
        if len(Times) < 2:
            return None
            
        Coefs = np.polyfit(Times, Barycenters, 1)
        VOut = Coefs[0]
        P0 = Coefs[1]
        
        TOut = (XEndBarrier - P0) / VOut
        
        VG = HBAR * K0 / M
        TIn = (XStartBarrier - X0) / VG
        
        return TOut - TIn

    def getBarycentersAndTimes(self, XEndBarrier):
        Nt = len(self.History)
        StartIdx = int(Nt * 0.8)
        
        Times = []
        Barycenters = []
        
        for N in range(StartIdx, Nt):
            PsiT = self.History[N]
            Mask = self.X > XEndBarrier 
            Density = np.abs(PsiT[Mask])**2
            Norm = np.sum(Density) * self.Dx
            
            if Norm > 1e-10: # Seuil un peu plus large
                Barycenter = np.sum(self.X[Mask] * Density) * self.Dx / Norm
                Barycenters.append(Barycenter)
                Times.append(N * DT)
                
        return Barycenters, Times
    
    def showQuantumWave(self):
        plt.figure(figsize=(10, 6))
        VNorm = self.V / np.max(self.V) * np.max(np.abs(self.History[0])**2)
        
        plt.plot(self.X, np.abs(self.History[0])**2, label='|ψ(x, t=0)|²', color='gray', linestyle=':')
        plt.plot(self.X, np.abs(self.Psi)**2, label='|ψ(x, t=final)|²', color='blue')
        plt.plot(self.X, VNorm, label='Potentiel V(x)', color='red', linestyle='--')
        
        plt.title('Effet Tunnel : État Initial vs Final')
        plt.xlabel('Position x')
        plt.ylabel('Densité de probabilité')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def getTransmission(self, XEndBarrier):
        Density = np.abs(self.Psi)**2
        self.Transmission = np.sum(Density[self.X > XEndBarrier]) * self.Dx
        return self.Transmission

    def getReflection(self, XStartBarrier):
        Density = np.abs(self.Psi)**2
        self.Reflection = np.sum(Density[self.X < XStartBarrier]) * self.Dx
        return self.Reflection


def createGaussianPacket(X, X0, K0, Sigma):
    Norm = (1 / (2 * np.pi * Sigma**2)) ** 0.25
    Enveloppe = np.exp(-((X - X0) ** 2) / (4 * Sigma**2))
    return Norm * Enveloppe * np.exp(1j * K0 * X)

def createBarrier(X, V0, XStart, L):
    V = np.zeros_like(X)
    V[(X >= XStart) & (X <= XStart + L)] = V0
    return V

def calculateTransmission(Psi, X, XEndBarrier):
    Density = np.abs(Psi) ** 2
    Dx = X[1] - X[0]
    return np.sum(Density[X > XEndBarrier]) * Dx

def getAnalyticalTransmission(E, V0, L):
    if E == V0:
        return 1 / (1 + (M * V0 * L**2) / (2 * HBAR**2))
    
    if E < V0: 
        Kappa = np.sqrt(2 * M * (V0 - E)) / HBAR
        return 1 / (1 + (V0**2 * np.sinh(Kappa * L)**2) / (4 * E * (V0 - E)))
    else:  
        K2 = np.sqrt(2 * M * (E - V0)) / HBAR
        return 1 / (1 + (V0**2 * np.sin(K2 * L)**2) / (4 * E * (E - V0)))
    

    
