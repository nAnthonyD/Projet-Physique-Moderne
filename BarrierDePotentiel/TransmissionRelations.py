from QuantumWaveClass import *

def WidthInfluenceTransmission(XArray, V0, X0, K0, Sigma, Energy, Dt, Nt):
    Lengths = np.linspace(0.4, 3, 10)
    TNum, TAna = [], []
    print(f"\nCalcul T(L) en cours... (V0={V0})")
    
    for L in Lengths:
        V = createBarrier(XArray, V0, 0, L)
        Psi0 = createGaussianPacket(XArray, X0, K0, Sigma)
        
        WaveSystem = QuantumWave(XArray, Psi0, V)
        PsiFinal = WaveSystem.evolve(Dt, Nt)[-1]
        
        TNum.append(calculateTransmission(PsiFinal, XArray, L))
        TAna.append(getAnalyticalTransmission(Energy, V0, L))

    fig, (Ax1, Ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    Ax1.plot(Lengths, TAna, "-", color="crimson", label="Theorie (onde plane)")
    Ax1.plot(Lengths, TNum, "o", color="#1f4e8c", label="Simulation (paquet)")
    Ax1.set_xlabel("largeur L de la barrière")
    Ax1.set_ylabel("Transmission T")
    Ax1.set_title(f"T(L)  -  V0 = {V0}, E = {Energy}")
    Ax1.legend()
    Ax1.grid(alpha=0.3)
    
    Ax2.semilogy(Lengths, TAna, "-", color="crimson", label="Theorie")
    Ax2.semilogy(Lengths, TNum, "o", color="#1f4e8c", label="Simulation")
    Ax2.set_xlabel("largeur L")
    Ax2.set_ylabel("T (échelle log)")
    Ax2.set_title("Echelle log : la droite confirme T ~ exp(-2 kappa L)")
    Ax2.legend()
    Ax2.grid(alpha=0.3, which="both")
    
    plt.tight_layout()
    plt.show()

def HeightInfluenceTransmission(XArray, X0, K0, Sigma, Energy, Dt, Nt):
    L = 1
    Heights = np.linspace(0.5, 6, 14)
    TNum, TAna = [], []
    print(f"\nCalcul T(V0) en cours... (L={L})")
    
    for V0 in Heights:
        V = createBarrier(XArray, V0, 0, L)
        Psi0 = createGaussianPacket(XArray, X0, K0, Sigma)
        
        WaveSystem = QuantumWave(XArray, Psi0, V)
        PsiFinal = WaveSystem.evolve(Dt, Nt)[-1]
        
        TNum.append(calculateTransmission(PsiFinal, XArray, L))
        TAna.append(getAnalyticalTransmission(Energy, V0, L))

    fig, Ax = plt.subplots(figsize=(8, 5.5))
    Ax.plot(Heights, TAna, "-", color="crimson", label="Theorie (onde plane)")
    Ax.plot(Heights, TNum, "o", color="#1f4e8c", label="Simulation (paquet)")
    Ax.axvline(Energy, color="gray", ls="--", label=f"E = {Energy} (V0 = E)")
    
    Ax.annotate("au-dessus\n(E > V0)", xy=(1.2, 0.55), fontsize=9, color="gray")
    Ax.annotate("tunnel\n(E < V0)", xy=(4.5, 0.4), fontsize=9, color="gray")
    
    Ax.set_xlabel("hauteur V0 de la barrière")
    Ax.set_ylabel("Transmission T")
    Ax.set_title(f"T(V0)  -  L = {L}, E = {Energy}")
    Ax.legend()
    Ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def showClassicalVsQuantum(XArray, BarrierHeight, BarrierWidth, InitialPosition, PacketWidth, Dt):
    print("Génération du graphe Classique vs Quantique...")

    DenseEnergyRatio = np.linspace(0.1, 2.0, 200)
    DenseAnalyticalTransmission = []
    
    for CurrentRatio in DenseEnergyRatio:
        CurrentEnergy = CurrentRatio * BarrierHeight
        AnalyticalResult = getAnalyticalTransmission(CurrentEnergy, BarrierHeight, BarrierWidth)
        DenseAnalyticalTransmission.append(AnalyticalResult)
        

    SimulationEnergyRatio = np.linspace(0.15, 1.9, 10)
    SimulationNumericalTransmission = []
    
    for CurrentRatio in SimulationEnergyRatio:
        CurrentEnergy = CurrentRatio * BarrierHeight
        WaveNumber = np.sqrt(2 * M * CurrentEnergy)
        
        GroupVelocity = WaveNumber / M
        SafetyMargin = 20.0
        TargetDistance = abs(InitialPosition) + BarrierWidth + SafetyMargin
        
        TotalTime = TargetDistance / GroupVelocity
        NumberOfSteps = int(TotalTime / Dt)
        
        PotentialBarrier = createBarrier(XArray, BarrierHeight, 0.0, BarrierWidth)
        InitialWave = createGaussianPacket(XArray, InitialPosition, WaveNumber, PacketWidth)
        QuantumSystem = QuantumWave(XArray, InitialWave, PotentialBarrier)
        
        FinalWave = QuantumSystem.evolve(Dt, NumberOfSteps)[-1]
        NumericalResult = calculateTransmission(FinalWave, XArray, BarrierWidth)
        
        SimulationNumericalTransmission.append(NumericalResult)

    plt.figure(figsize=(9, 6))
    
    plt.axvspan(0.1, 1.0, color='wheat', alpha=0.3)
    plt.text(0.55, 0.95, "Zone interdite\nclassiquement\n(E < V0)\n-> T=0 classique\n-> T>0 quantique", 
             color='goldenrod', ha='center', va='top', fontsize=9)
    
    ClassicalX = [0.1, 1.0, 1.0, 2.0]
    ClassicalY = [0.0, 0.0, 1.0, 1.0]
    plt.plot(ClassicalX, ClassicalY, color='black', linewidth=1.5, label="Classique (tout ou rien)")
    
    plt.plot(DenseEnergyRatio, DenseAnalyticalTransmission, color='crimson', linewidth=1.5, label="Quantique (théorie)")
    plt.plot(SimulationEnergyRatio, SimulationNumericalTransmission, 'o', color='#1f4e8c', label="Quantique (simulation)")
    
    plt.xlim(0.1, 2.05)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("E / V0")
    plt.ylabel("Transmission T")
    plt.title("Effet tunnel : la rupture entre classique et quantique")
    plt.legend(loc='center right')
    plt.grid(True, alpha=0.4)
    
    plt.tight_layout()
    plt.show()