from QuantumWaveClass import *

DT_TIME = 0.005
NT_TIME = 10000 

def HartmanEffectTraversalTime(XArray, X0, K0, Sigma, Vg):

    V0 = 2.5
    Lengths = np.linspace(0.4, 2.6, 10)
    
    TauTList = []
    Tau0List = [] 
    
    print("\nCalcul de l'influence de la largeur sur le temps de traversée...")
    for L in Lengths:
        V = createBarrier(XArray, V0, 0, L)
        Psi0 = createGaussianPacket(XArray, X0, K0, Sigma)
        
        WaveSystem = QuantumWave(XArray, Psi0, V)
        WaveSystem.evolve(DT_TIME, NT_TIME)
        
        TauT = WaveSystem.calculateTraversalTime(DT_TIME, 0, L, X0, K0)
        
        if TauT is not None:
            TauTList.append(TauT)
            Tau0List.append(L / Vg)

    plt.figure(figsize=(8, 5))
    plt.plot(Lengths, Tau0List, marker='o', color="green", label="Temps classique (L / v_g)")
    plt.plot(Lengths, TauTList, marker='s', color="blue", label="Temps quantique (Effet Hartman)")
    plt.xlabel("Largeur de la barrière L")
    plt.ylabel("Temps de traversée")
    plt.title(f"Saturation du temps de traversée (Effet Hartman) - V0 = {V0}")
    plt.legend()
    plt.grid(True)
    plt.show()

def HeightInfluenceTraversalTime(XArray, X0, K0, Sigma, Vg):

    L = 1.0
    Heights = np.linspace(2.0, 6.0, 10)
    
    TauTList = []
    Tau0Constant = L / Vg  
    
    print("\nCalcul de l'influence de la hauteur sur le temps de traversée...")
    for V0 in Heights:
        V = createBarrier(XArray   , V0, 0, L)
        Psi0 = createGaussianPacket(XArray, X0, K0, Sigma)
        
        WaveSystem = QuantumWave(XArray, Psi0, V)
        WaveSystem.evolve(DT_TIME, NT_TIME)
        
        TauT = WaveSystem.calculateTraversalTime(DT_TIME, 0, L, X0, K0)
        
        if TauT is not None:
            TauTList.append(TauT)

    plt.figure(figsize=(8, 5))
    plt.plot(Heights, TauTList, marker='s', color="blue", label="Temps quantique")
    plt.axhline(Tau0Constant, color="green", linestyle="--", label="Temps classique constant")
    plt.xlabel("Hauteur de la barrière V0")
    plt.ylabel("Temps de traversée")
    plt.title(f"Influence de la hauteur sur le temps de traversée - L = {L}")
    plt.legend()
    plt.grid(True)
    plt.show()

def showBarycenterEvolution(WaveSystem, XStartBarrier, XEndBarrier, X0, K0, Mass=1.0, Hbar=1.0):
    barycenters, times = WaveSystem.getBarycentersAndTimes(XEndBarrier)

    if len(barycenters) < 2:
        print("Erreur : Données insuffisantes pour tracer ou extrapoler la trajectoire.")
        return 
    
    coefs = np.polyfit(times, barycenters, 1)
    VelocityOut = coefs[0]
    PositionOffset = coefs[1]

    tTheory = np.linspace(0, max(times), 100)
    
    xExtrapol = VelocityOut * tTheory + PositionOffset

    groupVelocity = Hbar * K0 / Mass
    xFree = groupVelocity * tTheory + X0

    plt.figure(figsize=(10, 6))
    plt.axhspan(XStartBarrier, XEndBarrier, color='pink', alpha=0.4, label="Barriere")
    plt.plot(tTheory, xFree, color='#006400', linewidth=1.5, label="Paquet libre (V0=0)")
    plt.plot(tTheory, xExtrapol, color='#1f4e8c', linestyle='--', linewidth=1.5, label="Ajustement lineaire (transmis)")
    plt.plot(times, barycenters, marker='.', color="#1f4e8c", linestyle='', markersize=4, label="Barycentre paquet transmis")
    plt.xlabel("temps t")
    plt.ylabel("position du barycentre (x)")
    plt.title("Suivi du barycentre : mesure du temps de traversee")
    plt.legend(loc="upper left", bbox_to_anchor=(0.6, 1.0))
    plt.grid(True, alpha=0.3)
    plt.ylim(X0 - 10, max(barycenters) + 10)
    plt.tight_layout()
    plt.show()

def showBarycenterZoom(WaveSystem, XStartBarrier, XEndBarrier, X0, K0, Mass=1.0, Hbar=1.0):
    Barycenters, Times = WaveSystem.getBarycentersAndTimes(XEndBarrier)

    if len(Barycenters) < 2:
        print("Erreur : Données insuffisantes pour tracer l'extrapolation.")
        return
    LinearCoefficients = np.polyfit(Times, Barycenters, 1)
    VelocityOut = LinearCoefficients[0]
    PositionOffset = LinearCoefficients[1]
    
    GroupVelocity = Hbar * K0 / Mass
    TimeOutExtrapol = (XEndBarrier - PositionOffset) / VelocityOut
    TimeOutFree = (XEndBarrier - X0) / GroupVelocity

    ClassicalTraversalTime = (XEndBarrier - XStartBarrier) / GroupVelocity

    TimeMin = TimeOutFree - 1.5
    TimeMax = TimeOutFree + 1.5
    TimeTheory = np.linspace(TimeMin, TimeMax, 100)

    PositionExtrapol = VelocityOut * TimeTheory + PositionOffset
    PositionFree = GroupVelocity * TimeTheory + X0
    
    plt.figure(figsize=(8, 6))

    plt.axhspan(XStartBarrier, XEndBarrier, color='#f8cdda', alpha=0.7, 
                label=f"Barrière (L={XEndBarrier - XStartBarrier})")

    plt.axhline(XEndBarrier, color='crimson', linestyle=':', linewidth=1.2)
    plt.text(TimeMax - 0.1, XEndBarrier - 0.02, "x=1\n(Sortie)", color='crimson', va='top', ha='center', fontsize=9)

    plt.axhline(XStartBarrier, color='crimson', linestyle=':', linewidth=1.2)
    plt.text(TimeMax - 0.1, XStartBarrier - 0.02, "x=0\n(Entrée)", color='crimson', va='top', ha='center', fontsize=9)

    plt.plot(TimeTheory, PositionFree, color='#2e8b57', linewidth=2.5, label="Libre")
    plt.plot(TimeTheory, PositionExtrapol, color='#1f4e8c', linestyle='--', linewidth=2.5, label="Transmis (Extrapolé)")

    plt.plot(TimeOutFree, XEndBarrier, marker='o', color='#2e8b57', markersize=10, zorder=5)
    plt.plot(TimeOutExtrapol, XEndBarrier, marker='o', color='#1f4e8c', markersize=10, zorder=5)

    ArrowYPosition = XStartBarrier - 0.45
    plt.annotate('', xy=(TimeOutFree - ClassicalTraversalTime, ArrowYPosition), 
                 xytext=(TimeOutFree, ArrowYPosition),
                 arrowprops=dict(arrowstyle='<->', color='#2e8b57', lw=1.5))
    
    plt.text(TimeOutFree - ClassicalTraversalTime / 2, ArrowYPosition - 0.15, 
             f"$\\tau_0 = {ClassicalTraversalTime:.2f}$",
             color='#2e8b57', ha='center', va='top', fontsize=12, fontweight='bold')

    plt.xlim(TimeMin, TimeMax)
    plt.ylim(-1.0, 2.0)
    
    plt.xlabel("Temps t")
    plt.ylabel("Position du barycentre (x)")
    plt.title("Zoom sur la sortie : L'écart devient visible")
    plt.legend(loc="lower right")
    
    Axes = plt.gca()
    for Spine in Axes.spines.values():
        Spine.set_color('black')
        Spine.set_linewidth(1.0)

    plt.tight_layout()
    plt.show()
    