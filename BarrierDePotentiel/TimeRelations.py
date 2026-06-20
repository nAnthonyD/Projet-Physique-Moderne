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

def showBarycenterEvolution(WaveSystem, XEndBarrier):
    barycenters,times = WaveSystem.getBarycentersAndTimes(XEndBarrier)

    if len(barycenters) == 0:
        print("Erreur : Aucune donnée détectée après la barrière. Vérifiez si l'onde traverse !")
    else:
        print(f"Données trouvées : {len(barycenters)} points calculés.")
    plt.figure(figsize=(8, 5))
    plt.plot(times, barycenters, marker='o', color="blue", linewidth=0.8, markersize=3)
    plt.xlabel("Temps t")
    plt.ylabel("Position du barycentre <x>")
    plt.title("Évolution du barycentre de la densité de probabilité")
    plt.grid(True)
    plt.show()
