from QuantumWaveClass import *

DT_Time = 0.005
NT_Time = 10000  # Arrête la simulation à t = 50 secondes

def HartmanEffectTraversalTime(x_array, x0, k0, sigma, vg):

    V0 = 2.5
    lengths = np.linspace(0.4, 2.6, 10)
    
    tau_t_list = []
    tau_0_list = [] 
    
    print("\nCalcul de l'influence de la largeur sur le temps de traversée...")
    for L in lengths:
        V = createBarrier(x_array, V0, 0, L)
        psi0 = createGaussianPacket(x_array, x0, k0, sigma)
        
        wave_system = QuantumWave(x_array, psi0, V)
        wave_system.evolve(DT_Time, NT_Time)
        
        tau_t = wave_system.calculateTraversalTime(DT_Time, 0, L, x0, k0)
        
        if tau_t is not None:
            tau_t_list.append(tau_t)
            tau_0_list.append(L / vg)

    plt.figure(figsize=(8, 5))
    plt.plot(lengths, tau_0_list, marker='o', color="green", label="Temps classique (L / v_g)")
    plt.plot(lengths, tau_t_list, marker='s', color="blue", label="Temps quantique (Effet Hartman)")
    plt.xlabel("Largeur de la barrière L")
    plt.ylabel("Temps de traversée")
    plt.title(f"Saturation du temps de traversée (Effet Hartman) - V0 = {V0}")
    plt.legend()
    plt.grid(True)
    plt.show()

def HeightInfluenceTraversalTime(x_array, x0, k0, sigma, vg):

    L = 1.0
    heights = np.linspace(2.0, 6.0, 10)
    
    tau_t_list = []
    tau_0_constant = L / vg  
    
    print("\nCalcul de l'influence de la hauteur sur le temps de traversée...")
    for V0 in heights:
        V = createBarrier(x_array   , V0, 0, L)
        psi0 = createGaussianPacket(x_array, x0, k0, sigma)
        
        wave_system = QuantumWave(x_array, psi0, V)
        wave_system.evolve(DT_Time, NT_Time)
        
        tau_t = wave_system.calculateTraversalTime(DT_Time, 0, L, x0, k0)
        
        if tau_t is not None:
            tau_t_list.append(tau_t)

    plt.figure(figsize=(8, 5))
    plt.plot(heights, tau_t_list, marker='s', color="blue", label="Temps quantique")
    plt.axhline(tau_0_constant, color="green", linestyle="--", label="Temps classique constant")
    plt.xlabel("Hauteur de la barrière V0")
    plt.ylabel("Temps de traversée")
    plt.title(f"Influence de la hauteur sur le temps de traversée - L = {L}")
    plt.legend()
    plt.grid(True)
    plt.show()

