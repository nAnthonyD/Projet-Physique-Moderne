from QuantumWaveClass import *

def WidthInfluenceTransmission(x_array, v0, x0, k0, sigma, energy, dt, nt):
    lengths = np.linspace(0.4, 3, 10)
    T_num, T_ana = [], []
    print(f"\nCalcul T(L) en cours... (V0={v0})")
    
    for L in lengths:
        V = createBarrier(x_array, v0, 0, L)
        psi0 = createGaussianPacket(x_array, x0, k0, sigma)
        
        wave_system = QuantumWave(x_array, psi0, V)
        psi_final = wave_system.evolve(dt, nt)[-1]
        
        T_num.append(calculateTransmission(psi_final, x_array, L))
        T_ana.append(getAnalyticalTransmission(energy, v0, L))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(lengths, T_ana, "-", color="crimson", label="Theorie (onde plane)")
    ax1.plot(lengths, T_num, "o", color="#1f4e8c", label="Simulation (paquet)")
    ax1.set_xlabel("largeur L de la barrière")
    ax1.set_ylabel("Transmission T")
    ax1.set_title(f"T(L)  -  V0 = {v0}, E = {energy}")
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    ax2.semilogy(lengths, T_ana, "-", color="crimson", label="Theorie")
    ax2.semilogy(lengths, T_num, "o", color="#1f4e8c", label="Simulation")
    ax2.set_xlabel("largeur L")
    ax2.set_ylabel("T (échelle log)")
    ax2.set_title("Echelle log : la droite confirme T ~ exp(-2 kappa L)")
    ax2.legend()
    ax2.grid(alpha=0.3, which="both")
    
    plt.tight_layout()
    plt.show()

def HeightInfluenceTransmission(x_array, x0, k0, sigma, energy, dt, nt):
    L = 1
    heights = np.linspace(0.5, 6, 14)
    T_num, T_ana = [], []
    print(f"\nCalcul T(V0) en cours... (L={L})")
    
    for V0 in heights:
        V = createBarrier(x_array, V0, 0, L)
        psi0 = createGaussianPacket(x_array, x0, k0, sigma)
        
        wave_system = QuantumWave(x_array, psi0, V)
        psi_final = wave_system.evolve(dt, nt)[-1]
        
        T_num.append(calculateTransmission(psi_final, x_array, L))
        T_ana.append(getAnalyticalTransmission(energy, V0, L))

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(heights, T_ana, "-", color="crimson", label="Theorie (onde plane)")
    ax.plot(heights, T_num, "o", color="#1f4e8c", label="Simulation (paquet)")
    ax.axvline(energy, color="gray", ls="--", label=f"E = {energy} (V0 = E)")
    
    ax.annotate("au-dessus\n(E > V0)", xy=(1.2, 0.55), fontsize=9, color="gray")
    ax.annotate("tunnel\n(E < V0)", xy=(4.5, 0.4), fontsize=9, color="gray")
    
    ax.set_xlabel("hauteur V0 de la barrière")
    ax.set_ylabel("Transmission T")
    ax.set_title(f"T(V0)  -  L = {L}, E = {energy}")
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()

