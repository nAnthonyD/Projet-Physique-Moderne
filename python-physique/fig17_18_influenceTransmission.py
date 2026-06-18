"""
fig17_18_influence.py  --  SLIDES 17 & 18
-----------------------------------------
Etudie l'influence de la geometrie de la barriere sur la transmission T,
en comparant a chaque fois la simulation numerique (paquet d'ondes) a la
formule analytique de l'onde plane.

SLIDE 17 : T en fonction de la LARGEUR L  -> decroissance EXPONENTIELLE
SLIDE 18 : T en fonction de la HAUTEUR V0
           + au-dessus de la barriere (E > V0) : RESONANCES de transmission
"""

import numpy as np
import matplotlib.pyplot as plt
from tunnel_solver import (gaussian_packet, barriere, evolue_CN,
                           transmission_reflexion, T_analytique)

# --- Domaine commun ---
nx = 3000
x = np.linspace(-120, 120, nx)
dx = x[1] - x[0]
dt = 0.005

x0, k0, sigma = -45.0, 2.0, 8.0
E = k0**2 / 2
psi0 = gaussian_packet(x, x0, k0, sigma)


def mesure_T(V0, L, nt=10000):
    """Simule et retourne le T numerique apres separation des paquets."""
    V = barriere(x, V0, 0.0, L)
    PSI = evolue_CN(psi0, V, dx, dt, nt)
    T, _ = transmission_reflexion(PSI[-1], x, dx, L)
    return T


# ======================================================================
# SLIDE 17 : T(L)  a V0 fixe (regime tunnel E < V0)
# ======================================================================
V0_fixe = 2.5
liste_L = np.linspace(0.4, 3.0, 10)
T_num_L = np.array([mesure_T(V0_fixe, L) for L in liste_L])
T_th_L = np.array([T_analytique(E, V0_fixe, L) for L in liste_L])
print("SLIDE 17 : T(L) a V0 =", V0_fixe)
for L, tn, tt in zip(liste_L, T_num_L, T_th_L):
    print(f"  L={L:.2f}  T_num={tn:.4f}  T_th={tt:.4f}")

fig1, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
a1.plot(liste_L, T_th_L, "-", color="crimson", label="Theorie (onde plane)")
a1.plot(liste_L, T_num_L, "o", color="#1f4e8c", label="Simulation (paquet)")
a1.set_xlabel("largeur L de la barriere")
a1.set_ylabel("Transmission T")
a1.set_title(f"T(L)  -  V0 = {V0_fixe}, E = {E}")
a1.legend(); a1.grid(alpha=0.3)
# echelle log -> droite => decroissance exponentielle
a2.semilogy(liste_L, T_th_L, "-", color="crimson", label="Theorie")
a2.semilogy(liste_L, T_num_L, "o", color="#1f4e8c", label="Simulation")
a2.set_xlabel("largeur L")
a2.set_ylabel("T (echelle log)")
a2.set_title("Echelle log : la droite confirme T ~ exp(-2 kappa L)")
a2.legend(); a2.grid(alpha=0.3, which="both")
fig1.tight_layout()
fig1.savefig("slide17_influence_L.png", dpi=130)
print("-> slide17_influence_L.png\n")

# ======================================================================
# SLIDE 18 : T(V0)  a L fixe -- traverse le regime tunnel ET au-dessus
# ======================================================================
L_fixe = 1.0
liste_V0 = np.linspace(0.5, 6.0, 14)  # de E>V0 (au-dessus) a E<V0 (tunnel)
T_num_V = np.array([mesure_T(V0, L_fixe) for V0 in liste_V0])
T_th_V = np.array([T_analytique(E, V0, L_fixe) for V0 in liste_V0])
print("SLIDE 18 : T(V0) a L =", L_fixe)
for V0, tn, tt in zip(liste_V0, T_num_V, T_th_V):
    regime = "tunnel" if V0 > E else "au-dessus"
    print(f"  V0={V0:.2f} ({regime})  T_num={tn:.4f}  T_th={tt:.4f}")

fig2, ax = plt.subplots(figsize=(8, 5.5))
ax.plot(liste_V0, T_th_V, "-", color="crimson", label="Theorie (onde plane)")
ax.plot(liste_V0, T_num_V, "o", color="#1f4e8c", label="Simulation (paquet)")
ax.axvline(E, color="gray", ls="--", label=f"E = {E} (V0 = E)")
ax.annotate("au-dessus\n(E > V0)", xy=(1.0, 0.55), fontsize=9, color="gray")
ax.annotate("tunnel\n(E < V0)", xy=(4.5, 0.4), fontsize=9, color="gray")
ax.set_xlabel("hauteur V0 de la barriere")
ax.set_ylabel("Transmission T")
ax.set_title(f"T(V0)  -  L = {L_fixe}, E = {E}")
ax.legend(); ax.grid(alpha=0.3)
fig2.tight_layout()
fig2.savefig("slide18_influence_V0.png", dpi=130)
print("-> slide18_influence_V0.png")