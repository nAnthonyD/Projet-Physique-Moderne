"""
fig19_comparaison.py  --  SLIDE 19 : Comparaison classique / quantique
----------------------------------------------------------------------
On trace la transmission T en fonction du rapport E/V0 :

  - CLASSIQUE : fonction "tout ou rien"
        E < V0  ->  T = 0  (reflexion totale, traversee IMPOSSIBLE)
        E > V0  ->  T = 1  (la particule passe toujours)

  - QUANTIQUE : courbe continue
        T > 0 meme pour E < V0  ->  c'est l'EFFET TUNNEL
        T < 1 parfois pour E > V0 (reflexion quantique partielle)

  Origine physique (a dire au jury) : la fonction d'onde ne s'annule pas
  dans la zone classiquement interdite ; elle y decroit exponentiellement
  (onde evanescente), ce qui laisse une amplitude non nulle de l'autre cote.
"""

import numpy as np
import matplotlib.pyplot as plt
from tunnel_solver import (gaussian_packet, barriere, evolue_CN,
                           transmission_reflexion, T_analytique)

# --- domaine ---
nx = 3000
x = np.linspace(-120, 120, nx)
dx = x[1] - x[0]
dt = 0.005

L = 1.0
V0 = 2.5  # barriere fixe ; on fait varier E via k0

# Balayage de E en faisant varier k0 -> couvre E/V0 de ~0.2 a ~2
liste_k0 = np.linspace(0.8, 3.1, 12)
ratios, T_num, T_th = [], [], []
for k0 in liste_k0:
    E = k0**2 / 2
    psi0 = gaussian_packet(x, -45.0, k0, 8.0)
    V = barriere(x, V0, 0.0, L)
    # plus k0 est petit, plus il faut de temps pour atteindre/depasser la barriere
    nt = int(12000 * (2.0 / k0))
    PSI = evolue_CN(psi0, V, dx, dt, nt)
    T, _ = transmission_reflexion(PSI[-1], x, dx, L)
    ratios.append(E / V0)
    T_num.append(T)
    T_th.append(T_analytique(E, V0, L))
    print(f"k0={k0:.2f}  E/V0={E/V0:.2f}  T_num={T:.4f}  T_th={T_th[-1]:.4f}")

ratios = np.array(ratios)

# courbe classique (tout ou rien)
r_classique = np.linspace(0.2, 2.0, 500)
T_classique = (r_classique > 1).astype(float)

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(r_classique, T_classique, color="black", lw=2,
        label="Classique (tout ou rien)")
ax.plot(ratios, T_th, "-", color="crimson", label="Quantique (theorie)")
ax.plot(ratios, T_num, "o", color="#1f4e8c", label="Quantique (simulation)")
ax.axvline(1.0, color="gray", ls="--", alpha=0.7)
ax.fill_betweenx([0, 1], 0.2, 1.0, color="orange", alpha=0.08)
ax.text(0.45, 0.85, "Zone interdite\nclassiquement\n(E < V0)\n-> T=0 classique\n-> T>0 quantique",
        fontsize=9, color="darkorange")
ax.set_xlabel("E / V0")
ax.set_ylabel("Transmission T")
ax.set_ylim(-0.03, 1.05)
ax.set_title("Effet tunnel : la rupture entre classique et quantique")
ax.legend(loc="center right")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("slide19_comparaison.png", dpi=130)
print("-> slide19_comparaison.png")
