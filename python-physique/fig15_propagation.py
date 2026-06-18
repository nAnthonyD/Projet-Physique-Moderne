import numpy as np
import matplotlib.pyplot as plt
from tunnel_solver import (gaussian_packet, barriere, evolue_CN,
                           norme, transmission_reflexion)

# --- Domaine spatial et temporel ---
nx = 3000
x = np.linspace(-120, 120, nx)
dx = x[1] - x[0]
dt = 0.005
nt = 11000

# --- Parametres physiques (regime tunnel : E < V0) ---
x0, k0, sigma = -50.0, 2.0, 8.0
V0, L, x_debut = 2.5, 1.0, 0.0
E = k0**2 / 2

psi0 = gaussian_packet(x, x0, k0, sigma)
V = barriere(x, V0, x_debut, L)

# --- Evolution ---
PSI = evolue_CN(psi0, V, dx, dt, nt)
densite = np.abs(PSI) ** 2

T_fin, R_fin = transmission_reflexion(PSI[-1], x, dx, x_debut + L)
print(f"E = {E}, V0 = {V0}, L = {L}")
print(f"T = {T_fin:.3f}, R = {R_fin:.3f}, norme finale = {norme(PSI[-1], dx):.4f}")

# ----------------------------------------------------------------------
# Figure 1 : 4 instantanes
# ----------------------------------------------------------------------

instants = [0, int(0.42 * nt), int(0.55 * nt), nt]
titres = ["Paquet incident", "Arrivee sur la barriere",
          "Separation", "Transmis + reflechi"]

fig, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
ymax = 1.05 * densite[0].max()
for ax, n, titre in zip(axes, instants, titres):
    ax.plot(x, densite[n], color="#1f4e8c", lw=1.6)
    ax.fill_between(x, densite[n], color="#1f4e8c", alpha=0.25)
    ax.axvspan(x_debut, x_debut + L, color="crimson", alpha=0.25,
               label=f"Barriere V0={V0}")
    ax.set_ylim(0, ymax)
    ax.set_ylabel(r"$|\Psi|^2$")
    ax.set_title(f"{titre}   (t = {n*dt:.1f})", fontsize=10, loc="left")
    ax.legend(loc="upper right", fontsize=8)
axes[-1].set_xlabel("x")
fig.suptitle("Propagation d'un paquet d'ondes et effet tunnel", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.98])
fig.savefig("slide15_snapshots.png", dpi=130)
print("-> slide15_snapshots.png")