"""
fig_temps_influence.py  --  SLIDES 17 & 18 version TEMPS (enonce 1d / 1e)
-------------------------------------------------------------------------
Influence de la geometrie de la barriere sur les TEMPS de traversee
(et non plus seulement sur la transmission T).

Methode robuste : on place un detecteur loin apres la barriere (x_det) et on
compare l'instant d'arrivee du paquet TRANSMIS a celui du paquet LIBRE.
  delai(a) = t_transmis - t_libre        (l'erreur de vitesse numerique s'annule)
  tau_0(a) = a / v_g                      (temps libre pour parcourir la largeur a)
  tau_t(a) = tau_0(a) + delai(a)          (temps de traversee de la barriere)

Resultat marquant : tau_t sature (effet Hartman) alors que tau_0 croit lineairement.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, identity
from scipy.sparse.linalg import splu
from tunnel_solver import gaussian_packet, barriere, transmission_reflexion

HBAR = M = 1.0


def evolve_centroid(psi0, V, dx, dt, nt, x, masque):
    """Evolution Crank-Nicolson en ne gardant que le barycentre de la
    region 'masque' a chaque pas (economie de memoire)."""
    nx = len(psi0)
    c = -(HBAR**2) / (2 * M * dx**2)
    dp = -2 * c * np.ones(nx) + V
    do = c * np.ones(nx - 1)
    H = diags([do, dp, do], [-1, 0, 1], format="csc")
    I = identity(nx, format="csc")
    f = 1j * dt / (2 * HBAR)
    A = (I + f * H).tocsc()
    B = (I - f * H).tocsc()
    lu = splu(A)
    psi = psi0.copy()
    xc = np.full(nt + 1, np.nan)
    d = np.abs(psi) ** 2
    w = np.sum(d[masque]) * dx
    if w > 1e-9:
        xc[0] = np.sum(x[masque] * d[masque]) * dx / w
    for n in range(nt):
        psi = lu.solve(B.dot(psi))
        d = np.abs(psi) ** 2
        w = np.sum(d[masque]) * dx
        if w > 1e-9:
            xc[n + 1] = np.sum(x[masque] * d[masque]) * dx / w
    return xc, psi


# --- domaine ---
nx = 4000
x = np.linspace(-150, 150, nx)
dx = x[1] - x[0]
dt = 0.005
nt = 13000
t = np.arange(nt + 1) * dt
x0, k0, sigma = -60.0, 2.0, 8.0
E = k0**2 / 2
vg = k0
x_det = 40.0
psi0 = gaussian_packet(x, x0, k0, sigma)


def temps_arrivee(xc):
    good = ~np.isnan(xc)
    return np.interp(x_det, xc[good], t[good])


# reference libre (une seule fois)
xcL, _ = evolve_centroid(psi0, np.zeros_like(x), dx, dt, nt, x, x > 0.0)
t_libre = temps_arrivee(xcL)

# ======================================================================
# SLIDE 17 : influence de la largeur a  (V0 fixe = 2.5, regime tunnel)
# ======================================================================
V0_fixe = 2.5
liste_a = np.linspace(0.4, 2.6, 9)
tau0_a, taut_a, T_a = [], [], []
for a in liste_a:
    xcT, psiF = evolve_centroid(psi0, barriere(x, V0_fixe, 0.0, a), dx, dt, nt, x, x > a)
    delai = temps_arrivee(xcT) - t_libre
    tau0 = a / vg
    tau0_a.append(tau0)
    taut_a.append(tau0 + delai)
    Tf, _ = transmission_reflexion(psiF, x, dx, a)
    T_a.append(Tf)
    print(f"a={a:.2f}  T={Tf:.3f}  tau0={tau0:.3f}  delai={delai:+.3f}  taut={tau0+delai:.3f}")

# ======================================================================
# SLIDE 18 : influence de la hauteur V0  (a fixe = 1.0, regime tunnel)
# ======================================================================
a_fixe = 1.0
liste_V0 = np.linspace(2.1, 6.0, 9)
taut_V, T_V = [], []
tau0_fixe = a_fixe / vg
for V0 in liste_V0:
    xcT, psiF = evolve_centroid(psi0, barriere(x, V0, 0.0, a_fixe), dx, dt, nt, x, x > a_fixe)
    delai = temps_arrivee(xcT) - t_libre
    taut_V.append(tau0_fixe + delai)
    Tf, _ = transmission_reflexion(psiF, x, dx, a_fixe)
    T_V.append(Tf)
    print(f"V0={V0:.2f}  T={Tf:.3f}  delai={delai:+.3f}  taut={tau0_fixe+delai:.3f}")

NAVY = "#1f4e8c"
GREEN = "#2e8b57"
CRIM = "crimson"

# ---------- FIGURE SLIDE 17 ----------
fig1, ax = plt.subplots(figsize=(8, 5.2))
ax.plot(liste_a, tau0_a, "o-", color=GREEN, lw=2,
        label=r"$\tau_0(a)=a/v_g$ (libre)")
ax.plot(liste_a, taut_a, "s-", color=NAVY, lw=2,
        label=r"$\tau_t(a)$ (traversee tunnel)")
ax.set_xlabel("largeur a de la barriere")
ax.set_ylabel("temps de traversee")
ax.set_title(r"Influence de la largeur : $\tau_t$ SATURE (effet Hartman)")
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
ax.annotate("le temps libre croit\nlineairement avec a",
            xy=(liste_a[-1], tau0_a[-1]), xytext=(1.3, 1.15),
            fontsize=9, color=GREEN,
            arrowprops=dict(arrowstyle="->", color=GREEN))
ax.annotate("le temps tunnel\nne suit pas : il sature",
            xy=(liste_a[-2], taut_a[-2]), xytext=(1.6, 0.18),
            fontsize=9, color=NAVY,
            arrowprops=dict(arrowstyle="->", color=NAVY))
fig1.tight_layout()
fig1.savefig("slide17_temps_largeur.png", dpi=130)
print("-> slide17_temps_largeur.png")

# ---------- FIGURE SLIDE 18 ----------
fig2, ax2 = plt.subplots(figsize=(8, 5.2))
ax2.plot(liste_V0, taut_V, "s-", color=NAVY, lw=2, label=r"$\tau_t(V_0)$")
ax2.axhline(tau0_fixe, color=GREEN, ls="--", lw=1.5,
            label=r"$\tau_0 = a/v_g$ (libre, ref.)")
ax2.set_xlabel("hauteur V0 de la barriere")
ax2.set_ylabel("temps de traversee")
ax2.set_title(r"Influence de la hauteur sur $\tau_t$  (a = 1, E = 2)")
ax2.legend(fontsize=11)
ax2.grid(alpha=0.3)
fig2.tight_layout()
fig2.savefig("slide18_temps_hauteur.png", dpi=130)
print("-> slide18_temps_hauteur.png")
