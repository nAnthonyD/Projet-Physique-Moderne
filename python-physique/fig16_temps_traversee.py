"""
fig16_temps_traversee.py  --  SLIDE 16 : Temps de traversee
-----------------------------------------------------------
Mesure operationnelle du temps de traversee.

PRINCIPE (a expliquer au jury) :
  La notion de "temps tunnel" n'a PAS de definition universelle en physique.
  On adopte ici une definition OPERATIONNELLE claire, basee sur le suivi du
  barycentre <x>(t) du paquet transmis :

    tau_0 (libre)  : temps que met le paquet LIBRE (V0=0) pour que son
                     barycentre parcoure la distance L (largeur barriere).
                     Theorie : tau_0 = L / v_g = L / k0.

    tau_t (tunnel) : on suit le barycentre du paquet TRANSMIS (partie x>x_sortie),
                     qui se propage librement a la vitesse v_g apres la barriere.
                     On extrapole sa trajectoire lineaire pour trouver l'instant
                     ou il "sort" de la barriere (x = x_sortie), puis on compte
                     a partir de l'instant ou le barycentre incident "entre"
                     (x = x_entree).

  Si tau_t < tau_0 : le paquet transmis parait "en avance" -> effet Hartman.
  (Ce n'est PAS un signal supraluminique : la barriere reshape le paquet en
   privilegiant son front avant.)
"""

import numpy as np
import matplotlib.pyplot as plt
from tunnel_solver import gaussian_packet, barriere, evolue_CN

# --- Parametres ---
nx = 4000
x = np.linspace(-150, 150, nx)
dx = x[1] - x[0]
dt = 0.005
nt = 14000

x0, k0, sigma = -60.0, 2.0, 8.0
V0, L = 2.5, 1.0
x_entree, x_sortie = 0.0, L
E = k0**2 / 2
vg = k0  # vitesse de groupe (hbar=m=1)

psi0 = gaussian_packet(x, x0, k0, sigma)


def barycentre_region(psi, x, dx, masque):
    """Barycentre de |psi|^2 restreint a une region (masque booleen)."""
    d = np.abs(psi) ** 2
    poids = np.sum(d[masque]) * dx
    if poids < 1e-6:
        return np.nan, poids
    return np.sum(x[masque] * d[masque]) * dx / poids, poids


# ============================================================
# CAS 1 : barriere (tunnel)
# ============================================================
V = barriere(x, V0, x_entree, L)
PSI = evolue_CN(psi0, V, dx, dt, nt)
temps = np.arange(nt + 1) * dt

masque_trans = x > x_sortie
xc_trans, w_trans = [], []
for n in range(nt + 1):
    xc, w = barycentre_region(PSI[n], x, dx, masque_trans)
    xc_trans.append(xc)
    w_trans.append(w)
xc_trans = np.array(xc_trans)
w_trans = np.array(w_trans)
T_final = w_trans[-1]

# On ajuste une droite sur la trajectoire du barycentre transmis,
# une fois qu'il est clairement detache de la barriere (xc > x_sortie + 15)
ok = (~np.isnan(xc_trans)) & (xc_trans > x_sortie + 15) & (xc_trans < 120)
p = np.polyfit(temps[ok], xc_trans[ok], 1)
v_trans, b_trans = p  # xc = v_trans * t + b_trans
t_sortie = (x_sortie - b_trans) / v_trans  # instant ou le transmis etait en x_sortie

# Instant ou le barycentre INCIDENT entre dans la barriere (= x_entree),
# estime sur le mouvement libre du paquet complet avant impact :
# barycentre global ~ x0 + vg t  => t_entree = (x_entree - x0)/vg
t_entree = (x_entree - x0) / vg
tau_t = t_sortie - t_entree

# ============================================================
# CAS 2 : libre (V0 = 0) pour tau_0
# ============================================================
PSI0 = evolue_CN(psi0, np.zeros_like(x), dx, dt, nt)
xc_libre = np.array([np.sum(x * np.abs(PSI0[n])**2) * dx for n in range(nt + 1)])
# temps de passage en x_entree et x_sortie pour le paquet libre
t_libre_entree = np.interp(x_entree, xc_libre, temps)
t_libre_sortie = np.interp(x_sortie, xc_libre, temps)
tau_0 = t_libre_sortie - t_libre_entree

print(f"E={E}, V0={V0}, L={L}, v_g={vg}")
print(f"T_final = {T_final:.3f}")
print(f"tau_0 (libre, theorie L/v_g = {L/vg:.3f}) ~ mesure {tau_0:.3f}")
print(f"tau_t (tunnel) = {tau_t:.3f}")
print(f"Retard tunnel  = tau_t - tau_0 = {tau_t - tau_0:.3f}")

# ----------------------------------------------------------------------
# Figure : trajectoires des barycentres
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(temps, xc_libre, label="Paquet libre (V0=0)", color="green", lw=1.8)
ax.plot(temps[ok], xc_trans[ok], ".", ms=3, color="#1f4e8c",
        label="Barycentre paquet transmis")
ax.plot(temps, v_trans * temps + b_trans, "--", color="#1f4e8c",
        label="Ajustement lineaire (transmis)")
ax.axhline(x_entree, color="crimson", ls=":", lw=1)
ax.axhline(x_sortie, color="crimson", ls=":", lw=1)
ax.axhspan(x_entree, x_sortie, color="crimson", alpha=0.15, label="Barriere")
ax.set_xlabel("temps t")
ax.set_ylabel(r"position du barycentre $\langle x \rangle$")
ax.set_ylim(-70, 120)
ax.set_title("Suivi du barycentre : mesure du temps de traversee")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig("slide16_temps_traversee.png", dpi=130)
print("-> slide16_temps_traversee.png")
