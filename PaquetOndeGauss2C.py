from numpy import pi, exp, sqrt, real, imag, zeros, linspace, cos
from math import pow
import matplotlib.pyplot as plt

QUALITY = 500

HBAR = 1
M = 1


def GaussWP(a, k0, x, t):
    exp1 = (1 / (8 * (pi**3)))**(1/4)
    exp2 = sqrt((4 * pi * M * a) / ((M * (a**2)) + 2 * HBAR * t * 1j))
    num_exp3 = (M / 4) * ((a**2 * k0 + 2 * x * 1j)**2)
    den_exp3 = (M * (a**2)) + 2 * HBAR * t * 1j
    exp3 = exp(num_exp3 / den_exp3 - (a**2 * (k0**2) / 4))

    return exp1 * exp2 * exp3

def ShowGaussWP(a, k0, x, t):
    interval = linspace(x-10, x+10, QUALITY)
    wave = GaussWP(a, k0, interval, t)
    realside = real(wave)
    imagside = imag(wave)
    fig, ax = plt.subplots()
    ax.plot(interval, realside, linewidth=2, label="réel")
    ax.plot(interval, imagside, linewidth=2, label="imaginaire")
    ax.legend()

    plt.show()

print(GaussWP(1, 2, -10, 0))
ShowGaussWP(1, 2, 2, 0)

