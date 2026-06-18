from numpy import pi, exp, sqrt, real, imag, zeros, linspace, cos
import matplotlib.pyplot as plt

QUALITY = 500


def PlaneWave(amp, k, w, x, t):
    base = amp * exp(((k*x) - (w*t))*1j)
    return  base
    

def ShowPlaneWave(amp, k, w, x, t):
    interval = linspace(x-10, x+10, QUALITY)
    wave = PlaneWave(amp, k, w, interval, t)
    realside = real(wave)
    imagside = imag(wave)
    fig, ax = plt.subplots()
    ax.plot(interval, realside, linewidth=2, label="réel")
    ax.plot(interval, imagside, linewidth=2, label="imaginaire")
    ax.legend()

    plt.show()


def ShowSuperPosition(amp, k, deltak):
    x = linspace(-pi/deltak, pi/deltak, QUALITY)
    border = amp *(1 + cos(deltak/2 * x))

    wave1 = PlaneWave(amp, k, 0, x, 0)
    wave2 = PlaneWave(amp/2, k - deltak/2, 0, x, 0)
    wave3 = PlaneWave(amp/2, k + deltak/2, 0, x, 0)
    superposition = wave1 + wave2 + wave3

    fig, ax = plt.subplots()
    ax.plot(x, real(wave1), linewidth=2, label="onde 1")
    ax.plot(x, real(wave2), linewidth=2, label="onde 2")
    ax.plot(x, real(wave3), linewidth=2, label="onde 3")
    ax.plot(x, real(superposition), linewidth=2, label="superposition")
    ax.plot(x, border, linewidth=2, label="enveloppe superieure", linestyle="dashed")
    ax.plot(x, -border, linewidth=2, label="enveloppe inferieure", linestyle="dashed")
    ax.legend()
    plt.show()

ShowPlaneWave(1, 2, 1, 2, 1)
ShowSuperPosition(1, 2*pi, 1)