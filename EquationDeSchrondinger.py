from numpy import pi, exp, sqrt, real, imag, zeros, linspace, cos, empty
from math import pow
import matplotlib.pyplot as plt

NX = 50
NT = 50

X_MIN = -10.0
X_MAX = 10.0
X_ARRAY = linspace(X_MIN, X_MAX, NX)

DT = 0.01
DX = (X_MAX - X_MIN) / NX

def getWaveFunction(x, a, k0):
    return (2 / (pi * a**2))**(1/4) * exp(-(x**2) / a**2) * exp(1j * k0 * x)
    

def initWaveFunction():
    wave = empty((NX,NT), dtype=complex)
    for i in range(NX):
        wave[i,0] = getWaveFunction(X_ARRAY[i], 1, 2)
    return wave

def getWaveFunctionAtTime(wave, i, j, hbar, v0, m):
    spaceConstant = (hbar * DT * 1j) / (2 * m * (DX**2))
    timeConstant = - v0 * DT * 1j/hbar
    spaceDerivative = wave[i+1,j] - 2*wave[i,j] + wave[i-1,j]
    return wave[i,j] + spaceConstant * spaceDerivative + timeConstant * wave[i,j]

def doWave(hbar, v0, m):
    wave = initWaveFunction()
    for j in range(NT-1):
        for i in range(1,NX-1):
            wave[i,j+1] = getWaveFunctionAtTime(wave, i, j, hbar, v0, m)
    return wave

#initial_wavefunction(0)

wave = doWave(1, 0, 1)
print(wave[NX-1,0])