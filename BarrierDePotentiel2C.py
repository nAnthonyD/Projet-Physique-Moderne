from numpy import pi, exp, zeros, linspace, empty
import matplotlib.pyplot as plt

NX = 100  # Augmenté un peu pour une meilleure résolution de la barrière
NT = 2000 # Augmenté pour laisser le temps à l'onde de frapper la barrière

X_MIN = -10.0
X_MAX = 10.0
X_ARRAY = linspace(X_MIN, X_MAX, NX)

DX = (X_MAX - X_MIN) / NX
DT = 0.0005 # Stabilité numérique requise

def getWaveFunction(x, a, k0):
    return (2 / (pi * a**2))**(1/4) * exp(-(x**2) / a**2) * exp(1j * k0 * x)

def createBarrier(min_x, max_x, value):
    barrier = zeros(NX)
    for i in range(NX):
        if X_ARRAY[i] >= min_x and X_ARRAY[i] <= max_x:
            barrier[i] = value
    return barrier

def initWaveFunction():
    wave = zeros((NX, NT), dtype=complex)
    for i in range(NX):
        wave[i, 0] = getWaveFunction(X_ARRAY[i], 1, 2)
    return wave

def getAveragePosition(wave, j):
    averagePosition = 0 
    for i in range(NX):
        averagePosition += X_ARRAY[i] * abs(wave[i, j])**2 * DX
    return averagePosition

def getTimeToReach(wave, threshold):
    for j in range(NT):
        if getAveragePosition(wave, j) >= threshold:
            return j * DT
    return None

def getTimeToPassBarrier(wave, target_x, barrier_max_x):
    for j in range(NT):
        
        peak_prob = 0.0
        peak_x = X_MIN
        
        for i in range(NX):
            if X_ARRAY[i] > barrier_max_x:
                prob_actuelle = abs(wave[i, j])**2
                
                # On cherche le sommet du paquet transmis
                if prob_actuelle > peak_prob:
                    peak_prob = prob_actuelle
                    peak_x = X_ARRAY[i]
        
        if peak_x >= target_x and peak_prob > 1e-10:
            return j * DT
            
    return None

def getWaveFunctionAtTime(wave, i, j, hbar, m, barrier):
    spaceConstant = (hbar * DT * 1j) / (2 * m * (DX**2))
    timeConstant = - barrier[i] * DT * 1j / hbar
    spaceDerivative = wave[i+1, j] - 2 * wave[i, j] + wave[i-1, j]
    return wave[i, j] + spaceConstant * spaceDerivative + timeConstant * wave[i, j]

def doWave(hbar, m, barrier):
    wave = initWaveFunction()
    for j in range(NT-1):
        for i in range(1, NX-1):
            wave[i, j+1] = getWaveFunctionAtTime(wave, i, j, hbar, m, barrier)
    return wave

def getTotalProbability(wave, j):
    totalProb = 0.0
    for i in range(NX):
        totalProb += abs(wave[i, j])**2 * DX
    return totalProb

def doAll(barrier, hbar, m):
    wave = doWave(hbar, m, barrier)
    plt.figure(figsize=(10, 5))

    plt.plot(X_ARRAY, barrier * 0.1, color='red', label="Barrière de potentiel V_0 (mise à l'échelle)")

    plt.plot(X_ARRAY, abs(wave[:, 0])**2, linestyle='--', color='gray', label="Probabilité initiale (t=0)")
    plt.plot(X_ARRAY, abs(wave[:, -1])**2, color='blue', label="Probabilité finale")

    plt.xlabel("Position (x)")
    plt.ylabel("Densité de probabilité |Psi|^2")
    plt.legend()
    plt.grid(True)
    plt.show()


barrier = createBarrier(2.0, 3.0, 5.0)
doAll(barrier, 1.0, 1.0)