from QuantumWaveClass import *
from TransmissionRelations import *
from TimeRelations import *

x = np.linspace(-30, 30, 2000) 

V = createBarrier(x, 3, 0, 1)

psi0 = createGaussianPacket(x, -15, 2, 2) 

wave_system = QuantumWave(x, psi0, V)

wave_history = wave_system.evolve(DT, NT) 
print("transmission =", wave_system.getTransmission(1.0))
print("reflection =", wave_system.getReflection(0.0))
print("traversal time =", wave_system.calculateTraversalTime(DT, 0.0, 1.0, -15, 2))
wave_system.showQuantumWave()



nx = 3000
xArray = np.linspace(-120, 120, nx)
k0 = 2.0
energy = k0**2 / (2 * M)
x0 = -45.0
sigma = 8.0
v0 = 2.5
dt = 0.005
nt = 10000

WidthInfluenceTransmission(xArray, v0, x0, k0, sigma, energy, dt, nt)
HeightInfluenceTransmission(xArray, x0, k0, sigma, energy, dt, nt)

xArrayTime = np.linspace(-120, 120, 3000)
x0Time = -45
k0Time = 2
sigmaTime = 8
vg = HBAR * k0Time / M


HartmanEffectTraversalTime(xArrayTime, x0Time, k0Time, sigmaTime, vg)
HeightInfluenceTraversalTime(xArrayTime, x0Time, k0Time, sigmaTime, vg)
