from QuantumWaveClass import *
from TransmissionRelations import *
from TimeRelations import *

X = np.linspace(-30, 30, 2000) 

V = createBarrier(X, 3, 0, 1)

Psi0 = createGaussianPacket(X, -15, 2, 2) 

WaveSystem = QuantumWave(X, Psi0, V)

WaveHistory = WaveSystem.evolve(DT, NT) 
print("transmission =", WaveSystem.getTransmission(1.0))
print("reflection =", WaveSystem.getReflection(0.0))
print("traversal time =", WaveSystem.calculateTraversalTime(DT, 0.0, 1.0, -15, 2))
WaveSystem.showQuantumWave()

showBarycenterEvolution(WaveSystem,0, 1.0, -15.0, 2.0)

showBarycenterZoom(WaveSystem, 0, 1.0, -15, 2.0)

nx = 3000
XArray = np.linspace(-120, 120, nx)
K0 = 2.0
Energy = K0**2 / (2 * M)
X0 = -45.0
Sigma = 8.0
V0 = 2.5
Dt = 0.005
Nt = 10000


showClassicalVsQuantum(
    XArray=XArray, 
    BarrierHeight=V0,
    BarrierWidth=1.0, 
    InitialPosition=-45.0, 
    PacketWidth=8.0, 
    Dt=Dt
)


WidthInfluenceTransmission(XArray, V0, X0, K0, Sigma, Energy, Dt, Nt)
HeightInfluenceTransmission(XArray, X0, K0, Sigma, Energy, Dt, Nt)
GlobalXArray = np.linspace(-120, 120, 3000)


XArrayTime = np.linspace(-120, 120, 3000)
X0Time = -45
K0Time = 2
SigmaTime = 8
Vg = HBAR * K0Time / M


HartmanEffectTraversalTime(XArrayTime, X0Time, K0Time, SigmaTime, Vg)
HeightInfluenceTraversalTime(XArrayTime, X0Time, K0Time, SigmaTime, Vg)
