# 01_GPy_Example

import numpy as np
import GPy as g
import matplotlib.pyplot as plt
import abstracthm as ahm

# make some simulation data
# -------------------------
def sim(x):
    y = 3.0*x[:,0]**3 + np.exp(np.cos(5.0*x[:,1]))
    return y

inputs  = np.random.randn(300).reshape([150,2])
outputs = sim(inputs)
print(inputs, outputs)

# building GPy model
# ------------------
kernel = g.kern.Matern52(input_dim=2, active_dims=[0,1])
model = g.models.GPRegression(X = inputs, Y = outputs[:, None], kernel = kernel)
model.optimize_restarts(3)
print(model)
#model.plot()
#plt.show()

# pass the model to the class wrapper for GPy
# -------------------------------------------
E = ahm.GPyEmulator(model)
testX = np.array([[0.5, 0.25]])
print(E.posterior(testX))
x, y = E.data()
print(x, y)
print(E.active())


# history matching
# ----------------

zs = [2.0]
vs = [0.01]
cutoff = 3.0
maxno = 1
waveno = 1
nametag = "test"
LOAD = False

# list of waves
emuls = [ ]
emuls.append(E)

# some test points
NUM = 50000
tests = np.random.randn(NUM*2).reshape([NUM,2])

W = ahm.Wave(emuls=emuls, zs=zs, cm=cutoff, var=vs, tests=tests)

if not(LOAD):
    W.calcImp(chunkSize=5000)
    W.findNIMP(maxno=maxno)
    W.save("wave"+str(waveno)+nametag+".pkl")
else: 
    try:
        W.load("wave"+str(waveno)+nametag+".pkl")
    except FileNotFoundError as e:
        pass; exit()

factor = 0.10
print("█ NB: factor for NROY cloud set to:", factor, "(adjust to keep ~20%)")
if not(LOAD):
    W.findNROY(howMany = 100, maxno=maxno, factor=factor, chunkSize=5000)
    W.save("wave"+str(waveno)+nametag+".pkl")
else:
    W.load("wave"+str(waveno)+nametag+".pkl")

model.plot()
plt.scatter( W.NROY[:,0], W.NROY[:,1] )
plt.show()

MINMAXwave1 = emuls[0].minmax()

ahm.plotImp(W, grid=30, maxno=maxno, NROY=False, NIMP=True, manualRange=MINMAXwave1, vmin=1.0, sims=False, odp=True)
#ahm.plotImp(W, grid=30, maxno=maxno, NROY=True, NIMP=False, manualRange=MINMAXwave1, vmin=1.0, sims=True, odp=True)

