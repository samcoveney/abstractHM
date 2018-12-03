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

def sim2(x):
    y = 5.0*x[:,0]**2 + np.exp(np.sin(3.0*x[:,1]))
    return y


inputs  = np.random.randn(300).reshape([150,2])

# building GPy model 1
# --------------------
outputs = sim(inputs)
kernel = g.kern.Matern52(input_dim=2, active_dims=[0,1])
model1 = g.models.GPRegression(X = inputs, Y = outputs[:, None], kernel = kernel)
model1.optimize_restarts(3)
print(model1)
#model.plot()
#plt.show()


# building GPy model 2
# --------------------
outputs2 = sim2(inputs)
kernel2 = g.kern.Matern52(input_dim=2, active_dims=[0,1])
model2 = g.models.GPRegression(X = inputs, Y = outputs2[:, None], kernel = kernel2)
model2.optimize_restarts(3)
print(model2)


np.savetxt("inputs.txt", inputs)
np.savetxt("outputs.txt", np.hstack([outputs.reshape([-1,1]),outputs2.reshape([-1,1])]))


# pass the model to the class wrapper for GPy
# -------------------------------------------
E1 = ahm.GPyEmulator(model1)
E2 = ahm.GPyEmulator(model2)

testX = np.array([[0.5, 0.25]])
#print(E1.posterior(testX))
#print(E2.posterior(testX))


# history matching
# ----------------

zs = [2.0]
vs = [0.01]
cutoff = 3.0
maxno = 1
waveno = 1
nametag = "test"
LOAD = False


# subwave
subwave1 = ahm.Subwave(E1, zs[0], vs[0], name="sim1(x)")
print(subwave1.name)
subwave2 = ahm.Subwave(E2, zs[0], vs[0], name="sim2(x)")
print(subwave2.name)
MW = ahm.Multivar(subwaves=[subwave1, subwave2], covar=np.array([[vs[0],0],[0,vs[0]]]))


# some test points
NUM = 50000
tests = np.random.randn(NUM*2).reshape([NUM,2])

W = ahm.Wave(subwaves = [subwave1, subwave2], cm=cutoff, multivar=MW, tests=tests)
#W = ahm.Wave(subwaves = [subwave1, subwave2], cm=cutoff, tests=tests)

if not(LOAD):
    W.findNROY()
    W.save("wave"+str(waveno)+nametag+".pkl")
else: 
    try:
        W.load("wave"+str(waveno)+nametag+".pkl")
    except FileNotFoundError as e:
        pass; exit()


#factor = 0.10
#print("█ NB: factor for NROY cloud set to:", factor, "(adjust to keep ~20%)")
#if not(LOAD):
#    W.findNROY(chunkSize=5000)
#    W.save("wave"+str(waveno)+nametag+".pkl")
#else:
#    W.load("wave"+str(waveno)+nametag+".pkl")

#model.plot()
#plt.scatter( W.NROY[:,0], W.NROY[:,1] )
#plt.show()

MINMAXwave1 = E1.minmax()

#ahm.plotImp(W, grid=30, maxno=maxno, NROY=False, NIMP=True, manualRange=MINMAXwave1, vmin=1.0, sims=False, odp=True)
ahm.plotImp(W, grid=30, maxno=maxno, NROY=True, NIMP=False, manualRange=MINMAXwave1, vmin=1.0, sims=False, odp=True)

