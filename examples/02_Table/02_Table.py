# 02_Table
import numpy as np
import abstracthm as ahm

inputs = np.loadtxt("inputs.txt")
outputs = np.loadtxt("outputs.txt")


# I have to make some sort of Table class here... has methods and members which match those in TableEmulator
# needs members: X, Y
class simpleTable():
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y



# pass the model to the class wrapper for Table emulators
# -------------------------------------------------------
model1= simpleTable(inputs, outputs[:,1]) # column 1 is CV for s1-s2 = 336 ms
E1 = ahm.TableEmulator(model1)
E1.posterior(inputs[10:100,])
print("E1 data:\n", E1.data())

model2 = simpleTable(inputs, outputs[:,-1]) # the last column is ERP
E2 = ahm.TableEmulator(model2)
E2.posterior(inputs[10:100,])
print("E2 data:\n", E2.data())

# history matching
# ----------------
zs = [50.0, 200.0]
vs = [10.0, 25.0]
cutoff = 3.0
maxno = 1
waveno = 1
MINMAXwave1 = E1.minmax()
nametag = "test"
names = ["D", "T_in", "T_out", "T_open", "T_close"]

# setup the subwaves
# ------------------
# subwave
subwave1 = ahm.Subwave(E1, zs[0], vs[0], name="CV")
subwave2 = ahm.Subwave(E2, zs[1], vs[1], name="ERP")

# some test points
tests = inputs # NOTE: the test inputs are EXACTLY the inputs in the table

W = ahm.Wave(subwaves = [subwave1, subwave2], cm=cutoff, tests=tests)   # with ERP
#W = ahm.Wave(subwaves = [subwave1], cm=cutoff, tests=tests)            # without ERP
W.findNROY()

ahm.plotImp(W, grid=10, maxno=maxno, NROY=True, NIMP=True, manualRange=MINMAXwave1, vmin=1.0, sims=False, odp=True, colorbar=False, names = names)

