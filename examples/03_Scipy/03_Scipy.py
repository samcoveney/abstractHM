# 03_Scipy
# :: wrap the emulators up and pass them to the abstract base class

# reload those other modules...
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
np.set_printoptions(precision=10)

# load the data
# -------------
import pickle
import abstracthm as ahm
import diversipy as dp

# (to unpickle, I need to define this class again...)
class model():
    def __init__(self, mean, gp, X, Y):
        self.mean = mean
        self.gp = gp
        self.X = X
        self.Y = Y

with open("testModel0.pkl", "rb") as pickle_file:
    testModel0 = pickle.load(pickle_file)

with open("testModel1.pkl", "rb") as pickle_file:
    testModel1 = pickle.load(pickle_file)

# pass to the new abstract base class
# -----------------------------------

AHM0 = ahm.sclEmulator(testModel0)
AHM1 = ahm.sclEmulator(testModel1)

Xdata = np.loadtxt('inputs_7_693.txt', dtype=float)
Ydata = np.loadtxt('outputs_11_693.txt', dtype=float)


# okay, let's make some wave objects
# ----------------------------------

E1 = AHM0
E2 = AHM1

trueX = Xdata[0].reshape([1,-1])
zs = Ydata[0,[0,1]]
vs = 0.01*np.array([ np.ptp(Ydata[:,0]) , np.ptp(Ydata[:,1]) ])
cutoff = 3.0
maxno = 1
waveno = 1
nametag = "test"
LOAD = False


# subwave
subwave1 = ahm.Subwave(E1, zs[0], vs[0], name="y[0]")
print(subwave1.name)
subwave2 = ahm.Subwave(E2, zs[1], vs[1], name="y[1]")
print(subwave2.name)
#MW = ahm.Multivar(subwaves=[subwave1, subwave2], covar=np.array([[vs[0],0],[0,vs[0]]]))


# some test points
NUM = 100000 # 1 hundred thousand
minmax = AHM0.minmax()
tests = dp.centered_lhs(dp.lhd_matrix(NUM,len(minmax)))  # create tests (unit hypercube)
tests = tests.T
for dim, col in enumerate(tests):
    tests[dim] = tests[dim]*( minmax[dim][1] - minmax[dim][0] ) + minmax[dim][0]
tests = tests.T

# create wave object
W = ahm.Wave(subwaves = [subwave1, subwave2], cm=cutoff, tests=tests)

# do history matching
if not(LOAD):
    W.findNROY()
    W.save("wave"+str(waveno)+nametag+".pkl")
else: 
    try:
        W.load("wave"+str(waveno)+nametag+".pkl")
    except FileNotFoundError as e:
        pass; exit()


# plotting
MINMAXwave1 = E1.minmax()
#ahm.plotImp(W, grid=12, maxno=maxno, NROY=True, NIMP=False, manualRange=MINMAXwave1, vmin=1.0, sims=False, odp=True, points = [trueX])
ahm.plotImp(W, grid=12, maxno=maxno, NROY=True, NIMP=False, manualRange=MINMAXwave1, vmin=1.0, sims=True, odp=False, points = [trueX])


