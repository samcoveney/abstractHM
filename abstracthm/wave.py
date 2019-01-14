import numpy as np
import pickle
from abstracthm.utils import *
from scipy.stats import chi2

## subwave class
class Subwave:
    """Container for emulator(s) and corresponding obervation(s): each subwave will generate an implausibility score"""
    def __init__(self, emul, z, v, name="subwave"):
        self.emul, self.z, self.v, self.name = emul, z, v, name
        self.pmean, self.pvar = None, None

    def calcImp(self, TESTS, chunkSize=5000):
        P = TESTS.shape[0]
        self.pmean, self.pvar = np.zeros([P]), np.zeros([P])

        chunkNum = int(np.ceil(P / chunkSize)) if P > chunkSize else 1
        prefix = '  ' + self.name + ' progress:'
        printProgBar(0, chunkNum, prefix = prefix, suffix = '')
        for c in range(chunkNum):
            L, U = c*chunkSize, (c+1)*chunkSize if c < chunkNum -1 else P
            self.pmean[L:U], self.pvar[L:U] = self.emul.posterior(TESTS[L:U])
            printProgBar((c+1), chunkNum, prefix = prefix, suffix = '')

        I = np.sqrt( ( self.pmean - self.z )**2 / ( self.pvar + self.v ) )
        return I


## multivar class for multivariate implausibility from univariate emulators in subwave
class Multivar:
    """Container for subwaves that will return a multivariate implausibility"""
    def __init__(self, subwaves, covar):
        self.subwaves, self.covar = subwaves, covar

    def calcImp(self, TESTS):

        pmean = np.hstack((s.pmean[:,None] for s in self.subwaves))
        pvar  = np.hstack((s.pvar[:,None]  for s in self.subwaves))
        zs = np.array([s.z for s in self.subwaves])

        P = TESTS.shape[0]
        print("  Calculating multivariate implausibility...")

        # broadcast the solve and the dot product
        b = pmean - zs
        A = np.tile(self.covar[None,:],(P,1,1))
        for p in range(P):  A[p] += np.diag(pvar[p]) 
        mI = np.sqrt( np.einsum( 'ij,ij->i', b, np.linalg.solve(A,b) ) )
        
        return mI


## wave class
class Wave:
    """Stores data for wave of HM search."""
    def __init__(self, subwaves, cm, maxno=1, multivar=None, cmv=None, tests=None):

        self.subwaves, self.multivar = subwaves, multivar
        self.cm, self.maxno = cm, maxno 
        # appropriate cutoff for multivar emulator
        if self.multivar is not None:
            self.cmv = chi2.isf(q=0.01, df=len(self.multivar.subwaves)) if cmv is None else cmv
        self.TESTS, self.I, self.mI, self.NROY, self.NROY_I = None, None, None, None, None

        if isinstance(tests, np.ndarray) and tests is not None:
            #self.TESTS = tests.astype(np.float16)
            self.TESTS = tests
        else: print("ERROR: if given, tests must be a numpy array")


    ## pickle a list of relevant data
    def save(self, filename):
        print("= Pickling wave data in", filename, "=")
        w = [ self.TESTS, self.I, self.NROY, self.NROY_I, self.mI ]
        with open(filename, 'wb') as output:
            pickle.dump(w, output, pickle.HIGHEST_PROTOCOL)
        return

    ## unpickle a list of relevant data
    def load(self, filename):
        print("= Unpickling wave data in", filename, "=")
        with open(filename, 'rb') as input:
            w = pickle.load(input)
        self.TESTS, self.I, self.NROY, self.NROY_I, self.mI = [i for i in w]
        return

    ## search through the test inputs to find non-implausible points
    def calcImp(self, TESTS, chunkSize=5000):
        P = TESTS.shape[0]
        chunkNum = int(np.ceil(P / chunkSize)) if P > chunkSize else 1
        print("= Calculating Implausibilities of", P, "points =")
        print("  Using", chunkNum, "chunks of", chunkSize)

        I  = np.zeros((TESTS.shape[0],len(self.subwaves)), dtype=np.float16)
        for o, s in enumerate(self.subwaves):
            I[:,o] = s.calcImp(TESTS, chunkSize = chunkSize)

        # calculate multivariate implausibility of collection of univariate emulators
        mI = self.multivar.calcImp(TESTS) if self.multivar is not None else None
            

        return I, mI

    ## returns NIMP and IMP by maximizing across implausibilites
    def findNIMP(self, I, mI):
        
        ## find maximum implausibility across different outputs
        print("  Determining", self.maxno, "max'th implausibility...")
        Imaxes = np.partition(I, -self.maxno)[:,-self.maxno]

        ## check cut-off, store indices of points matching condition
        tA = (Imaxes < self.cm) if self.multivar is None else ((Imaxes < self.cm) & (mI < self.cmv))

        NIMP = np.argwhere(tA)[:,0]
        IMP = np.argwhere(np.invert(tA))[:,0]

        return NIMP, IMP, Imaxes

    ## find all the non-implausible points in the test points
    def findNROY(self, chunkSize=5000):

        # if called twice, would lose NROY! -> recombine TESTS and NROY if called again
        if self.I is not None:  self.TESTS = np.vstack((self.TESTS, self.NROY))

        # set this waves Implausibilities for self.TESTS using calcImp
        self.I, self.mI = self.calcImp(self.TESTS, chunkSize=chunkSize)

        NIMP, IMP, Imaxes = self.findNIMP(self.I, self.mI)

        percent = ("{0:3.2f}").format(100*float(len(NIMP))/float(self.TESTS.shape[0]))
        print("  NIMP fraction:", percent, "%  (", len(NIMP), "points )" )
 
        # add non-implausible points to NROY
        self.NROY, self.NROY_I = self.TESTS[NIMP], self.I[NIMP]
        # remove non-implausible points from TESTS to save on storage
        self.TESTS, self.I = self.TESTS[IMP], self.I[IMP]
        print("\n!!! WARNING !!! : NROY: only non-imp points, TESTS: now only imp points\n")

        return

    ## search through the test inputs to find non-implausible points
    def simImp(self):
        print("  Calculating Implausibilities of simulation points of first subwave")

        X, Y = self.subwaves[0].emul.data()
        Isim, mIsim = self.calcImp(X)
        NIMP, IMP, Imaxes = self.findNIMP(Isim, mIsim)

        return X, Y, Imaxes, X[NIMP], Y[NIMP]

