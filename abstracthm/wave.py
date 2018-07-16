import numpy as np
import pickle
from abstracthm.utils import *

## subwave class
class Subwave:
    """Container for emulator(s) and corresponding obervation(s): each subwave will generate an implausibility score"""
    def __init__(self, emul, z, v, name="subwave"):
        self.emul, self.z, self.v, self.name = emul, z, v, name
        self.pmean, self.pvar = None, None
        self.simImp()

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

    def simImp(self):
        X, Y = self.emul.data()
        self.Isim = np.sqrt( ( Y[:,0] - self.z )**2 / ( self.v ) )
        

# multivar class for multivariate implausibility from univariate emulators in subwave
class Multivar:
    """Container for subwaves that will return a multivariate implausibility"""
    def __init__(self, subwaves, covar):
        self.subwaves, self.covar = subwaves, covar
        self.simImp()

    def calcImp(self, TESTS):
        P = TESTS.shape[0]

        pmean = np.hstack((s.pmean[:,None] for s in self.subwaves))
        pvar  = np.hstack((s.pvar[:,None]  for s in self.subwaves))
        zs = np.array([s.z for s in self.subwaves])

        print("= Calculating multivariate implausibility of", P, "points =")
        mI = np.zeros(P, dtype=np.float16)
        for p in range(P):
            diff = zs - pmean[p]
            var  = self.covar + np.diag(pvar[p])
            mI[p] = np.sqrt( (diff.T).dot(np.linalg.solve(var,diff)) )
        
        return mI

    def simImp(self):

        X, Y = self.subwaves[0].emul.data()
        mI = np.zeros(X.shape[0])
        zs = np.array([s.z for s in self.subwaves])
        ys = np.hstack([s.emul.data()[1] for s in self.subwaves])

        ## calculate multivariate implausibility
        print("= Calculating multivariate implausibility of simulation points =")
        P = X.shape[0]
        mIsim = np.zeros(P)
        for p in range(P):
            diff = zs - ys[p]
            var  = self.covar
            mIsim[p] = np.sqrt( (diff.T).dot(np.linalg.solve(var,diff)) )
         
        self.Isim = mIsim


## wave class
class Wave:
    """Stores data for wave of HM search."""
    def __init__(self, subwaves, cm, multivar=None, cmv=None, tests=[]):
        ## passed in
        print("█ NOTE: Data points should be in same order (not shuffled) for all emulators")
        self.subwaves = subwaves
        self.multivar = multivar

        self.cm, self.cmv = cm, cmv
        self.I, self.mI = None, None

        if isinstance(tests, np.ndarray) and tests is not []:
            self.TESTS = tests.astype(np.float16)
            self.I  = np.zeros((self.TESTS.shape[0],len(self.subwaves)), dtype=np.float16)
        else: print("ERROR: tests must be a numpy array")

        self.NIMP = None  # for storing indices of TESTS with Im < cm 
        self.NROY = None  # create a design to fill NROY space based of found NIMP points
        self.NROY_I = None # for storing NROY implausibility


    ## pickle a list of relevant data
    def save(self, filename):
        print("= Pickling wave data in", filename, "=")
        w = [ self.TESTS, self.I, self.NIMP, self.NROY, self.NROY_I, self.mI ]
        with open(filename, 'wb') as output:
            pickle.dump(w, output, pickle.HIGHEST_PROTOCOL)
        return

    ## unpickle a list of relevant data
    def load(self, filename):
        print("= Unpickling wave data in", filename, "=")
        with open(filename, 'rb') as input:
            w = pickle.load(input)
        self.TESTS, self.I, self.NIMP, self.NROY, self.NROY_I, self.mI = [i for i in w]
        return

    ## search through the test inputs to find non-implausible points
    def calcImp(self, chunkSize=5000):
        P = self.TESTS.shape[0]
        chunkNum = int(np.ceil(P / chunkSize)) if P > chunkSize else 1
        print("= Calculating Implausibilities of", P, "points =")
        print("  Using", chunkNum, "chunks of", chunkSize, "=")

        for o, s in enumerate(self.subwaves):
            self.I[:,o] = s.calcImp(self.TESTS, chunkSize = chunkSize)

        # calculate multivariate implausibility of collection of univariate emulators
        if self.multivar is not None:
            self.mI = self.multivar.calcImp(self.TESTS)

        self.doneImp = True
        return

    ## search through the test inputs to find non-implausible points
    def simImp(self, maxno=1):
        print("  Calculating Implausibilities of simulation points")

        # FIXME: assumes all emuls built with same inputs in same order data
        X = self.subwaves[0].emul.data()[0]

        Isim = np.zeros([X.shape[0], len(self.subwaves)])
        for o, s in enumerate(self.subwaves):
            Isim[:,o] = self.subwaves[0].Isim

        # FIXME: multivar implementation makes some assumption that all emuls build with same inputs in same order data
        if self.multivar is not None:
            mIsim = self.multivar.Isim
 
        return X, Isim, mIsim

    def findNIMPsim(self, maxno=1):
        print("  Returning non-implausible simulation points")
        X, Isim, mIsim = self.simImp()
        Y = self.emuls[0].Data.yAll
        Imaxes = np.partition(Isim, -maxno)[:,-maxno]
        if self.covar is None:
            NIMP = np.argwhere(Imaxes < self.cm)[:,0]
        else:
            NIMP = np.argwhere((Imaxes < self.cm) & (mIsim < self.cmv))[:,0]
        xx, yy = X[NIMP], Y[NIMP]
        return self.unscale(xx, prnt=False), yy

    ## find all the non-implausible points in the test points
    def findNIMP(self, maxno=1):

        if self.I is None:
            print("ERROR: implausibilities must first be calculated with calcImp()")
            return

        P = self.TESTS.shape[0]
        ## find maximum implausibility across different outputs
        print("  Determining", maxno, "max'th implausibility...")
        Imaxes = np.partition(self.I, -maxno)[:,-maxno]

        ## check cut-off, store indices of points matching condition
        if self.multivar is None:
            self.NIMP = np.argwhere(Imaxes < self.cm)[:,0]
        else:
            self.NIMP = np.argwhere((Imaxes < self.cm) & (self.mI < self.cmv))[:,0]

        percent = ("{0:3.2f}").format(100*float(len(self.NIMP))/float(P))
        print("  NIMP fraction:", percent, "%  (", len(self.NIMP), "points )" )

        if self.NIMP.shape[0] == 0: self.NIMP = None


    ## fill out NROY space to use as tests for next wave
    def findNROY(self, howMany, maxno=1, factor = 0.1, chunkSize=5000, restart=False):

        if self.NIMP is None:
            print("  Cannot calculate NROY from zero non-implauisible test points")
            return

        ## reset if requested
        if restart == True:
            print("= Setting NROY blank, start from NIMP points")
            self.NROY, self.NROY_I = None, None
 
        if self.NROY is None:
            # initially, add NIMP to NROY
            self.NROY, self.NROY_I = self.TESTS[self.NIMP], self.I[self.NIMP]
            print("= Creating", howMany, "NROY cloud from", self.NIMP.size , "NIMP points =")
        else:
            # if we're recalling this routine, begin with known NROY point
            print("= Creating", howMany, "NROY cloud from", self.NROY.shape[0], "NROY points =")

        ## store minmax of NIMP points along each dimension
        NROYminmax = {}
        for i in range(self.TESTS.shape[1]):
            NROYminmax[i] = [np.amin(self.TESTS[self.NIMP,i]), np.amax(self.TESTS[self.NIMP,i])]

        ## exit if condition already satisfied
        if howMany <= self.NROY.shape[0]:
            print("  Already have", self.NROY.shape[0], "/", howMany, "requested NROY points")
            return

        ## OUTER LOOP - HOW MANY POINTS NEEDED IN TOTAL
        howMany = int(howMany)
        printProgBar(self.NROY.shape[0], howMany, prefix = '  NROY Progress:', suffix = '\n')
        while self.NROY.shape[0] < howMany:
        
            # now LOC and dic can just use NROY in all cases
            LOC, dic = self.NROY, NROYminmax

            ## scale factor for normal distribution
            SCALE = np.array( [dic[mm][1]-dic[mm][0] for mm in dic] )
            SCALE = SCALE * factor

            ## we won't accept value beyond the emulator ranges
            print("  Generating (scaled) normal samples within original search range...")
            #minmax = self.emuls[0].minmaxScaled()
            minmax = self.emuls[0].minmax() # FIXME: I've replaced this by minmax for now
            minlist = [minmax[key][0] for key in minmax]
            maxlist = [minmax[key][1] for key in minmax]

            ## initial empty structure to append 'candidate' points to
            NROY = np.zeros([0,self.TESTS.shape[1]])

            ## create random points - known NROY used as seeds
            temp = np.random.normal(loc=LOC, scale=SCALE)

            ## we only regenerate points that failed to be within bounds
            ## this means that every seed points gets a new point
            A, B = temp, LOC
            NT = np.zeros([0,self.TESTS.shape[1]])
            repeat = True
            while repeat:
                minFilter = A < minlist
                maxFilter = A > maxlist
                for i in range(A.shape[0]):
                    A[i,minFilter[i]] = \
                      np.random.normal(loc=B[i,minFilter[i]], scale=SCALE[minFilter[i]])
                    A[i,maxFilter[i]] = \
                      np.random.normal(loc=B[i,maxFilter[i]], scale=SCALE[maxFilter[i]])
                minFilter = np.prod( (A > minlist) , axis=1 )
                maxFilter = np.prod( (A < maxlist) , axis=1 )
                NT = np.concatenate( (NT, A[minFilter*maxFilter == 1]), axis = 0)
                if NT.shape[0] >= LOC.shape[0]: repeat = False
                A = (A[minFilter*maxFilter == 0])
                B = (B[minFilter*maxFilter == 0])

            ## add viable test points to NROY (tested for imp below)
            NROY = np.concatenate((NROY, NT), axis=0)

            ## hack part 1 - save the results of initial test points
            TEMP = [self.TESTS, self.I, self.NIMP, self.NIMPminmax]

            self.setTests(NROY)
            self.calcImp(chunkSize=chunkSize)
            self.findNIMP(maxno=maxno) # use to get indices of NROY that are imp < cutoff

            self.NROY = np.concatenate( (self.TESTS[self.NIMP], LOC), axis=0 ) # LOC = seeds
            printProgBar(self.NROY.shape[0], howMany, prefix = '  NROY Progress:', suffix = '\n')
            print("  NROY has", self.NROY.shape[0], "points, including original",
                  LOC.shape[0], "seed points")
            # NOTE: not storing multivariate implausibility - NROY_I only used for plotting
            if len(self.NROY_I) > 0:
                self.NROY_I = np.concatenate( (self.I[self.NIMP], self.NROY_I), axis=0 )
            else:
                self.NROY_I = np.concatenate( (self.I[self.NIMP], TEMP[3][TEMP[4]]), axis=0 )

            ## store minmax of NROY points along each dimension
            for i in range(self.NROY.shape[1]):
                NROYminmax[i] = [np.amin(self.NROY[:,i]), np.amax(self.NROY[:,i])]

            ## hack part 2 - reset these variables back to normal
            [self.TESTS, self.I, self.NIMP, self.NIMPminmax] = TEMP

