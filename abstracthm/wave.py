import numpy as np
import pickle
from abstracthm.utils import *

## wave class
class Wave:
    """Stores data for wave of HM search."""
    def __init__(self, emuls, zs, cm, var, cmv=None, covar=None, tests=[]):
        ## passed in
        print("█ NOTE: Data points should be in same order (not shuffled) for all emulators")
        self.emuls = emuls
        self.zs, self.var, self.cm = zs, var, cm
        self.covar = covar
        self.cmv = cmv
        if self.cm < 2.0:
            print("ERROR: cutoff cannot be less than 2.0 (should start/stay at 3.0)"); exit()
        if self.cmv is not None:
            print("█ NOTE: multivariate imp cutoff (cmv) should be selected from appropiate percentile of chi-sq table with", len(zs), "degrees of freedom")
        elif self.covar is not None:
            print("ERROR: multivariate imp cutoff (cmv) must be provided if covar is provided"); exit()
        self.I = []
        self.doneImp = False
        if tests is not []:
            self.setTests(tests)

        self.NIMP = []  # for storing indices of TESTS with Im < cm 
        self.NIMPminmax = {}
        self.NROY = []  # create a design to fill NROY space based of found NIMP points
        self.NROYminmax = {}
        self.NROY_I = [] # for storing NROY implausibility


    ## pickle a list of relevant data
    def save(self, filename):
        print("= Pickling wave data in", filename, "=")
        w = [ self.TESTS, self.I, self.NIMP, self.NIMPminmax, self.doneImp, self.NROY, self.NROYminmax, self.NROY_I, self.mI ]
        with open(filename, 'wb') as output:
            pickle.dump(w, output, pickle.HIGHEST_PROTOCOL)
        return

    ## unpickle a list of relevant data
    def load(self, filename):
        print("= Unpickling wave data in", filename, "=")
        with open(filename, 'rb') as input:
            w = pickle.load(input)
        self.TESTS, self.I, self.NIMP, self.NIMPminmax, self.doneImp, self.NROY, self.NROYminmax, self.NROY_I, self.mI = [i for i in w]
        return

    ## set the test data
    def setTests(self, tests):
        if isinstance(tests, np.ndarray):
            self.TESTS = tests.astype(np.float16)
            self.I  = np.zeros((self.TESTS.shape[0],len(self.emuls)) )#,dtype=np.float16)
            self.mI = np.zeros((self.TESTS.shape[0]) ) if self.covar is not None else None
        else:
            print("ERROR: tests must be a numpy array")
        return


    ## search through the test inputs to find non-implausible points
    def calcImp(self, chunkSize=5000):
        P = self.TESTS[:,0].size
        print("= Calculating Implausibilities of", P, "points =")
        if P > chunkSize:
            chunkNum = int(np.ceil(P / chunkSize))
            print("  Using", chunkNum, "chunks of", chunkSize, "=")
        else:
            chunkNum = 1

        pmean = np.zeros((self.TESTS.shape[0],len(self.emuls)) )#,dtype=np.float16)
        pvar  = np.zeros((self.TESTS.shape[0],len(self.emuls)) )#,dtype=np.float16)

        ## loop over outputs (i.e. over emulators)
        printProgBar(0, len(self.emuls)*chunkNum, prefix = '  Progress:', suffix = '')
        for o in range(len(self.emuls)):
            E, z, v = self.emuls[o], self.zs[o], self.var[o]

            for c in range(chunkNum):
                L = c*chunkSize
                U = (c+1)*chunkSize if c < chunkNum -1 else P
                pm, pv = E.posterior(self.TESTS[L:U])
                pmean[L:U,o] = pm
                pvar[L:U,o]  = pv
                #self.I[L:U,o] = np.sqrt( ( pmean - z )**2 / ( pvar + v ) )
                self.I[L:U,o] = np.sqrt( ( pm - z )**2 / ( pv + v ) )
                printProgBar((o*chunkNum+c+1), len(self.emuls)*chunkNum,
                              prefix = '  Progress:', suffix = '')

        ## calculate multivariate implausibility
        if self.covar is not None:
            print("= Calculating multivariate implausibility of", P, "points =")
            #self.mI = np.zeros(P)
            for p in range(P):
                diff = self.zs - pmean[p]
                var  = self.covar + np.diag(pvar[p])
                self.mI[p] = np.sqrt( (diff.T).dot(np.linalg.solve(var,diff)) )

        self.doneImp = True
        return

    ## search through the test inputs to find non-implausible points
    def simImp(self):
        print("  Calculating Implausibilities of simulation points")

        X, Y = self.emuls[0].data() # FIXME: assumes all emuls built with same data
        Isim = np.zeros([X.shape[0], len(self.emuls)])

        ## loop over outputs (i.e. over emulators)
        for o in range(len(self.emuls)):
            E, z, v = self.emuls[o], self.zs[o], self.var[o]
            X, Y = E.data()
            pmean = Y[0]
            Isim[:,o] = np.sqrt( ( pmean - z )**2 / ( v ) )
        
        ## calculate multivariate implausibility
        mIsim = []
        if self.covar is not None:
            print("= Calculating multivariate implausibility of simulation points =")
            P = X.shape[0]
            mIsim = np.zeros(P)
            for p in range(P):
                diff = self.zs - np.array([E.data()[1][p] for E in self.emuls])
                var  = self.covar
                mIsim[p] = np.sqrt( (diff.T).dot(np.linalg.solve(var,diff)) )
 
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

        if self.doneImp == False:
            print("ERROR: implausibilities must first be calculated with calcImp()")
            return

        P = self.TESTS[:,0].size
        ## find maximum implausibility across different outputs
        print("  Determining", maxno, "max'th implausibility...")
        Imaxes = np.partition(self.I, -maxno)[:,-maxno]

        ## check cut-off, store indices of points matching condition
        if self.covar is None:
            self.NIMP = np.argwhere(Imaxes < self.cm)[:,0]
        else:
            self.NIMP = np.argwhere((Imaxes < self.cm) & (self.mI < self.cmv))[:,0]

        percent = ("{0:3.2f}").format(100*float(len(self.NIMP))/float(P))
        print("  NIMP fraction:", percent, "%  (", len(self.NIMP), "points )" )

        ## store minmax of NIMP points along each dimension
        if self.NIMP.shape[0] > 0:
            for i in range(self.TESTS.shape[1]):
                NIMPmin = np.amin(self.TESTS[self.NIMP,i])
                NIMPmax = np.amax(self.TESTS[self.NIMP,i])
                self.NIMPminmax[i] = [NIMPmin, NIMPmax]
        else:
            print("  No points in NIMP, set NIMPminmax to [None, None]")
            for i in range(self.TESTS.shape[1]):
                self.NIMPminmax[i] = [None, None]
        #print("  NIMPminmax:", self.NIMPminmax)

        return 100*float(len(self.NIMP))/float(P)

    ## fill out NROY space to use as tests for next wave
    def findNROY(self, howMany, maxno=1, factor = 0.1, chunkSize=5000, restart=False):

        ## reset if requested
        if restart == True:
            print("= Setting NROY blank, start from NIMP points")
            self.NROY, self.NROY_I = [], []

        if len(self.NROY) == 0:
            # initially, add NIMP to NROY
            self.NROY = self.TESTS[self.NIMP]
            self.NROY_I = self.I[self.NIMP]
            self.NROYminmax = self.NIMPminmax
            print("= Creating", howMany, "NROY cloud from", self.NIMP.size , "NIMP points =")
        else:
            # if we're recalling this routine, begin with known NROY point
            print("= Creating", howMany, "NROY cloud from", self.NROY.shape[0], "NROY points =")

        ## exit if condition already satisfied
        if howMany <= self.NROY.shape[0]:
            print("  Already have", self.NROY.shape[0], "/", howMany, "requested NROY points")
            return

        ## OUTER LOOP - HOW MANY POINTS NEEDED IN TOTAL
        howMany = int(howMany)
        printProgBar(self.NROY.shape[0], howMany, prefix = '  NROY Progress:', suffix = '\n')
        while self.NROY.shape[0] < howMany:
        
            # now LOC and dic can just use NROY in all cases
            LOC, dic = self.NROY, self.NROYminmax

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
                NROYmin = np.amin(self.NROY[:,i])
                NROYmax = np.amax(self.NROY[:,i])
                self.NROYminmax[i] = [NROYmin, NROYmax]

            ## hack part 2 - reset these variables back to normal
            [self.TESTS, self.I, self.NIMP, self.NIMPminmax] = TEMP

