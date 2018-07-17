
## fill out NROY space to use as tests for next wave
def cloudNROY(self, howMany, maxno=1, factor = 0.1, chunkSize=5000, restart=False):

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

