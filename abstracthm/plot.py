import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pickle
from abstracthm.utils import *


## colormaps
def myGrey():
    #return '#696988'
    return 'lightgrey'

def colormap(cmap, b, t, mode="imp"):
    n = 500
    cb   = np.linspace(b, t, n)
    cm = cmap( cb )
    #cm[0:50,3] = np.linspace(0.80,1.0,50) # adjust alphas (transparencies)
    new_cmap = colors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=b, b=t), cm )
    #new_cmap.set_under(color=myGrey())
    #print(new_cmap)
    return new_cmap

## implausibility and optical depth plots for all pairs of active indices
def plotImp(wave, maxno=1, grid=10, filename="hexbin.pkl", points=[], odp=True, sims=False, replot=False, colorbar=True, activeId = [], NROY=False, NIMP=True, manualRange={}, vmin=1.0, vmax=None, names=None):

    print("= Creating History Matching plots =")

    ## option to plot NROY against ODP
    if NROY == True and wave.NROY == []:
        print("█ WARNING: cannot using NROY = True because NROY not calculated"); exit()
    if NIMP == False and NROY == False:
        print("█ WARNING: cannot have NIMP = False and NROY = False because nothing to plot"); exit()

    ## useful messages
    print("  Considering points NIMP:", NIMP, "& NROY:", NROY)
 
    ## make list of all the active indices across all emulators
    active = []
    for s in wave.subwaves:
        for a in s.emul.active():
            if a not in active: active.append(a)
    active.sort()
    print("  Active features:", active)

    ## restrict to smaller set of active indices
    if activeId != []:
        for a in activeId:
            if a not in active:
                print("ERROR: activeId", a, "not an active emulator index"); exit()
        active = activeId; active.sort()

    ## reference global index into subplot index
    pltRef = {}
    for count, key in enumerate(active): pltRef[key] = count
    print("  {Features: Subplots} : ", pltRef)

    ## create list of all pairs of active inputs
    gSets = []
    for i in active:
        for j in active:
            if i!=j and i<j and [i,j] not in gSets:  gSets.append([i,j])
    #print("  GLOBAL SETS:", gSets)

    if replot == False:

        ## determine the max'th Implausibility
        print("  Determining", maxno, "max'th implausibility...")
        # NOTE: not using multivariate implausibility for plotting, since MVI values on different scale
        if NIMP == True:
            T = wave.TESTS
            Imaxes = np.partition(wave.I, -maxno)[:,-maxno]
            if NROY == True:
                T = np.concatenate( (T, wave.NROY), axis=0)
                Imaxes = np.concatenate((Imaxes, np.partition(wave.NROY_I, -maxno)[:,-maxno]),axis=0)
        else:
            T = wave.NROY
            Imaxes = np.partition(wave.NROY_I, -maxno)[:,-maxno]

        ## plots simulation points colored by imp (no posterior variance since these are sim points)
        print("  Plotting simulations points coloured by implausibility...")
        if sims:
            simPoints, Y, IsimMaxes, pointsXnimp, pointsYnimp = wave.simImp()
            Temp = np.hstack([IsimMaxes[:,None], simPoints])
            Temp = Temp[(-Temp[:,0]).argsort()] # sort by Imp, lowest first...
            IsimMaxes, simPoints = Temp[:,0], Temp[:,1:]

        ## space for all plots, and reference index to subplot indices
        print("  Creating HM plot objects...")
        rc = len(active)
        fig, ax = plt.subplots(nrows = rc, ncols = rc)

        print("  Making subplots of paired indices...")
        printProgBar(0, len(gSets), prefix = '  Progress:', suffix = '')
        for i, s in enumerate(gSets):

            # reference correct subplot
            impPlot, odpPlot = ax[pltRef[s[1]],pltRef[s[0]]], ax[pltRef[s[0]],pltRef[s[1]]]

            # set background color of plot 
            impPlot.patch.set_facecolor(myGrey()); odpPlot.patch.set_facecolor(myGrey())

            # imp subplot - bin points by Imax value, 'reduce' bin points by minimum of these Imaxes
            # FIXME: extent keyword here needs to be generalized
            im_imp = impPlot.hexbin(
              T[:,s[0]], T[:,s[1]], C = Imaxes,
              gridsize=grid, cmap=colormap(plt.get_cmap('nipy_spectral'),0.60,0.825), vmin=vmin, vmax=wave.cm if vmax is None else vmax,
              #gridsize=grid, cmap=colormap(plt.get_cmap('hot'),1.0,0.0), vmin=vmin, vmax=wave.cm if vmax is None else vmax,
              extent=( T[:,s[0]].min(),T[:,s[0]].max(),T[:,s[1]].min(),T[:,s[1]].max() ),
              linewidths=0.2, mincnt=1, reduce_C_function=np.min)
            if colorbar: plt.colorbar(im_imp, ax=impPlot); 

            # odp subplot - bin points if Imax < cutoff, 'reduce' function is np.mean() - result gives fraciton of points satisfying Imax < cutoff
            # FIXME: extent keyword here needs to be generalized
            if sims == True:
                #im_odp = odpPlot.scatter(simPoints[:,s[0]], simPoints[:,s[1]], s=25, c=IsimMaxes, cmap=colormap(plt.get_cmap('nipy_spectral'),0.60,0.825), vmin=vmin, vmax=wave.cm if vmax is None else vmax)#, edgecolor='black')
                im_odp = odpPlot.scatter(simPoints[:,s[0]], simPoints[:,s[1]], s=25, c=IsimMaxes, cmap=colormap(plt.get_cmap('hot'),1.0,0.0), vmin=vmin, vmax=wave.cm if vmax is None else vmax)#, edgecolor='black')
            if odp == True and sims == False:
                im_odp = odpPlot.hexbin(
                  T[:,s[0]], T[:,s[1]], C = Imaxes<wave.cm,
                  gridsize=grid, cmap=colormap(plt.get_cmap('gist_stern'),1.0,0.25, mode="odp"), vmin=0.0, vmax=None, # vmin = 0.00000001, vmax=None,
                  extent=( T[:,s[0]].min(),T[:,s[0]].max(),T[:,s[1]].min(),T[:,s[1]].max() ),
                  linewidths=0.2, mincnt=1)
            if colorbar and (odp or sims): plt.colorbar(im_odp, ax=odpPlot)

            # force equal axes
            #impPlot.set_xlim(exPlt[0], exPlt[1]); impPlot.set_ylim(exPlt[2], exPlt[3])
            #odpPlot.set_xlim(exPlt[0], exPlt[1]); odpPlot.set_ylim(exPlt[2], exPlt[3])

            printProgBar(i+1, len(gSets), prefix = '  Progress:', suffix = '')


        ## can set labels on diagaonal
        for i, a in enumerate(active):
            ax[a,a].set(adjustable='box-forced', aspect='equal')
            if names is not None:
                ax[a,a].text(.5,.5,names[i], horizontalalignment='center', transform=ax[a,a].transAxes)        
 # + "\n"
            else:
                ax[a,a].text(.25,.5,"Input " + str(a)) # + "\n"
            ax[a,a].set_xticks([]); ax[a,a].set_yticks([]);
               #+ str(minmax[key][0]) + "\n-\n" + str(minmax[key][1]))


        # delete 'empty' central plot
        #for a in range(rc): fig.delaxes(ax[a,a])

        # force range of plot to be correct
        for a in ax.flat:
            a.set(adjustable='box-forced', aspect='equal')
            x0,x1 = a.get_xlim(); y0,y1 = a.get_ylim()
            a.set_aspect((x1-x0)/(y1-y0))
            #a.set_xticks([]); a.set_yticks([]); 

        ## set the ticks on the edges
        for i in range(len(active)):
            for j in range(len(active)):
                if i != len(active) - 1:
                    ax[i,j].set_xticks([])
                if j != 0:
                    ax[i,j].set_yticks([])

        ## set useful extra tics
        for ab in [[0, len(active)-1], [len(active)-1,len(active)-1]]:
            a, b = ab[0], ab[1]

        a, b = 0, len(active)-1
        ax[a,a].set_yticks(ax[b,a].get_xticks())
        ax[a,a].set_xlim(ax[b,a].get_xlim()); ax[a,a].set_ylim(ax[b,a].get_xlim())
        ax[a,a].set(adjustable='box-forced', aspect='equal')
        
        ax[b,b].set_xticks(ax[b,a].get_yticks())
        ax[b,b].set_xlim(ax[b,a].get_ylim()); ax[b,b].set_ylim(ax[b,a].get_ylim())
        ax[b,b].set(adjustable='box-forced', aspect='equal')

        #plt.tight_layout()

        #print("  Pickling plot in", filename)
        #pickle.dump([fig, ax], open(filename, 'wb'))  # save plot - for Python 3 - py2 may need `file` instead of `open`
    else:
        print("  Unpickling plot in", filename, "...")
        fig, ax = pickle.load(open(filename,'rb'))  # load plot

    ## plots points passed to this function
    if isinstance(points, list) == False:
        print("█ WARNING: 'points' must be a list of either [inputs] or [inputs, outputs], where inputs/outputs are numpy arrays"); return
    if points is not []:
        if len(points) == 1:
            pointsX = points[0]
            print("  Plotting 'points'...")
            for s in gSets:
                ax[pltRef[s[1]],pltRef[s[0]]].scatter(pointsX[:,s[0]], pointsX[:,s[1]], s=25, c='black')
                ax[pltRef[s[0]],pltRef[s[1]]].scatter(pointsX[:,s[0]], pointsX[:,s[1]], s=25, c='black')
        if len(points) == 2:
            print("  Plotting 'points' coloured by implausibility (assuming these points are simulation points...)")
            pointsX, pointsY, IsimMaxes, pointsXnimp, pointsYnimp = wave.simImp()
            IsimMaxes = np.partition(Isim, -maxno)[:,-maxno] # NOTE: not for multivariate implausibility
            Temp = np.hstack([IsimMaxes[:,None], pointsX])
            Temp = Temp[(-Temp[:,0]).argsort()] # sort by Imp, lowest first...
            IsimMaxes, pointsX = Temp[:,0], Temp[:,1:]
            for s in gSets:
                #ax[pltRef[s[1]],pltRef[s[0]]].scatter(pointsX[:,s[0]], pointsX[:,s[1]], s=25, c=IsimMaxes, cmap=colormap(plt.get_cmap('nipy_spectral'),0.60,0.85), vmin=vmin, vmax=wave.cm, edgecolor='black')
                #pointImp = ax[pltRef[s[0]],pltRef[s[1]]].scatter(pointsX[:,s[0]], pointsX[:,s[1]], s=25, c=IsimMaxes, cmap=colormap(plt.get_cmap('nipy_spectral'),0.60,0.825), vmin=vmin, vmax=wave.cm if vmax is None else vmax)#, edgecolor='black')
                pointImp = ax[pltRef[s[0]],pltRef[s[1]]].scatter(pointsX[:,s[0]], pointsX[:,s[1]], s=25, c=IsimMaxes, cmap=colormap(plt.get_cmap('hot'),1.0,0.0), vmin=vmin, vmax=wave.cm if vmax is None else vmax)#, edgecolor='black')
                if colorbar and not(sims) and not(odp): plt.colorbar(pointImp, ax=ax[pltRef[s[0]],pltRef[s[1]]])

    plt.show()
    return


