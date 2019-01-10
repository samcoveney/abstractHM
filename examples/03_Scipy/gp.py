import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
np.set_printoptions(precision=10)

#----------

def plot_dataset(Xdata, Ydata, xlabels, ylabels):
	in_dim = Xdata.shape[1]
	out_dim = Ydata.shape[1]
	sample_dim = Xdata.shape[0]

	fig, axes = plt.subplots(nrows=out_dim, ncols=in_dim, sharex='col', sharey='row', figsize=(16, 16))                   
	for i, a in enumerate(axes.flatten()):
		a.scatter(Xdata[:, i % in_dim], Ydata[:, i // in_dim])
		if i // in_dim == out_dim - 1:
			a.set_xlabel(xlabels[i % in_dim])
		if i % in_dim == 0:
			a.set_ylabel(ylabels[i // in_dim])
	plt.suptitle('Sample dimension = {} points'.format(sample_dim))
	plt.show()

	return

## main ##

Xdata = np.loadtxt('inputs_7_693.txt', dtype=float)
Ydata = np.loadtxt('outputs_11_693.txt', dtype=float)

in_dim = Xdata.shape[1]
out_dim = Ydata.shape[1]

xlabels = ['p', 'ap', 'z', 'c1', 'ca50', 'kxb', 'Tref']
ylabels = ['EDV', 'ESV', 'EF', 'ICT', 'ET', 'IRT', 'Tdiast', 'PeakP', 'Tpeak', 'maxdP', 'mindP']

# plot_dataset(Xdata, Ydata, xlabels, ylabels) # uncomment to spot pair-wise correlations

pipe = Pipeline([
        ('poly', PolynomialFeatures()),
        ('lr', LinearRegression())
])

kfolds = None

par_grid1 = {"poly__degree": [1, 2, 3, 4, 5]}
gs1 = GridSearchCV(pipe, par_grid1, cv=kfolds, n_jobs=-1)

par_grid2 = {"kernel": [Matern(length_scale=in_dim*[1.0], nu=i) for i in [1.5, 2.5]] + [RBF(length_scale=in_dim*[1.0])]}
gs2 = GridSearchCV(GaussianProcessRegressor(n_restarts_optimizer=10), par_grid2, cv=kfolds, n_jobs=-1)

best_models = []
emulators = []

print('\n=== Fitting {} Gaussian Process emulators (one for each output feature):'.format(1))
for i in range(2):
        gs1.fit(Xdata, Ydata[:, i])
        best_models.append(gs1.best_estimator_)

        poly = best_models[i].steps[0][1]
        lr = best_models[i].steps[1][1]

        Xdata_ = poly.transform(Xdata)
        mean_data = lr.predict(Xdata_)
        Ydata_new = Ydata[:, i] - mean_data

        gs2.fit(Xdata, Ydata_new)
        gp = gs2.best_estimator_
        emulators.append(gp)

for c, (bm, gp) in enumerate(zip(best_models, emulators)):
        print('=== Fitted linear regressor and emulator obtained for output feature #{}:\n'.format(c))
        print(bm.steps[0][1])
        print(gp.kernel_)
        print('\n\n')


# simple wrapper, because abstract class requires a single class argument
# -----------------------------------------------------------------------

# this is the very simple wrapper...
class model():
    def __init__(self, mean, gp, X, Y):
        self.mean = mean
        self.gp = gp
        self.X = X
        self.Y = Y

for i in range(2):

    # gp & mean
    ee = emulators[i]
    bb = best_models[i]
    testModel = model(bb, ee, Xdata, Ydata[:,i])

    # pickle the object, so I don't have to keep rerunning code...
    import pickle
    with open("testModel" + str(i) + ".pkl", "wb") as pickle_file:
        pickle.dump(testModel, pickle_file)


