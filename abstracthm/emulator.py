from abc import ABC, abstractmethod
import numpy as np


# abstract base class for emulators
# ---------------------------------
class AbstractEmulator(ABC, object):

    def __init__(self, model):
        #AbstractEmulator.model = model
        self.model = model

    @abstractmethod
    def data(self):
        pass

    @abstractmethod
    def minmax(self):
        pass

    @abstractmethod
    def active(self):
        pass

    @abstractmethod
    def posterior(self):
        pass


# class for wrapping GPy methods
# ------------------------------
class GPyEmulator(AbstractEmulator):

    def data(self):
        x, y = self.model.X, self.model.Y
        return x, y

    def minmax(self):
        x = self.model.X
        minmax = {}
        for i, xc in enumerate(x.T):
            minmax[i] = [np.min(xc), np.max(xc)]
        return minmax

    def active(self):
        return self.model.kern.active_dims

    def posterior(self, x):
        m, v = self.model.predict(x, full_cov=False, include_likelihood=False)
        return m[:,0], v[:,0]


# class for wrapping scikit learn model with separate mean and gp
# ---------------------------------------------------------------
class sclEmulator(AbstractEmulator):

    def data(self):
        # FIXME: since the gp is trained on the residuals, it is not the original data...
        # x, y = self.model.gp.X_train_, self.model.gp.y_train_
        x, y = self.model.X, self.model.Y
        return x, y

    def minmax(self):
        x = self.model.X
        minmax = {}
        for i, xc in enumerate(x.T):
            minmax[i] = [np.min(xc), np.max(xc)]
        return minmax

    def active(self):
        # I believe all input dimensions are used
        return np.arange(self.model.X.shape[1])

    def posterior(self, x):
        m, s = self.model.gp.predict(x, return_std=True)
        m = m + self.model.mean.predict(x)
        return m, s**2


# class for wrapping table as if it were an emulator
# --------------------------------------------------
class TableEmulator:

    def data(self):
        x, y = self.model.X, self.model.Y
        return x, y

    def minmax(self):
        x = self.model.X
        minmax = {}
        for i, xc in enumerate(x.T):
            minmax[i] = [np.min(xc), np.max(xc)]
        return minmax

    def active(self):
        # return list of integers for column indices
        return np.arange(self.model.X.shape[1])

    def posterior(self, x):
        
        # get index of self.model.X row that matches first row in x
        #print( np.where((self.model.X == x[0,:]).all(axis=1)) )
        index1 = int(np.where((self.model.X == x[0,:]).all(axis=1))[0][0] ) # NOTE: why do I get match to two rows sometimes?
        index2 = index1 + x.shape[0]
        #print(index1, index2)

        m = self.model.Y[ index1:index2 ]  # return rows of Y with index matching x[:,0] which are row indices
        v = 0.0  # NOTE: zero variance, might get an error if this is not an array...
        return m, v

