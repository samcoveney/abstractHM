from abc import ABC, abstractmethod
import numpy as np


# abstract base class for emulators
# ---------------------------------
class AbstractEmulator(ABC, object):

    def __init__(self, model):
        AbstractEmulator.model = model

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


