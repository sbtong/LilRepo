import numpy as np
import pandas

import statsmodels.api as sm

def nullspace(A, atol=1e-13, rtol=0):
    """Compute an approximate basis for the nullspace of A.

    The algorithm used by this function is based on the singular value
    decomposition of `A`.

    A : ndarray
        A should be at most 2-D.  A 1-D array with length k will be treated
        as a 2-D with shape (1, k)
    atol : float
        The absolute tolerance for a zero singular value.  Singular values
        smaller than `atol` are considered to be zero.
    rtol : float
        The relative tolerance.  Singular values less than rtol*smax are
        considered to be zero, where smax is the largest singular value.

    If both `atol` and `rtol` are positive, the combined tolerance is the
    maximum of the two; that is::
        tol = max(atol, rtol * smax)
    Singular values smaller than `tol` are considered to be zero.

    ns : ndarray
        If `A` is an array with shape (m, k), then `ns` will be an array
        with shape (k, n), where n is the estimated dimension of the
        nullspace of `A`.  The columns of `ns` are a basis for the
        nullspace; each element in numpy.dot(A, ns) will be approximately
        zero.
    """

    A = np.atleast_2d(A)
    u, s, vh = np.linalg.svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return ns

class ConstrainedLinearModel(object):
    """ y = X f + e
         C f = 0
    """
    def __init__(self, Y, X, C, weights=None):
        Y = Y.reindex(index=X.index, copy=False)
        self.Y = Y
        self.X = X
        
        if np.any(np.any(pandas.isnull(self.Y),axis=0),axis=0):
            raise Exception('Y variable contains missing values')
        if np.any(np.any(pandas.isnull(self.X),axis=0),axis=0):
            raise Exception('X variable contains missing values')
        self.C = C.reindex(columns=X.columns, copy=False)
        self.weights = pandas.Series(1.0, index=X.index)
        if weights is not None:
            self.weights = weights.reindex(index=X.index)
        if np.any(pandas.isnull(self.weights)):
            raise Exception('Weights are missing values')
        self._result = None
        
    def __compute(self):
        if isinstance(self.Y, pandas.DataFrame):
            raise Exception('matrix left-hand side currently not supported')
        y = self.Y.copy()
        N = pandas.DataFrame(nullspace(self.C.values),index=self.X.columns)
        Xbar = self.X.dot(N)
        result = sm.WLS(y, Xbar, weights=self.weights).fit() 
        fcovbar = result.cov_params()
        
        self._params = N.dot(result.params) 
        self._result = result
        self._fcov = N.dot(fcovbar).dot(N.T)
        self._bse = pandas.Series(np.sqrt(np.diag(self._fcov)), index=self._fcov.index)
        self._tvalues = self._params/self._bse
    
    @property
    def params(self):
        if self._result is None:
            self.__compute()
        return self._params

    @property
    def tvalues(self):
        if self._result is None:
            self.__compute()
        return self._tvalues

    @property
    def fcov(self):
        if self._result is None:
            self.__compute()
        return self._fcov
    
    @property
    def rsquared(self):
        if self._result is None:
            self.__compute()
        return self._result.rsquared
    
    @property
    def rsquared_adj(self):
        if self._result is None:
            self.__compute()
        return self._result.rsquared_adj
