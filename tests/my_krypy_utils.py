'''
Collection of standard functions.
This method provides functions like inner products, norms, ...
'''

import numpy
import warnings
import time
import scipy.linalg
from scipy.sparse import isspmatrix
#from scipy.sparse.linalg import LinearOperator, aslinearoperator
from scipy.sparse.sputils import isintlike

def inner(X, Y, ip_B=None):
    '''inner product
    '''

    return numpy.dot(X.T, Y)

def norm(x, y=None, ip_B=None):
    '''Compute norm (Euclidean and non-Euclidean).
    :param x: a 2-dimensional ``numpy.array``.
    :param y: a 2-dimensional ``numpy.array``.
    :param ip_B: see :py:meth:`inner`.
    Compute :math:`\sqrt{\langle x,y\rangle}` where the inner product is
    defined via ``ip_B``.
    '''

    return numpy.linalg.norm(x)


class Arnoldi(object):
    def __init__(self, A, v,
                 maxiter=None,
                 ortho='mgs',
                 M=None,
                 Mv=None,
                 Mv_norm=None,
                 ip_B=None
                 ):
        """Arnoldi algorithm.
        Computes V and H such that :math:`AV_n=V_{n+1}\\underline{H}_n`.  If
        the Krylov subspace becomes A-invariant then V and H are truncated such
        that :math:`AV_n = V_n H_n`.
        :param A: a linear operator that can be used with scipy's
          aslinearoperator with ``shape==(N,N)``.
        :param v: the initial vector with ``shape==(N,1)``.
        :param maxiter: (optional) maximal number of iterations. Default: N.
        :param ortho: (optional) orthogonalization algorithm: may be one of
            * ``'mgs'``: modified Gram-Schmidt (default).
            * ``'dmgs'``: double Modified Gram-Schmidt.
            * ``'lanczos'``: Lanczos short recurrence.
            * ``'house'``: Householder.
        :param M: (optional) a self-adjoint and positive definite
          preconditioner. If ``M`` is provided, then also a second basis
          :math:`P_n` is constructed such that :math:`V_n=MP_n`. This is of
          importance in preconditioned methods. ``M`` has to be ``None`` if
          ``ortho=='house'`` (see ``B``).
        :param ip_B: (optional) defines the inner product to use. See
          :py:meth:`inner`.
          ``ip_B`` has to be ``None`` if ``ortho=='house'``. It's unclear to me
          (andrenarchy), how a variant of the Householder QR algorithm can be
          used with a non-Euclidean inner product. Compare
          http://math.stackexchange.com/questions/433644/is-householder-orthogonalization-qr-practicable-for-non-euclidean-inner-products
        """
        N = v.shape[0]

        # save parameters
        self.A = A #get_linearoperator((N, N), A)
        self.maxiter = N if maxiter is None else maxiter
        self.ortho = ortho
        self.M = None #get_linearoperator((N, N), M)
        #if isinstance(self.M, IdentityLinearOperator):
        #    self.M = None
        self.ip_B = ip_B

        self.dtype = float # find_common_dtype(A, v, M)
        # number of iterations
        self.iter = 0
        # Arnoldi basis
        self.V = numpy.zeros((N, self.maxiter+1), dtype=self.dtype)
        if self.M is not None:
            self.P = numpy.zeros((N, self.maxiter+1), dtype=self.dtype)
        # Hessenberg matrix
        self.H = numpy.zeros((self.maxiter+1, self.maxiter),
                             dtype=self.dtype)
        # flag indicating if Krylov subspace is invariant
        self.invariant = False

        if ortho in ['mgs', 'dmgs', 'lanczos']:
            self.reorthos = 0
            if ortho == 'dmgs':
                self.reorthos = 1
            if self.M is not None:
                p = v
                if Mv is None:
                    v = self.M*p
                else:
                    v = Mv
                if Mv_norm is None:
                    self.vnorm = norm(p, v, ip_B=ip_B)
                else:
                    self.vnorm = Mv_norm
                if self.vnorm > 0:
                    self.P[:, [0]] = p / self.vnorm
            else:
                if Mv_norm is None:
                    self.vnorm = norm(v, ip_B=ip_B)
                else:
                    self.vnorm = Mv_norm
        else:
            raise RuntimeError(
                'Invalid value \'{0}\' for argument \'ortho\'. '.format(ortho)
                + 'Valid are house, mgs, dmgs and lanczos.')
        if self.vnorm > 0:
            self.V[:, [0]] = v / self.vnorm
        else:
            self.invariant = True

    def advance(self):
        """Carry out one iteration of Arnoldi."""
        if self.iter >= self.maxiter:
            raise ArgumentError('Maximum number of iterations reached.')
        if self.invariant:
            raise ArgumentError('Krylov subspace was found to be invariant '
                                'in the previous iteration.')

        N = self.V.shape[0]
        k = self.iter

        # the matrix-vector multiplication
        Av = self.A * self.V[:, [k]]

        # determine vectors for orthogonalization
        start = 0

        # (double) modified Gram-Schmidt
        for reortho in range(self.reorthos+1):
            # orthogonalize
            for j in range(start, k+1):
                alpha = inner(self.V[:, [j]], Av, ip_B=self.ip_B)[0, 0]
                self.H[j, k] += alpha
                Av -= alpha * self.V[:, [j]]

        self.H[k+1, k] = norm(Av, ip_B=self.ip_B)

        if self.H[k+1, k] / numpy.linalg.norm(self.H[:k+2, :k+1], 2)\
                <= 1e-14:
            self.invariant = True
        else:
            self.V[:, [k+1]] = Av / self.H[k+1, k]

        # increase iteration counter
        self.iter += 1

    def get(self):
        k = self.iter
        if self.invariant:
            V, H = self.V[:, :k], self.H[:k, :k]
            if self.M:
                return V, H, self.P[:, :k]
            return V, H
        else:
            V, H = self.V[:, :k+1], self.H[:k+1, :k]
            if self.M:
                return V, H, self.P[:, :k+1]
            return V, H

    def get_last(self):
        k = self.iter
        if self.invariant:
            V, H = None, self.H[:k, [k-1]]
            if self.M:
                return V, H, None
            return V, H
        else:
            V, H = self.V[:, [k]], self.H[:k+1, [k-1]]
            if self.M:
                return V, H, self.P[:, [k]]
            return V, H


def arnoldi(*args, **kwargs):
    _arnoldi = Arnoldi(*args, **kwargs)
    while _arnoldi.iter < _arnoldi.maxiter and not _arnoldi.invariant:
        _arnoldi.advance()
    return _arnoldi.get()

