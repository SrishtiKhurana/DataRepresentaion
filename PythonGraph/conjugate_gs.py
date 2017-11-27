import numpy as np
from numpy.linalg import norm
from gram_schmidt import *
from matrix_stuff import *
tol = 1E-8

def new_gs(X,A,noe):
    U = (X)
    r, c = U.shape
    P = U.copy()
    R =  np.zeros((r, noe))
    cc = 1
    Rc = np.zeroes(r)
    for i in range(1,noe):
        maxNormId, maxNorm  = get_col_id_of_max_norm(P)
        R[:,cc] = U[:,maxNormId]
        p = P[:,maxNormId]
        Rc[maxNormId] = 1 
        for j in range (1,c):
            q = P[:,j]
            denom = q.T.dot(A.dot(q))
            if denom > tol :
                numer = q.T.dot(A.dot(p))
                
            proj = np.dot(p,P[:,j])
            P[:, j] = P[:, j] - proj*p
        P[:,maxNormId] = np.zeros(r)
        
    return R


def conjugate_gs(X, A):
    #U = X
    U = remove_zero_cols(modified_gs(X))
    r, c = U.shape
    P = U.copy()
    for i in range(1,c):
        p = P[:, i]
        for j in range(i):
            q = P[:, j]
            denom = q.T.dot(A.dot(q))
            # print("q.T ",q.T)
            #  print ("D[i] : %d ",denom)
            # D[i] = denom;
            if denom > tol:
                numer = q.T.dot(A.dot(p))
                p = p - (numer/denom)*q
    return P

def conjugate_pinv(X, A):
    P = conjugate_gs(X, A)
    A_pinv = np.zeros(A.shape)
    for i in range(P.shape[1]):
        p = P[:, i]
        d = p.T.dot(A).dot(p)
        if d > tol:
            A_pinv += np.outer(p,p)/d
    return A_pinv
