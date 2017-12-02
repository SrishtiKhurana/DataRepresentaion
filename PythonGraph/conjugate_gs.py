import numpy as np
from numpy.linalg import norm
from gram_schmidt import *
from matrix_stuff import *
tol = 1E-8

def maxgs(X,A,noe):
    U = X
    #print("Inital Incidence matrix ",X)
    r,c = X.shape 
    P = U.copy()
    R = U.copy()
    cc= 1
    RC = np.zeros(c)
    
    #select noe edges
    for i in range(0,noe):
        print ("Finding edge no : ",i)
        maxVal = 0 
        maxId = 0 
        for j in range (1,c):
            q = P[:,j]
            numer = q.T.dot(A.dot(q))
            print("numer ",j,": ",numer)
            if (numer > maxVal):
                maxId = j 
                maxVal = numer
                
        
        # fpund the max edge - orthonormalize the others
        max_norm = norm(P[:,maxId])
        print ("Max Id ", maxId)
        print ("Max Num ", maxVal)  
        e = P[:,maxId]    
        u = np.where(e == 1)[0][0]
        v = np.where(e == -1)[0][0]
        print("Edge added between : ", u , " - ",v)
        RC[maxId] = 1
        for j in range(1, c):
            proj = np.dot(P[:, maxId], P[:, j])
            #print("proj",proj)
            temp = P[:, j] - proj*P[:, maxId]
           # print("P[:, j]",P[:, j])
            if np.count_nonzero(temp) == 0:
                print("Zeroed out" ,j) 
            
            P[:,j] = temp       
        print("recomputed P " , P)
        P[:,maxId]= 0
        print("Zeroed : ",P[:, maxId],"--",np.dot(P[:, maxId], P[:, maxId])) 
    #print(P)
    for i in range(0,c):
        if RC[i] == 0:
           R[:,i] = 0
   
    return R  


def mings(X,A,noe):
    U = X
    print("Inital Incidence matrix ",X)
    r,c = X.shape 
    P = U.copy()
    R = U.copy()
    cc= 1
    RC = np.zeros(c)
    
    
    DenomArr = np.zeros(c)
    for i in range(1,c):
        q = P[:,i]
        DenomArr[i] = q.T.dot(A.dot(q))
    
    idc = DenomArr.argsort()[noe:]
    print("DenomArr",DenomArr)
    print("idc",idc)
    
    #select noe edges
    for i in range(1,noe):
        print ("Finding edge no : ",i)
                
        minId = idc[i]
        minVal = DenomArr[i]
        # fpund the max edge - orthonormalize the others
        max_norm = norm(P[:,minId])
        print ("Min Id ", minId)
        print ("Min Num ", minVal)
        RC[minId] = 1
        for j in range(1, c):
            proj = np.dot(P[:, minId], P[:, j])
            P[:, j] = P[:, j] - proj*P[:, minId]
        
        P[:,minId]= 0   
        print (RC)
    #print(P)
    for i in range(1,c):
        if RC[i] == 0:
           R[:,i] = 0
    return R  
  
def new_gs(X,A,noe):
    U = (X)
    r, c = U.shape
    P = U.copy()
    R = U.copy()
    cc = 1
    Rc = np.zeroes(r)
    for i in range(1,noe):
        maxNormId, maxNorm  = get_col_id_of_max_norm(P)
        R[:,cc] = P[:,maxNormId]
        p = P[:,maxNormId]
        Rc[maxNormId] = 1
        #wHAT TO DO HERE ?? 
        for j in range (1,c):
            q = P[:,j]
            denom = q.T.dot(A.dot(q))
            if denom > tol :
                numer = q.T.dot(A.dot(p))
                
            proj = np.dot(p,P[:,j])
            P[:, j] = P[:, j] - proj*p
        P[:,maxNormId] = np.zeros(r)
    for i in range(1,c):
        if Rc[i] == 0:
           R[i] = np.zeroes(r)
    return R

def  new_conj_pinv(X,A,noe):
    P = new_gs(X, A,noe)
    A_pinv = np.zeros(A.shape)
    for i in range(P.shape[1]):
        p = P[:, i]
        d = p.T.dot(A).dot(p)
        if d > tol:
            A_pinv += np.outer(p,p)/d
    return A_pinv

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
