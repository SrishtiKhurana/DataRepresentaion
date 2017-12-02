import argparse
import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import matrix_rank, svd
import networkx as nx
import math
import sys
import os
import re
from conjugate_gs import *
from matrix_stuff import *
from networkx.generators.threshold import cluster_sequence

def GenrateGraphCluster (noc, clustersize):
    G1 = nx.complete_graph(clustersize) 
    mapping = {}
    for i in range(noc-1):
        G2 = nx.complete_graph(clustersize) 
        for j in range(clustersize):
            mapping[j] = ((i+1)*clustersize)+j
        print ("mapping" ,mapping)
        G2 = nx.relabel_nodes(G2, mapping=mapping)
        I = nx.union(G1,G2)
        I.add_edge(mapping[0]-1,mapping[0])
        I.weighted = False
        for e in I.edges_iter():
            I.add_edge(e[0],e[1], weight = 1)
        G1 = I 
    return I 

def ThreeClusterGraph(n1,n2,n3):
    G1 = nx.complete_graph(n1)
    G2 = nx.complete_graph(n2)
    G3 = nx.complete_graph(n3) 
    
    mapping = {}
    for i in range(n2):
        mapping[i] = i+n1
    G2 =  nx.relabel_nodes(G2, mapping=mapping)
    mapping3 = {}    
    for i in range(n3):
        mapping3[i] = i+n1+n2
    G3 =  nx.relabel_nodes(G3, mapping=mapping3)
    I1 = nx.union(G1,G2)
    I1.add_edge(n1-1,n1)
    I1.weighted = False
    
    for e in I1.edges_iter():
        I1.add_edge(e[0],e[1], weight = 1)
    I2 = nx.union(I1,G3)
    
    I2.add_edge(n2+n1-1,n2+n1)
    I2.weighted = False
    #set weight to 1
    for e in I2.edges_iter():
        I2.add_edge(e[0],e[1], weight = 1)
        
    return I2



def FourClusterGraph(n1,n2,n3,n4):
    G = ThreeClusterGraph(n1,n2,n3)
    H = nx.complete_graph(n4)
    mapping = {}
    for i in range(n4):
        mapping[i] = i+n1+n2+n3
    H = nx.relabel_nodes(H, mapping=mapping)

    I = nx.union(G,H)
    I.add_edge(n1+n2+n3-1,n1+n2+n3)
    I.weighted = False
    #set weight to 1
    for e in I.edges_iter():
        I.add_edge(e[0],e[1], weight = 1)

    print(I.number_of_edges())
    print(I.number_of_nodes())
    
    print(I.edges());
    #Draw(I);
    return I


def GenDumbbellGraph(n1, n2):
    """ 
    generates graph with two clusters of n1-clique
    and n2-clique with one node from each clique 
    connected to each other
    """
    G = nx.complete_graph(n1)
    H = nx.complete_graph(n2)

    mapping = {}
    for i in range(n2):
        mapping[i] = i+n1
    H = nx.relabel_nodes(H, mapping=mapping)

    I = nx.union(G,H)
    I.add_edge(n1-1,n1)
    I.weighted = False
    #set weight to 1
    for e in I.edges_iter():
        I.add_edge(e[0],e[1], weight = 1)

    print(I.number_of_edges())
    print(I.number_of_nodes())
    
    print(I.edges());
    #Draw(I);
    return I

def GraphFromIncidenceMatrix(E):
    G = nx.Graph()
    n, m = E.shape
    for i in range(n):
        G.add_node(i)

    for e in E.T:
        if np.all(np.isclose(e, 0)):
            continue
        u = np.where(e == 1)[0][0]
        v = np.where(e == -1)[0][0]
        G.add_edge(u, v)
        print("Edge added between : ", u , " - ",v)

    return G


def IncidenceMatrix(G):
    n = len(G.nodes())
    e = len(G.edges())
    X = np.zeros((n,e))
    for idx, edge in enumerate(G.edges(data = True)):
        X[edge[0], idx] = 1
        X[edge[1], idx] = -1
    #print('Incidence matrix ')
    #print(X)
    return X

def DegreeMatrix(G):
    degrees = nx.degree(G).values()
    matrix = np.diag(list(degrees))
    return matrix

def AdjacancyMatrix(G):
    return nx.attr_matrix(G)[0]

def Laplacian(G):
    E = IncidenceMatrix(G)
    L = E.dot(E.T)
    # Sanity Check
    # L should be same as 
    # DegreeMatrix(G) - AdjacancyMatrix(G)
    # Laplacian generated with nx : 
    # scipy.sparse.csr_matrix.todense(nx.laplacian_matrix(G))
    return L

def Draw(G):
    pos = nx.fruchterman_reingold_layout(G)
    nx.draw_networkx(G, node_color = 'orange', alpha = 0.6, pos = pos)
    plt.show()


def test_pinv():
    G = GenDumbbellGraph( 5,5)
    G = nx.erdos_renyi_graph(50,0.15)
    G= nx.wheel_graph(50, create_using=None)
    
    deg_tri=[[1,0],[1,0],[1,0],[2,0],[1,0],[2,1],[0,1],[0,1]]
    G = nx.random_clustered_graph(deg_tri)
    
    noc = 3 # number of clusters
    npc = 5 # node per cluster
    G = GenrateGraphCluster(noc,npc)
   # G = ThreeClusterGraph(10,10,10)
    #G = FourClusterGraph(5,10,8,2)
    plt.subplot(2,2,1)
    pos = nx.fruchterman_reingold_layout(G)
    nx.draw_networkx(G, node_color = 'orange', alpha = 0.6, pos = pos)
    plt.plot()
    plt.title('Orignal Graph')
    
    #G = nx.complete_graph(10)
    L = Laplacian(G)
    E = IncidenceMatrix(G)
    L_plus = np.linalg.pinv(L)

    print("with L ", norm(conjugate_pinv(L, L) - L_plus))
    print("with E ", norm(conjugate_pinv(E, L) - L_plus))
    
    noe = 2
    NG = maxgs(E,L,noe)
    print ("NG",NG)
    NGP = GraphFromIncidenceMatrix(NG)
    plt.subplot(2, 2, 2)
    nx.draw_networkx(NGP, node_color = 'orange', alpha = 0.6, pos = pos)
    plt.plot()
    plt.title('Reduced Graph with 2 nodes')
    
    noe = 3
    NG = maxgs(E,L,noe)
    print ("NG",NG)
    NGP = GraphFromIncidenceMatrix(NG)
    plt.subplot(2, 2, 3)
    nx.draw_networkx(NGP, node_color = 'orange', alpha = 0.6, pos = pos)
    plt.plot()
    plt.title('Reduced Graph with 3 nodes')
    
    noe = 12
    NG = maxgs(E,L,noe)
    print ("NG",NG)
    NGP = GraphFromIncidenceMatrix(NG)
    plt.subplot(2, 2, 4)
    nx.draw_networkx(NGP, node_color = 'orange', alpha = 0.6, pos = pos)
    plt.plot()
    plt.title('Reduced Graph with 12 nodes')
    plt.show()
    #Draw(NG
    print("with new ",norm(NG))
    
    ''''
    #Min graph - Does not work -------

   # NG = mings(E,L,noe)
    print ("NG",NG)
    NGP = GraphFromIncidenceMatrix(NG)
   # nx.draw_networkx(NGP, node_color = 'orange', alpha = 0.6, pos = pos)
   # plt.show()
    print("with new ",norm(NG))
    
    

    ES = E[:, sort_cols_by_norm(E)]
    print("sorted E ", norm(conjugate_pinv(ES, L) - L_plus))

    EL = np.zeros(E.shape)
    n = E.shape[1]; ES = E[:, np.random.randint(n, size = n)]
    print("shuffled E ", norm(conjugate_pinv(EL, L) - L_plus))

    EC = E[:, sort_cols_conjugate(E, L)]
    print("conj sorted E ", norm(conjugate_pinv(EC, L) - L_plus))

    n = L.shape[1]
    I = np.eye(n)
    R = np.random.uniform(size = (n,2*n))
   
    print("Identity",norm(conjugate_pinv(I, L) - L_plus))
    print("Random", norm(conjugate_pinv(R, L) - L_plus))

    RL = np.zeros(R.shape)
    n = RL.shape[1]; RS = RL[:, np.random.randint(n, size = n)]
    print("Random shuffled", norm(conjugate_pinv(RS, L) - L_plus))

    n = L.shape[1]; RL[:,range(n)] = L[:,range(n)]
    print("Random + L cols (first)", norm(conjugate_pinv(RL, L) - L_plus)) 
    '''
    


    #n = L.shape[1]; RL[:,range(20,n+20)] = L[:,range(n)]
    #print("Random + L cols (middle)", norm(conjugate_pinv(RL, L) - L_plus)) 

if __name__ == "__main__":
    #G = GenDumbbellGraph(7,8)
    G = nx.complete_graph(10)
    L = Laplacian(G)
    E = IncidenceMatrix(G)
    L_plus = np.linalg.pinv(L)

    #print(np.linalg.svd(L)[1])

    ##for i in range(E.shape[1]):
    ##    print(norm(proj(E[:,i], L)))

    #for i in range(E.shape[1]):
    #    print(norm(proj_ortho_to_null(E[:,i], L)))

    #print()
    #for i in range(L.shape[1]):
    #    print(norm(proj_ortho_to_null(L[:,i], L)))
    #
    test_pinv()
