'''
Created on 24-Aug-2017

@author: Srishti
'''
import networkx as nx
import matplotlib.pyplot as plt

G=nx.Graph()
i=1
G.add_node(i,pos=(i,i))
G.add_node(2,pos=(2,2))
G.add_node(3,pos=(1,0))
G.add_edge(1,2,weight=0.5)
G.add_edge(1,3,weight=9.8)
pos=nx.get_node_attributes(G,'pos')
nx.draw(G,pos)
plt.savefig("path.png")