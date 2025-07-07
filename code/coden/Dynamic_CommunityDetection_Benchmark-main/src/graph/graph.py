from networkx.generators.community import LFR_benchmark_graph
import networkx as nx
import numpy as np
import community

def generate_weighted_LFR_graph(nodes=1000, tau1=3, tau2=1.5, mu=0.1, seed=10):
    # Initialize the LFR graph
    G = LFR_benchmark_graph(nodes, tau1, tau2, 
                            mu, average_degree=10, 

                            
                            min_community=int(0.1*nodes),
                            max_community=int(0.4*nodes),

                            # 1k of nodes
                            #min_community=50,
                            
                             seed=seed, 
                            max_iters=100)
    G.remove_edges_from(nx.selfloop_edges(G)) # Remove self-loops
    init_lookup = dict()
    comm_cnt = 0
    for node in G.nodes():
        if node not in init_lookup:
            for same_comm_node in G.nodes[node]['community']:
                init_lookup[same_comm_node] = comm_cnt
            comm_cnt+=1
    # Run the first louvain to detect the "ground truth communities"
    #nx.set_node_attributes(G, init_lookup, name='community')
    
    # Assign the weights: 
    #   Within community edge [0.7, 1.0]
    #   Between community edge [0.3, 0.6]
    for u, v in G.edges():
        if init_lookup[u] == init_lookup[v]:
            weight = 0.7 + 0.3 * np.random.random()
            assert(weight>=.7)
        else:
            weight = 0.3 + 0.3 * np.random.random()
            assert(weight<.7)
        G[u][v]['weight'] = weight   
    
    return G,init_lookup