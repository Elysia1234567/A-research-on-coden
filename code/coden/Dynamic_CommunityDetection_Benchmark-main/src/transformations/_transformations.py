import pandas as pd
import numpy as np
from networkx import from_pandas_edgelist
import random
import networkx as nx

CLIP_MIN, CLIP_MAX = .001, 1
BATCH_MIN, BATCH_MAX = 0.005,0.02
BATCH_MIN_RED, BATCH_MAX_RED = 0.005,0.02

TURN_OFF_NODES = .2


def split_community(G, subarrays, delta):
    """ Split a community by redistributing edge weights between two 
    sub-communities.

    Parameters
    ----------
        G (networkx.Graph): The input graph.
        subarrays (list of two lists): Two sub-community lists representing the 
            communities to be split.
        delta (float): The amount of weight to be redistributed.

    Returns
    -------
        G_moved (networkx.Graph): The graph after the community split with 
            redistributed edge weights.

    """
    # Get the two sub-communities to split
    sub0, sub1 = subarrays
    weights = [(x[0], x[1], G.edges[x]['weight']) for x in G.edges]
    edges = pd.DataFrame(weights, columns=['src', 'dst', 'weight'])
    
    # Get the between-sub-community edge weights
    to_reduce = edges[((edges.src.isin(sub0)) & (edges.dst.isin(sub1))) | \
                      ((edges.dst.isin(sub0)) & (edges.src.isin(sub1)))]#.index
    
    for _,(src,trg,weight) in to_reduce.iterrows():
        G.edges[(src,trg)]['weight']=max(weight-delta,CLIP_MIN)

    # Perform the community movement
#     edges.loc[to_reduce, 'weight'] -= delta
#     #edges['weight'] = edges['weight'].clip(0, 1) # Ensure connectivity
    
#     # Re-load graph
#     G_moved = from_pandas_edgelist(edges, source='src', target='dst', 
#                                    edge_attr='weight')
    
#     assert nx.is_connected(G_moved)
    
#     return G_moved

def merge_community(G, subarrays, delta):
    """ Merge two communities by redistributing edge weights between them.

    Parameters
    ----------
        G (networkx.Graph): The input graph.
        subarrays (list of two lists): Two sub-community lists representing the 
            communities to be merged.
        delta (float): The amount of weight to be redistributed.

    Returns
    -------
        G_moved (networkx.Graph): The graph after the community merge with 
            redistributed edge weights.

    """
    # Get the two sub-communities to split
    sub0, sub1 = subarrays
    weights = [(x[0], x[1], G.edges[x]['weight']) for x in G.edges]
    edges = pd.DataFrame(weights, columns=['src', 'dst', 'weight'])
    
    # Get the between-sub-community edge weights
    to_reduce = edges[((edges.src.isin(sub0)) & (edges.dst.isin(sub1))) | \
                      ((edges.dst.isin(sub0)) & (edges.src.isin(sub1)))]#.index
    
    
    for _,(src,trg,weight) in to_reduce.iterrows():
        G.edges[(src,trg)]['weight']=min(weight+delta,CLIP_MAX)

#     # Perform the community movement
#     edges.loc[to_reduce, 'weight'] += delta
#     edges['weight'] = edges['weight'].clip(CLIP_MIN, CLIP_MAX) # Ensure connectivity
    
#     # Re-load graph
#     G_moved = from_pandas_edgelist(edges, source='src', target='dst', 
#                                    edge_attr='weight')
    
#     return G_moved

def fragment_community(G, sub_internal, sub_external, delta):
    """ Fragment a community by redistributing edge weights between internal 
    and external sub-communities.

    Parameters
    ----------
        G (networkx.Graph): The input graph.
        sub_internal (list of tuples): List of tuples representing internal 
            sub-communities to split.
        sub_external (list of tuples): List of tuples representing external 
            sub-communities to merge with internal sub-communities.
        delta (float): The amount of weight to be redistributed.

    Returns
    -------
        G_moved (networkx.Graph): The graph after the community fragmentation 
            with redistributed edge weights.

    """
    # Get the two sub-communities to split
    weights = [(x[0], x[1], G.edges[x]['weight']) for x in G.edges]
    edges = pd.DataFrame(weights, columns=['src', 'dst', 'weight'])
    
    # Get the between-sub-community edge weights
    sub_internal=np.array(sub_internal)
    sub0=sub_internal[:,0]
    sub1=sub_internal[:,1]
    
    # Get edges whose weight is reduced
    to_reduce0 = edges[(edges.src.isin(sub0) & edges.dst.isin(sub1))]
    to_reduce1 = edges[(edges.src.isin(sub1) & edges.dst.isin(sub0))]
    to_reduce = pd.concat([to_reduce0, to_reduce1])#.index

    # Get the sub-community external edge weights
    sub_external=np.array(sub_external)
    sub0=sub_external[:,0]
    sub1=sub_external[:,1]
    
    # Get edges whose weight is increased
    to_augment0 = edges[(edges.src.isin(sub0) & edges.dst.isin(sub1))]
    to_augment1 = edges[(edges.src.isin(sub1) & edges.dst.isin(sub0))]
    to_augment = pd.concat([to_augment0, to_augment1])#.index
    
    
    
    for _,(src,trg,weight) in to_reduce.iterrows():
        G.edges[(src,trg)]['weight']=max(weight-delta,CLIP_MIN)
        
        
        
    for _,(src,trg,weight) in to_augment.iterrows():
        G.edges[(src,trg)]['weight']=min(weight+delta,CLIP_MAX)
    
#     # Perform the community movement
#     edges.loc[to_reduce, 'weight'] -= delta
#     edges.loc[to_augment, 'weight'] += delta
#     edges['weight'] = edges['weight'].clip(CLIP_MIN, CLIP_MAX) # Ensure connectivity
    
#     # Re-load graph
#     G_moved = from_pandas_edgelist(edges, source='src', target='dst', 
#                                    edge_attr='weight')
    
#     return G_moved
    return None


def birth_community(G, sub_internal, sub_external, delta):
    """ Fragment a community by redistributing edge weights between internal 
    and external sub-communities.

    Parameters
    ----------
        G (networkx.Graph): The input graph.
        sub_internal (list of tuples): List of tuples representing internal 
            sub-communities to split.
        sub_external (list of tuples): List of tuples representing external 
            sub-communities to merge with internal sub-communities.
        delta (float): The amount of weight to be redistributed.

    Returns
    -------
        G_moved (networkx.Graph): The graph after the community fragmentation 
            with redistributed edge weights.

    """
    # Get the two sub-communities to split
    weights = [(x[0], x[1], G.edges[x]['weight']) for x in G.edges]
    edges = pd.DataFrame(weights, columns=['src', 'dst', 'weight'])
    
    # Get the between-sub-community edge weights
    sub_internal=np.array(sub_internal)
    sub0=sub_internal[:,0]
    sub1=sub_internal[:,1]
    
    # Get edges whose weight is reduced
    to_augment0 = edges[(edges.src.isin(sub0) & edges.dst.isin(sub1))]
    to_augment1 = edges[(edges.src.isin(sub1) & edges.dst.isin(sub0))]
    to_augment = pd.concat([to_augment0, to_augment1])#.index

    # Get the sub-community external edge weights
    sub_external=np.array(sub_external)
    sub0=sub_external[:,0]
    sub1=sub_external[:,1]
    
    # Get edges whose weight is increased
    to_reduce0 = edges[(edges.src.isin(sub0) & edges.dst.isin(sub1))]
    to_reduce1 = edges[(edges.src.isin(sub1) & edges.dst.isin(sub0))]
    to_reduce = pd.concat([to_reduce0, to_reduce1])#.index
    
    
    for _,(src,trg,weight) in to_reduce.iterrows():
        G.edges[(src,trg)]['weight']=max(weight-delta,CLIP_MIN)
        
        
        
    for _,(src,trg,weight) in to_augment.iterrows():
        G.edges[(src,trg)]['weight']=min(weight+delta,CLIP_MAX)
    
#     # Perform the community movement
#     edges.loc[to_reduce, 'weight'] -= delta
#     edges.loc[to_augment, 'weight'] += delta
#     edges['weight'] = edges['weight'].clip(CLIP_MIN, CLIP_MAX) # Ensure connectivity
    
#     # Re-load graph
#     G_moved = from_pandas_edgelist(edges, source='src', target='dst', 
#                                    edge_attr='weight')
    
#     return G_moved

    return None



def add_nodes(G, node2com):
    # Original number of nodes for nodes ID
    N = max(G.nodes)+1
    n=G.number_of_nodes()
    # Generate new nodes
    batch = np.random.randint(min(1,int(n*BATCH_MIN)), min(2,int(n*BATCH_MAX)))
    new_nodes = [n for n in range(N, N + batch)]

    # Define to which community the new nodes belong to
    comm_sizes = node2com.value_counts('c').to_dict()
    keys, values = [], []
    for k,v in comm_sizes.items():
        keys.append(k[0])
        values.append(v)
    communities = random.choices(keys, weights=values, k=batch)
    
    
    # Temporary store the new nodes and the assigned communities
    nodes_to_add = [i for i in zip(new_nodes, communities)]

    inner_com_nodes = node2com.groupby('c').agg({'n':list}).to_dict()['n']
    inner_com_degree, outer_com_degree = dict(), dict()
    outer_com_nodes = dict()
    # Get internal and external community edges distribution
    for c, inner_nodes in inner_com_nodes.items():
        
        outer_nodes = list(set(G.nodes())-set(inner_nodes))
        outer_com_nodes[c] = outer_nodes
        inner_com_degree[c], outer_com_degree[c] = [], []
        for node in inner_nodes:
            edges_in_community = []
            try:
                for e in G.edges(node):
                    if G.nodes[e[1]]['community'] == c:
                        edges_in_community.append(e)
            except:
                print('Try')
                print('Node',node)
                print("node in G",node in G)
                print("community",inner_nodes)
                
                return node, "err"


            inner_com_degree[c].append(len(edges_in_community))
            edges_not_in_community = [
                e for e in G.edges(node) if G.nodes[e[1]]['community'] != c
            ]
            outer_com_degree[c].append(len(edges_not_in_community))

                
                
    
    # Manage edges
    edges_to_add = []
    for node, c in nodes_to_add:
        # Internal connections:
        # How many edges the new nodes will have inside the assigned community?
        arr, freq = np.unique(inner_com_degree[c], return_counts=True)
        num_inner_edges = random.choices(arr, weights=freq)[0]

        # To which internal nodes the new nodes will be attached?
        inner_dsts = np.random.choice(inner_com_nodes[c], size=num_inner_edges)
        for dst_node in inner_dsts:
            weight = 0.7 + 0.3 * np.random.random()
            edges_to_add.append((node, dst_node, weight))

        # External connections:
        # How many edges the new nodes will have outside the assigned community?
        arr, freq = np.unique(outer_com_degree[c], return_counts=True)
        num_outer_edges = random.choices(arr, weights=freq)[0]

        # To which external nodes the new nodes will be attached?
        if num_outer_edges > 0:
            outer_dsts = np.random.choice(outer_com_nodes[c], size=num_outer_edges)
            for dst_node in outer_dsts:
                weight = 0.3 + 0.3 * np.random.random()
                edges_to_add.append((node, dst_node, weight))
    
    # Finalization
    nodes_to_add = [(x[0], {'community':x[1]}) for x in nodes_to_add]
    edges_to_add = [(x[0], x[1], {'weight':x[2]}) for x in edges_to_add]
    
    return nodes_to_add, edges_to_add


def remove_nodes(G):
    # Generate new nodes
    n=G.number_of_nodes()
    batch = np.random.randint(min(1,int(n*BATCH_MIN_RED)), min(2,int(n*BATCH_MAX_RED)))
    to_remove = np.random.choice(G.nodes, size=batch)

    return to_remove

def intermittence_nodes(G,frac=None):
    if frac==None:
        size=int(G.number_of_nodes()*TURN_OFF_NODES)
    else:
        size=int(G.number_of_nodes()*frac)
        
        
    nodes=np.array(list(dict(G.degree).items()))
    p=1/nodes[:,1]
    p/=sum(p)
    to_remove=list(np.random.choice(nodes[:,0],size=size,replace=False,p=p))
    return to_remove

def update_weights(G_prev, G,ALPHA=0.8):
    new_G = G_prev.copy()
    for e in G.edges:
        if G_prev.has_edge(e[0], e[1]):
            new_G.edges[e]['weight'] = ALPHA*G_prev.edges[e]['weight'] + (1-ALPHA)*G.edges[e]['weight']
        else:
            new_G.add_edge(e[0], e[1], weight = (1-ALPHA)*G.edges[e]['weight'])
    for e in G_prev.edges:
        if not G.has_edge(e[0], e[1]):
            new_G.edges[e]['weight'] = ALPHA*G_prev.edges[e]['weight']

    return new_G

def edges_reshuffling(G,community):
    
    weights = [(x[0], x[1], G.edges[x]['weight']) for x in G.edges]
    edges = pd.DataFrame(weights, columns=['src', 'dst', 'weight'])
    
    internal_edges=edges.query('(src in @community) and (dst in @community)').reset_index(drop=True)
    
    external_edges= pd.concat([edges.query('(src in @community) and (not dst in @community)'),
                               edges.query('(dst in @community) and (not src in @community)')
                              ],ignore_index=True)#edges from the community to external nodes
    other_edges=edges.query('(not src in @community) and (not dst in @community)').reset_index(drop=True)
    #reshuffle internal edges
    weighted_internal_degree=dict(G.degree(community,weight="weight"))
    unweighted_internal_degree=dict(G.degree(community))
    reshuffled_internal_edges=np.random.randint(0,len(internal_edges),size=int(SHUFFLE_P*len(internal_edges)))# represent the link which will be reshuffled
    for i in reshuffled_internal_edges:
        
        w=internal_edges.loc[i,'weight']
        src=internal_edges.loc[i,'src']
        dst=internal_edges.loc[i,'dst']
        
        if unweighted_internal_degree[dst]>1:
            dest_n=preferential_attachment(weighted_internal_degree,src)[0]
            internal_edges.loc[i,'dst']=dest_n
            weighted_internal_degree[dst]-=w
            unweighted_internal_degree[dst]-=1
            weighted_internal_degree[dest_n]+=w
            unweighted_internal_degree[dest_n]+=1
        elif unweighted_internal_degree[src]>1:
            dest_n=preferential_attachment(weighted_internal_degree,dst)[0]
            internal_edges.loc[i,'src']=dest_n
            weighted_internal_degree[src]-=w
            unweighted_internal_degree[src]-=1
            weighted_internal_degree[dest_n]+=w
            unweighted_internal_degree[dest_n]+=1
            
            
            
    #reshuffle external edges
    weighted_external_degree=dict(G.degree(set(G.nodes)-set(community),weight="weight"))
    unweighted_external_degree=dict(G.degree(set(G.nodes)-set(community)))
    
    reshuffled_external_edges=np.random.randint(0,len(external_edges),size=int(SHUFFLE_P*len(external_edges)))# represent the link which will be reshuffled
    for i in reshuffled_external_edges:
        dest_n=preferential_attachment(weighted_external_degree)[0]
        w=external_edges.loc[i,'weight']
        src=external_edges.loc[i,'src']
        dst=external_edges.loc[i,'dst']
        if src in community:
            if unweighted_external_degree[dst]>1:
                weighted_external_degree[dest_n]+=w
                unweighted_external_degree[dest_n]+=1
                
                weighted_external_degree[dst]-=w
                unweighted_external_degree[dst]-=1
                
                external_edges.loc[i,'dst']=dest_n
        
        if dst in community:
            if unweighted_external_degree[src]>1:
                weighted_external_degree[dest_n]+=w
                unweighted_external_degree[dest_n]+=1
                
                unweighted_external_degree[src]-=1
                weighted_external_degree[src]-=w
                
                external_edges.loc[i,'src']=dest_n
    edges=pd.concat([internal_edges,external_edges,other_edges],ignore_index=True)
    
    G_moved = from_pandas_edgelist(edges, source='src', target='dst', 
                                   edge_attr='weight')
    
    
    
    return G_moved



def switch(G,inactive_nodes,delta=0.005,T_off=10):
    size=max(1,int(G.number_of_nodes()*delta))
    degree=G.degree
    nodes=np.array([(node,degree[node]) for node in G.nodes if not node in inactive_nodes])
    p=1/nodes[:,1]
    p/=sum(p)
    to_remove=list(np.random.choice(nodes[:,0],size=size,replace=False,p=p))
    
    nodes=np.array([(node, T_off-T) for node, T in inactive_nodes.items()])
    p=np.exp(-nodes[:,1])
    p/=sum(p)
    
    to_add=np.random.choice(nodes[:,0],size=size,replace=False,p=p)
    to_remove=list(set(nodes[:,0])-set(to_add))+to_remove
    inactive_nodes={node: inactive_nodes[node]+1 if node in inactive_nodes else 1 for node in to_remove}
    
    return inactive_nodes

def break_up_communities(G):
    # Generate new nodes
    n=G.number_of_nodes()
    batch = np.random.randint(min(1,int(n*BATCH_MIN)), min(2,int(n*BATCH_MAX)))
    edges=dict(zip(np.arange(G.number_of_edges()),list(G.edges(data='weight'))))
    to_remove = np.random.randint(0,len(edges)-1, size=(batch,2))
    to_remove=[(edges[key][0],edges[key][1]) for key in to_remove[:,0]]
    to_add=[(edges[key_source][0],edges[key_target][1],{'weight':edges[key_source][2]}) for key_source,key_target in to_remove]
    return to_remove,to_add


def remove_edges(G):
    # Generate new nodes
    n=G.number_of_nodes()
    batch = np.random.randint(min(1,int(n*BATCH_MIN)), min(2,int(n*BATCH_MAX)))
    edges=dict(zip(np.arange(G.number_of_edges()),list(G.edges)))
    to_remove = np.random.randint(0,len(edges)-1, size=batch)
    to_remove=[edges[key] for key in to_remove]
    return to_remove

