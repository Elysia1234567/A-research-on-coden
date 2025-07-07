import numpy as np

def preferential_attachment_fragment(G, target, other_comms, weight=0.01,weight_internal=1):
    ''' Fragment a community by adding through preferential attachments n 
    weighted links (where n is their starting degree) from each node 
    of target community to a different one.
    Note: each node is linked to a unique different community, but different 
    nodes can join different communities.

    Parameters
    ----------
        G (networkx.Graph): The graph object before transformation.
        target (list): List of nodes belonging to the target community.
        other_comms (dict): A dictionary mapping community identifiers to lists 
            of nodes representing the communities other than the target community.
        weight (float): Initial weight referred to the new attached links.

    Returns
    -------
        G (networkx.Graph): The graph object after the fragmentation.
        internal_edges_to_loosen (list): List of internal links within the 
            target community that need to be loosened.
        new_edges (list): List of newly added edges connecting nodes from the 
            target community to nodes in different communities.
        lookup (dict): A dictionary mapping nodes from the target community to 
            their corresponding destination communities.

    '''
    # Get the degree of the nodes
    degree=dict(G.degree())
    degree_c_target={node:degree[node] for node in target}
    degree_c_other={c:{
        n:degree[n] for n in other_comms[c]
    } for c in other_comms}

    new_edges, removed_edges = [], []
    
    # Randomly select destination communities
    new_comm = np.random.choice(list(other_comms.keys()), size=len(target))
    # Perform the target node -> dst community mapping
    lookup = dict(zip(target, new_comm))
    
    #edges_to_loosen=[]
    # Perform preferential attachment
    for src_n in target:
        
        dst_c = lookup[src_n] # Destination community of the considered node
        
        # Get the probabilities
        p=np.array(list(degree_c_other[dst_c].values()))
        # Select the destination nodes
        dsts = np.random.choice(other_comms[dst_c], 
                                size=degree_c_target[src_n], 
                                p=p/p.sum()
        )
        for dst_n in dsts:
            new_edges.append((src_n, dst_n, {'weight':weight}))
            degree_c_other[dst_c][dst_n] += 1
            
            if G.has_edge(src_n, dst_n) or G.has_edge(dst_n, src_n):
                degree_c_other[dst_c][dst_n] -= 1
                removed_edges.append((src_n, dst_n))
                removed_edges.append((dst_n, src_n))
        
    G.remove_edges_from(removed_edges) # Remove the existing edges

    internal_link_to_loosen=list(G.subgraph(target).edges)
    for edge in internal_link_to_loosen:
        G.edges[edge]['weight']=weight_internal
        
    G.add_edges_from(new_edges, attr='weight') # Add new edges

    return G, internal_link_to_loosen, new_edges, lookup

def preferential_attachment_merging(G, c1, c2, weight=1):
    ''' Merge two different communities, C1 and C2, by establishing preferential 
    attachment links. For each node in C1, a weighted link is added to a node in 
    C2. Similarly, for each node in C2, a link with the same weight is created 
    to a node in C1. 

    Parameters
    ----------
        G (networkx.Graph): The graph object before the merging operation.
        c1 (list): List of nodes representing community C1.
        c2 (list): List of nodes representing community C2.
        weight (float): The weight assigned to the newly added links.

    Returns
    -------
        G (networkx.Graph): The graph object after the merging operation.
    
    '''
    c1_to_c2_edges, c2_to_c1_edges = [], []
    
    # Get the degree of the two communities
    degree_c1=dict(G.degree(c1, weight='weight'))
    degree_c2=dict(G.degree(c2, weight='weight'))

    # Run preferential attachment from C1 to C2
    for src_n in c1:
        # Get the probabilities
        p = np.array(list(degree_c2.values()))
        # Select the destination node
        dst_n = np.random.choice(c2, size=1, p=p/p.sum())[0]
        
        # Assign the new weights
        if not G.has_edge(src_n, dst_n):
            degree_c2[dst_n] += weight
            degree_c1[src_n] += weight
            c1_to_c2_edges.append((src_n, dst_n, {'weight':weight}))

    # Add new edges from C1 to C2
    G.add_edges_from(c1_to_c2_edges, attr='weight')

    # Run preferential attachment from C1 to C2
    for src_n in c2:
        # Get the probabilities
        p=np.array(list(degree_c1.values()))
        # Select the destination node
        dst_n=np.random.choice(c1, size=2, p=p/p.sum())[0]

        if not G.has_edge(src_n, dst_n):
            degree_c1[dst_n] += weight
            degree_c2[src_n] += weight
            c2_to_c1_edges.append((src_n, dst_n, {'weight':weight}))

    # Add new edges from C2 to C1       
    G.add_edges_from(c2_to_c1_edges, attr='weight')

    return G

