import networkx as nx

OVERLAP_TH = .6
MOD_TH = -1e-5

def compute_jaccard(A, B):
    # Compute the jaccard index between community A of snapshot t-1 and 
    # community B of snapshot t
    num = set(A).intersection(set(B))
    denom = set(A).union(set(B))

    return len(num)/len(denom)

def map_communities(previous, next):
    # Manage the indices of the t-1 vs. t communities
    commA, commB, mapping = dict(), dict(), dict()
    for nA, cA in previous.items():
        if cA not in commA: commA[cA] = [nA]
        else: commA[cA].append(nA)
    for nB, cB in next.items():
        if cB not in commB: commB[cB] = [nB]
        else: commB[cB].append(nB)

    total_comms_b = len(commA)
    for cB in commB: # For each community of snapshot t
        mapped = False
        for cA in commA: # For each community of snapshot t-1
            # Get the community members overlap
            overlap = compute_jaccard(commA[cA], commB[cB])
            # If the overlap is greater than a threshold, perform the mapping
            if overlap >= OVERLAP_TH:
                mapping[cB] = cA
                mapped = True
                break
        # If any meaningful mapping is found, assign the community of 
        # snapshot t to a new community
        if not mapped:
            mapping[cB] = total_comms_b
            total_comms_b += 1

    return mapping

def replace_communities(comm, mapping):
    # Perform the mapping between communities of snapshot t and t-1
    return {n: int(mapping[c]) for n,c in comm.items()}

def modularity_comm(partition, graph, weight='weight'):
    # Compute the modularity of each community
    if graph.is_directed():
        raise TypeError("Bad graph type, use only non directed graph")

    inc = dict([])
    deg = dict([])
    links = graph.size(weight=weight)

    if links == 0:
        raise ValueError("A graph without link has an undefined modularity")
    if len(partition)==0:
        raise TypeError(f"partition is empty, while the graph has {graph.number_of_nodes()} nodes")
    for node in graph:
        # try:
        com = partition[node]
        deg[com] = deg.get(com, 0.) + graph.degree(node, weight=weight)
        for neighbor, datas in graph[node].items():
            edge_weight = datas.get(weight, 1)
            if partition[neighbor] == com:
                if neighbor == node:
                    inc[com] = inc.get(com, 0.) + float(edge_weight)
                else:
                    inc[com] = inc.get(com, 0.) + float(edge_weight) / 2.

    com_res = dict([])
    for com in set(partition.values()):
        com_res[com] = (inc.get(com, 0.) / links) - \
               (deg.get(com, 0.) / (2. * links)) ** 2        
    return com_res

def get_bad_communities(prev_modcom, modcom, mod_th=MOD_TH):
    # Retrieve the communities whose modularity gain after the initialization
    # is decreased
    bad_comms = []
    for c, mod in modcom.items():
        # Compute the modularity gain
        if c not in prev_modcom:
            bad_comms.append(c)
        else:
            gain_modcom = mod-prev_modcom[c]
            if gain_modcom < mod_th:
                bad_comms.append(c)

    return bad_comms

def iGMA_init(G, G_prev, previous_comm):
    new_init = dict()
    new_comms_cnt = max(previous_comm.values())
    # First initialization
    for node in G.nodes:
        if node in G_prev.nodes():
            
            prev_c = previous_comm[node]
            new_init[node] = prev_c
        else:
            new_comms_cnt+=1
            prev_c = new_comms_cnt
            new_init[node] = prev_c
    
    return new_init

def eGMA_init(G, G_prev, previous_comm):
    new_init = dict()
    new_comms_cnt = max(previous_comm.values())
    # First initialization. Same as iGMA
    for node in G.nodes:
        if node in G_prev.nodes():
            prev_c = previous_comm[node]
            new_init[node] = prev_c
        else:
            new_comms_cnt+=1
            prev_c = new_comms_cnt
            new_init[node] = prev_c

    G1 = G.copy()
    nx.set_node_attributes(G1, new_init, name='community')

    # Get bad communities wrt. modularity gain
    prev_modcom = modularity_comm(previous_comm, G_prev, weight='weight')
    modcom = modularity_comm(new_init, G, weight='weight')
    badcom = get_bad_communities(prev_modcom, modcom, mod_th=MOD_TH)

    # Finalize initialization
    if len(badcom)>0:
        new_com_cnt = max(previous_comm.values())+1
        new_init = dict()
        for node in G1.nodes:
            node_com = G1.nodes[node]['community']
            if node_com in badcom:
                new_com_cnt += 1
                node_com = new_com_cnt
            new_init[node] = node_com
    return new_init

import os    
def _weighted_majority_voting(new_nodes, G,previous_comm,rec=0):
    neighbors = {}
    for node in new_nodes:
        for neigh,data in G[node].items():
            if neigh in new_nodes:
                pass
            else:
                if neigh in neighbors:
                    neighbors[neigh].append((node,data['weight']))
                else:
                    neighbors[neigh]=[(node,data['weight'])]
    votes = {}

    for neigh in neighbors:
        neighbor_community = previous_comm[neigh]
        for node,weight in neighbors[neigh]:
            if node in votes:
                if neighbor_community in votes[node]: votes[node][neighbor_community] += weight
                else: votes[node][neighbor_community] = weight
            else:
                votes[node]={}
                votes[node][neighbor_community] = weight
    new_init={}
    not_initialized=[]
    for node in new_nodes:
        if node in votes:
            new_init[node]= max(votes[node], key=votes[node].get) 
        else: 
            not_initialized.append(node)

    if len(not_initialized)>0:
       
        if rec<4:    
            second_new_init=_weighted_majority_voting(set(not_initialized), G,new_init,rec=rec+1)
        else:
            second_new_init={}
            for node in not_initialized:
                # for neigh in G.neighbors(node): 
                #     assert neigh in not_initialized
                second_new_init[node]=node
        new_init.update(second_new_init)
    
    # Assign the winning community to the node
    return new_init


def NeGMA_init(G, G_prev, previous_comm,mod_th=MOD_TH):
    new_init = dict()
    # First initialization. Same as iGMA

    old_nodes=set(G.nodes).intersection(G_prev.nodes)
    new_nodes=set(G.nodes)-set(G_prev.nodes)
    
    for node in old_nodes:
        prev_c =previous_comm[node]
        new_init[node] = prev_c


    new_node_init=_weighted_majority_voting(new_nodes, G,previous_comm)
    new_init.update(new_node_init)

    G1 = G.copy()
    nx.set_node_attributes(G1, new_init, name='community')

    # Get bad communities wrt. modularity gain
    prev_modcom = modularity_comm(previous_comm, G_prev, weight='weight')
    modcom = modularity_comm(new_init, G, weight='weight')
    badcom = get_bad_communities(prev_modcom, modcom, mod_th=mod_th)

    if len(badcom)>0:
        new_com_cnt = max(previous_comm.values())+1
        new_init = dict()
        for node in G1.nodes:
            node_com = G1.nodes[node]['community']
            if node_com in badcom:
                new_com_cnt += 1
                node_com = new_com_cnt
            new_init[node] = node_com
    return new_init





def MutiTh_NeGMA_init(G, G_prev, previous_comm,mod_ths=[MOD_TH]):
    new_init = dict()
    # First initialization. Same as iGMA

    old_nodes=set(G.nodes).intersection(G_prev.nodes)
    new_nodes=set(G.nodes)-set(G_prev.nodes)
    
    for node in old_nodes:
        prev_c =previous_comm[node]
        new_init[node] = prev_c


    new_node_init=_weighted_majority_voting(new_nodes, G,previous_comm)
    new_init.update(new_node_init)

    # G1 = G.copy()
    # nx.set_node_attributes(G1, new_init, name='community')

    # Get bad communities wrt. modularity gain for each threshold
    prev_modcom = modularity_comm(previous_comm, G_prev, weight='weight')
    modcom = modularity_comm(new_init, G, weight='weight')
    badcoms={}
    new_inits={}
    for th in mod_ths:
        badcoms[th] = get_bad_communities(prev_modcom, modcom, mod_th=th)
        new_inits[th]=new_init.copy()
        # Finalize initialization
        if len(badcoms[th])>0:          
            new_com_cnt = max(previous_comm.values())
            for node in badcoms[th]:
                new_com_cnt += 1
                node_com = new_com_cnt
                new_inits[th][node] = node_com
    # assert len(set(G1.nodes())-set(new_init.keys()))==0
    return new_inits