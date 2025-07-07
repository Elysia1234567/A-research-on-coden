import pandas as pd
import networkx as nx
from tqdm import tqdm
import numpy as np

from ..graph import generate_weighted_LFR_graph

from ..transformations.preferential_attachment import (
    preferential_attachment_fragment,
    preferential_attachment_merging
)
from ..transformations import (
    split_community, 
    fragment_community, 
    birth_community,
    merge_community,
    add_nodes,
    remove_nodes,
    intermittence_nodes,
    update_weights,
    edges_reshuffling,
    switch,
    break_up_communities,
    remove_edges
    
)



class GraphMovementSimulation():
    def __init__(self, n=1000, t1=3, t2=1.5, mu=.2, seed=10, gname=None):
        # Generate base graph
        self.seed = seed
        self.G_base,comm = generate_weighted_LFR_graph(nodes=n, 
                                                  tau1=t1, 
                                                  tau2=t2, 
                                                  mu=mu,
                                                  seed=self.seed)
        
        # Handle isolated nodes and self loops.
        self.G_base.remove_edges_from(nx.selfloop_edges(self.G_base))
        self.G_base.remove_nodes_from(list(nx.isolates(self.G_base))) 
        
        self.gname = gname
        self.subarrays = []

        # Retrieve community info
        self.orig_comm = {x:comm[x] for x in self.G_base.nodes}
        self.node2com = pd.DataFrame(self.orig_comm.items(), columns=['n', 'c'])
        # nx.set_node_attributes(self.G_base, self.orig_comm, name='community')

    
    def _setup_fragment(self, G, n_splits):
        """ Set up the fragmentation process for the Louvain algorithm.
        Prepare the fragmentation by dividing the communities into subarrays 
        and performing preferential attachment fragmentation.

        Parameters
        ----------
            n_splits (int): The number of splits to perform on the communities.

        """
        self.internal_subarrays = []

        # Collect communities and nodes  
        comms = self.node2com.groupby('c')['n'].apply(list).to_dict()      
        other_comms = {c:comms[c] for c in range(n_splits, len(comms))}
        
        true_init  = self.orig_comm.copy()
        true_final = self.orig_comm.copy()

        for k in range(n_splits):
            (
                G, internal_edges_to_loosen , new_edges, lookup
            ) = preferential_attachment_fragment(
                G,
                comms[k],
                other_comms,
                weight=0.001,
                weight_internal=1
                
            )
            
            # Update the nodes arrays
            self.internal_subarrays+=internal_edges_to_loosen
            self.subarrays+=new_edges
            
            # Update the Ground truth
            for n in comms[k]:
                true_final[n]=lookup[n]

        return G, true_init, true_final
    
    def _setup_birth(self, G, n_splits):
        """ Set up the birth process for the Louvain algorithm.
        Prepare the birth by dividing the communities into subarrays 
        and performing preferential attachment fragmentation.

        Parameters
        ----------
            n_splits (int): The number of splits to perform on the communities.

        """
        self.internal_subarrays = []

        # Collect communities and nodes  
        comms = self.node2com.groupby('c')['n'].apply(list).to_dict()      
        other_comms = {c:comms[c] for c in range(n_splits, len(comms))}
        
        true_init  = self.orig_comm.copy()
        true_final = self.orig_comm.copy()

        for k in range(n_splits):
            (
                G, internal_edges_to_tighten , new_edges, lookup
            ) = preferential_attachment_fragment(
                G,
                comms[k],
                other_comms,
                weight=1,
                weight_internal=0.001
            )
            
            # Update the nodes arrays
            self.internal_subarrays+=internal_edges_to_tighten
            self.subarrays+=new_edges
            
            # Update the Ground truth
            for n in comms[k]:
                true_init[n]=lookup[n]

        return G, true_init, true_final


    def _setup_merge(self, G, n_splits):
        # Collect communities and nodes
        c_counts = self.node2com.value_counts('c').reset_index()    

        true_init  = self.orig_comm.copy()   
        true_final = self.orig_comm.copy()
        
        for k in range(n_splits):
            # Get the two communities to merge.
            comm_to_merge = list(c_counts.iloc[[k, -(k+1)]].c)
            local_subarr = [
                self.node2com[self.node2com.c.isin([comm_to_merge[0]])].n.values,
                self.node2com[self.node2com.c.isin([comm_to_merge[1]])].n.values
            ]
            self.subarrays += local_subarr

            # Merge the communities with preferential attachment
            G = preferential_attachment_merging(
                    G, 
                    local_subarr[0], 
                    local_subarr[1],
                    weight=0.001
                
            )
            
            # Update the Ground truth
            for n in local_subarr[1]:
                true_final[n]=comm_to_merge[0]
        
        return G, true_init, true_final
    
    
    
    def _setup_split(self, G, n_splits):
        # Collect communities and nodes
        c_counts = self.node2com.value_counts('c').reset_index()       

        true_init  = self.orig_comm.copy() 
        true_final = self.orig_comm.copy()
        
        for k in range(n_splits):
            # Get the two communities to merge.
            comm_to_merge = list(c_counts.iloc[[k, -(k+1)]].c)
            local_subarr = [
                self.node2com[self.node2com.c.isin([comm_to_merge[0]])].n.values,
                self.node2com[self.node2com.c.isin([comm_to_merge[1]])].n.values
            ]
            self.subarrays += local_subarr

            # Merge the communities with preferential attachment
            G = preferential_attachment_merging(
                    G, 
                    local_subarr[0], 
                    local_subarr[1],
                    weight=1
            )
            
            # Update the Ground truth
            for n in local_subarr[1]:
                true_init[n]=comm_to_merge[0]

        # Reinitialize comm info
        self.orig_comm = true_init.copy()
        self.node2com = pd.DataFrame(self.orig_comm.items(), columns=['n', 'c'])
        
        return G, true_init, true_final
        


    def setup_transformation(self, transformation, n_splits, save=False):
        # Copy original graph
        G = self.G_base.copy()
        
        # Set up the different transformations
        if transformation == 'fragment':
            G, true_init, true_final = self._setup_fragment(G, n_splits)
        elif transformation == 'birth':
            G, true_init, true_final = self._setup_birth(G, n_splits)
        elif transformation == 'merge':
            G, true_init, true_final = self._setup_merge(G, n_splits)
        elif transformation == 'split':
            G, true_init, true_final = self._setup_split(G, n_splits)
        elif transformation == 'add_nodes':
            true_init = self.orig_comm.copy()
            true_final = self.orig_comm.copy()
        elif transformation == 'remove_nodes':
            true_init = self.orig_comm.copy()
            true_final = self.orig_comm.copy()
        elif transformation == 'remove_edges':
            true_init = self.orig_comm.copy()
            true_final = self.orig_comm.copy()
        elif transformation == 'intermittence_nodes':
            true_init = self.orig_comm.copy()
            true_final = self.orig_comm.copy()
        elif transformation == 'shuffle_edges':
            true_init = self.orig_comm.copy()
            true_final = self.orig_comm.copy()
        # elif transformation == 'intermittence_edges':
        #     true_init = self.orig_comm.copy()
        #     true_final = self.orig_comm.copy()
        elif transformation == 'switch':
            size=max(1,int(G.number_of_nodes()*0.05))
            degree=G.degree
            nodes=np.array([(node,degree[node]) for node in G.nodes])
            p=1/nodes[:,1]
            p/=sum(p)
            starting_missing_nodes=list(np.random.choice(nodes[:,0],size=size,replace=False,p=p))
            G.remove_nodes_from(starting_missing_nodes)
            G.remove_nodes_from(list(nx.isolates(G))) 
            true_init = self.orig_comm.copy()
            true_final = self.orig_comm.copy()
            self.inactive_nodes={node:1 for node in set(self.G_base.nodes)-set(G.nodes)}
        elif transformation == 'break_up_communities':
            true_init = self.orig_comm.copy()
            true_final = self.orig_comm.copy()

        

        # Get the initial Ground truth before the transformation
        self.y_true_init = true_init
        # Get the final Ground truth after the transformation
        self.y_true_final = true_final
        # print(transformation,len(set(self.y_true_init.values())),len(set(self.y_true_final.values())))     
        
        return G
    
    def run(self, G,N_Graph=None, timesteps=300, delta=.05, transformation='split', start_trans=25,stop_trans=125,instant=False):
        info=self.gname.split("_")
        if N_Graph==None:
            N_Graph=int(info[-1])
        mu=info[-2]
        if N_Graph==0:
            pb=True
        else: pb=False
        if pb: 
            progress_bar = tqdm(total=timesteps, 
                                desc=f'Processing {self.gname}')        
        
        for i in range(timesteps): 
            
            if pb: progress_bar.update(1)

            ###################################################################
            # RUN TRASNFORMATIONS
            ###################################################################
            if i >= start_trans and i < stop_trans:            
            # Run one fragment iteration
                if transformation=='fragment':
                    fragment_community(
                        G, 
                        self.internal_subarrays,
                        self.subarrays, 
                        delta)

                elif transformation=='birth':
                    birth_community(
                        G, 
                        self.internal_subarrays,
                        self.subarrays, 
                        delta)


                elif transformation in ['split', 'merge']:
                    for j in range(len(self.subarrays)):    
                        if j%2 == 0:
                            subarrays = self.subarrays[j], self.subarrays[j+1]
                            # Run one split iteration
                            if transformation=='split':
                                split_community(G, subarrays, delta)
                            # Run one merge iteration
                            if transformation=='merge':
                                merge_community(G, subarrays, delta)

                elif transformation == 'add_nodes':
                    nodes_to_add, edges_to_add = add_nodes(G, self.node2com)
                    # Update the graph
                    G.add_nodes_from(nodes_to_add, attr='community')
                    G.add_edges_from(edges_to_add, attr='weight') 
                    G.remove_nodes_from(list(nx.isolates(G)))
                    self.y_true_final.update({
                        x[0]:x[1]['community'] for x in nodes_to_add if x[0] in G.nodes()})

                    # Retrieve community info
                    self.orig_comm.update({x[0]:x[1]['community'] for x in nodes_to_add if x[0] in G.nodes()})
                    self.node2com = pd.DataFrame(self.orig_comm.items(), columns=['n', 'c'])


                elif transformation == 'remove_nodes':
                    nodes_to_remove = remove_nodes(G)
                    G.remove_nodes_from(nodes_to_remove)
                    G.remove_nodes_from(list(nx.isolates(G)))

                    self.orig_comm = {x:self.orig_comm[x] for x in G.nodes}
                    self.node2com = pd.DataFrame(self.orig_comm.items(), columns=['n', 'c'])

                elif transformation == 'remove_edges':
                    edges_to_remove = remove_nodes(G)
                    G.remove_edges_from(edges_to_remove)
                    G.remove_nodes_from(list(nx.isolates(G)))

                    self.orig_comm = {x:self.orig_comm[x] for x in G.nodes}
                    self.node2com = pd.DataFrame(self.orig_comm.items(), columns=['n', 'c'])

                elif transformation == 'intermittence_nodes':
                    G = self.G_base.copy()
                    nodes_to_remove = intermittence_nodes(G)
                    G.remove_nodes_from(nodes_to_remove)
                    G.remove_nodes_from(list(nx.isolates(G)))


                elif transformation == 'shuffle_edges':
                    for j in range(len(self.subarrays)):
                        G=edges_reshuffling(G, self.subarrays[j])

                # elif transformation == 'intermittence_edges':
                #     G = self.G_base.copy()
                #     edges_to_remove = intermittence_edges(G)
                #     G.remove_nodes_from(edges_to_remove)
                #     G.remove_nodes_from(list(nx.isolates(G)))    

                elif transformation == 'switch':


                    G = self.G_base.copy()
                    self.inactive_nodes = switch(G,self.inactive_nodes)
                    nodes_to_remove=list(self.inactive_nodes.keys())
                    G.remove_nodes_from(nodes_to_remove)
                    G.remove_nodes_from(list(nx.isolates(G))) 

                elif transformation == 'break_up_communities':
                    edges_to_remove,edges_to_add = break_up_communities(G)
                    G.remove_edges_from(edges_to_remove)
                    G.add_edges_from(edges_to_add)
                    G.remove_nodes_from(list(nx.isolates(G)))

                    self.orig_comm = {x:self.orig_comm[x] for x in G.nodes}
                    self.node2com = pd.DataFrame(self.orig_comm.items(), columns=['n', 'c'])

            
            if instant:
                nx.write_weighted_edgelist(G, f'../results/graphs/{mu}/{transformation}/G{N_Graph:02}_INST/T{i:03}.txt.gz')
            else:
                nx.write_weighted_edgelist(G, f'../results/graphs/{mu}/{transformation}/G{N_Graph:02}/T{i:03}.txt.gz')
        
        
        return N_Graph,{'y_true_init':self.y_true_init,'y_true_final':self.y_true_final}
