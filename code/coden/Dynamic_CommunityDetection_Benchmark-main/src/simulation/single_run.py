import pandas as pd
import networkx as nx
from tqdm import tqdm
import numpy as np
import time

from ..graph import generate_weighted_LFR_graph
from ..communities import (
    Leiden,
    Louvain, 
    iGMA_init,
    eGMA_init,
    NeGMA_init
)
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

def _renumber(dictionary):
    """Renumber the values of the dictionary from 0 to n
    """
    values = set(dictionary.values())
    target = set(range(len(values)))

    if values == target:
        # no renumbering necessary
        ret = dictionary.copy()
    else:
        # add the values that won't be renumbered
        renumbering = dict(zip(target.intersection(values),
                               target.intersection(values)))
        # add the values that will be renumbered
        renumbering.update(dict(zip(values.difference(target),
                                    target.difference(values))))
        ret = {k: renumbering[v] for k, v in dictionary.items()}

    return ret


class GraphMovementSimulation():
    def __init__(self, n=1000, t1=3, t2=1.5, mu=.2, seed=10, gname=None):
        # Generate base graph
        self.G_base,comm = generate_weighted_LFR_graph(nodes=n, 
                                                  tau1=t1, 
                                                  tau2=t2, 
                                                  mu=mu,
                                                  seed=seed)
        
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
                weight=0.0001,
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
            starting_missing_nodes=np.random.randint(0,high=G.number_of_nodes(),size=int(0.95*G.number_of_nodes()))
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
        # Save the original graph
        if save:
            # Set the graph attributes
            G_copy=G.copy()
            nx.set_node_attributes(G_copy, true_init, name='community')
            nx.write_gexf(G_copy, f'../results/graphs/{self.gname}_original.gexf')
    
        return G
    
    def run(self, G, seed, timesteps=300, delta=.05, transformation='split', save=False,start_trans=25,stop_trans=125,report=False):
        if (self.gname.split("_")[-1]=="0" and save) or report==True:
            pb=True
        else: pb=False
        if pb: 
            progress_bar = tqdm(total=timesteps, 
                                desc=f'Processing {self.gname}')
        metrics = []
        partitions = []
        t={}
        
        # Init algorithms
        louvain_base = Louvain(seed)
        louvain = Louvain(seed)
        louvain_i = Louvain(seed)
        louvain_e = Louvain(seed)
        louvain_ne = Louvain(seed)
        
        aleiden = Leiden(seed)
        leiden = Leiden(seed)
        leiden_i = Leiden(seed)
        leiden_ne = Leiden(seed)
        
        
        t['louvain_base'] = 0
        t['louvain'] = 0
        t['louvain_i'] = 0
        t['louvain_e'] = 0
        t['louvain_ne'] = 0
        
        t['aleiden'] = 0
        t['leiden'] = 0
        t['leiden_i'] = 0
        t['leiden_ne'] = 0
        
        
        for i in range(timesteps):   

            # Update predictions and communities    
            if i==0: 
                previous_pred_base = self.y_true_init.copy()
                
                # GMA
                previous_pred = self.y_true_init.copy()
                
                # iGMA
                previous_pred_i = self.y_true_init.copy()
                
                # eGMA
                previous_pred_e = self.y_true_init.copy()
                
                # NeGMA
                previous_pred_ne = self.y_true_init.copy()
                

                # aLeiden
                previous_pred_alei = self.y_true_init.copy()
                
                # Leiden
                previous_pred_lei = self.y_true_init.copy()
                # iLeiden
                
                previous_pred_ilei = self.y_true_init.copy()

                # NeLei
                previous_pred_nelei = self.y_true_init.copy()
                
                
            else: 

                # SoA
                previous_pred_base = y_pred_base.copy()
                

                # GMA
                previous_pred = y_pred.copy()
                

                # iGMA
                previous_pred_i = y_pred_i.copy()  
                

                # eGMA
                previous_pred_e = y_pred_e.copy()  
                

                # NeGMA
                previous_pred_ne = y_pred_ne.copy()                 
                

                # aLeiden
                previous_pred_alei = y_pred_alei.copy()
                

                # Leiden
                previous_pred_lei = y_pred_lei.copy()
                
                
                # iLeiden
                previous_pred_ilei = y_pred_lei.copy()
                

                # NeLeiden
                previous_pred_nelei = y_pred_nelei.copy()  
              
                
                         
            # Get the previous graph
            G_prev = G.copy()
            if i == 0:
                G_prev_base = G.copy()
            else:
                G_prev_base = _G.copy()
            
            

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
                    if edges_to_add=='err':
                        print(self.gname,'seed',seed,'step',i)
                        raise "Error"
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

            # For alpha-based framework
            _G = update_weights(G_prev_base, G)

            ###################################################################
            # DETECT COMMUNITIES
            ###################################################################
            
            
            t0=time.time()
            # Run GMA
            zero_init = None #{x:i for i,x in enumerate(G.nodes)}
            y_pred = louvain.run(G, zero_init, previous_pred)
            ari = louvain.get_metrics(previous_pred, y_pred)
            ari_fin = louvain.get_metrics (self.y_true_final, y_pred)
            ari_init = louvain.get_metrics (self.y_true_init, y_pred)
            communities={}
            for node in y_pred:
                if not y_pred[node] in communities: communities[y_pred[node]]=list()
                communities[y_pred[node]].append(node)
            modularity=nx.community.modularity(G, list(communities.values()))
            t['louvain']+=time.time()-t0
            
            
            # Run SoA
            t0=time.time()
            # For alpha-based framework
            _G = update_weights(G_prev_base, G)
            t_memory=time.time()-t0 #time for updating the dictionary
            zero_init = None
            y_pred_base = louvain_base.run(_G, zero_init, previous_pred_base)
            # Here nodes are always active. Thus retrieve the correct nodes from 
            # GMA application
            y_pred_base = {k:v for k,v in y_pred_base.items() if k in y_pred}
            ari_base = louvain_base.get_metrics(previous_pred_base, y_pred_base)
            ari_fin_base = louvain_base.get_metrics (self.y_true_final, y_pred_base)
            ari_init_base = louvain_base.get_metrics (self.y_true_init, y_pred_base)
            communities={}
            for node in y_pred_base:
                if not y_pred_base[node] in communities: communities[y_pred_base[node]]=list()
                communities[y_pred_base[node]].append(node)
            modularity_base=nx.community.modularity(G, list(communities.values()))
            t['louvain_base']+=time.time()-t0
            
            
            # Run iGMA
            t0=time.time()
            if i==0:
                custom_init=None
            else:
                custom_init = iGMA_init(G, G_prev, previous_pred_i)
                custom_init = _renumber(custom_init)
            y_pred_i = louvain_i.run(G, custom_init, previous_pred_i)
            ari_i = louvain_i.get_metrics(previous_pred_i, y_pred_i)
            ari_fin_i = louvain_i.get_metrics (self.y_true_final, y_pred_i)
            ari_init_i = louvain_i.get_metrics (self.y_true_init, y_pred_i)
            communities={}
            for node in y_pred_i:
                if not y_pred_i[node] in communities: communities[y_pred_i[node]]=list()
                communities[y_pred_i[node]].append(node)
            modularity_i=nx.community.modularity(G, list(communities.values()))
            t['louvain_i']+=time.time()-t0
            
            t0=time.time()
            # Run eGMA
            if i==0:
                custom_init=None
            else:
                custom_init = eGMA_init(G, G_prev, previous_pred_e)
                custom_init = _renumber(custom_init)
            y_pred_e = louvain_e.run(G, custom_init, previous_pred_e)
            ari_e = louvain_e.get_metrics(previous_pred_e, y_pred_e)
            ari_fin_e = louvain_e.get_metrics (self.y_true_final, y_pred_e)
            ari_init_e = louvain_e.get_metrics (self.y_true_init, y_pred_e)
            communities={}
            for node in y_pred_e:
                if not y_pred_e[node] in communities: communities[y_pred_e[node]]=list()
                communities[y_pred_e[node]].append(node)
            modularity_e=nx.community.modularity(G, list(communities.values()))
            t['louvain_e']+=time.time()-t0
            
            t0=time.time()
            
            # Run NeGMA
            if i==0:
                custom_init=None
            else:
                custom_init = NeGMA_init(G, G_prev, previous_pred_ne,mod_th=0)
                custom_init = _renumber(custom_init)
            y_pred_ne = louvain_ne.run(G, custom_init, previous_pred_ne)
            ari_ne = louvain_ne.get_metrics(previous_pred_ne, y_pred_ne)
            ari_fin_ne = louvain_ne.get_metrics (self.y_true_final, y_pred_ne)
            ari_init_ne = louvain_ne.get_metrics (self.y_true_init, y_pred_ne)
            communities={}
            for node in y_pred_ne:
                if not y_pred_ne[node] in communities: communities[y_pred_ne[node]]=list()
                communities[y_pred_ne[node]].append(node)
            modularity_ne=nx.community.modularity(G, list(communities.values()))
            t['louvain_ne']+=time.time()-t0
            
            
            # Run Leiden
            t0=time.time()
            zero_init = None
            y_pred_lei = leiden.run(G, zero_init, previous_pred_lei)
            ari_lei = leiden.get_metrics(previous_pred_lei, y_pred_lei)
            ari_fin_lei = leiden.get_metrics (self.y_true_final, y_pred_lei)
            ari_init_lei = leiden.get_metrics (self.y_true_init, y_pred_lei)
            communities={}
            for node in y_pred_lei:
                if not y_pred_lei[node] in communities: communities[y_pred_lei[node]]=list()
                communities[y_pred_lei[node]].append(node)
            modularity_lei=nx.community.modularity(G, list(communities.values()))
            t['leiden']+=time.time()-t0
            
            # Run aLeiden
            t0=time.time()
            _G = update_weights(G_prev_base, G)
            zero_init = None #{x:i for i,x in enumerate(_G.nodes)}
            y_pred_alei = aleiden.run(_G, zero_init, previous_pred_alei)
            # Here nodes are always active. Thus retrieve the correct nodes from 
            # GMA application
            y_pred_alei = {k:v for k,v in y_pred_alei.items() if k in y_pred}
            ari_alei = aleiden.get_metrics(previous_pred_alei, y_pred_alei)
            ari_fin_alei = aleiden.get_metrics (self.y_true_final, y_pred_alei)
            ari_init_alei = aleiden.get_metrics (self.y_true_init, y_pred_alei)
            communities={}
            for node in y_pred_alei:
                if not y_pred_alei[node] in communities: communities[y_pred_alei[node]]=list()
                communities[y_pred_alei[node]].append(node)
            modularity_alei=nx.community.modularity(G, list(communities.values()))
            t['aleiden']+=time.time()-t0#time for updating the dictionary
            
            # Run iLeiden
            t0=time.time()
            if i==0:
                custom_init=None
            else:
                custom_init = iGMA_init(G, G_prev, previous_pred_ilei)
                
                custom_init=_renumber(custom_init)
            y_pred_ilei = leiden_i.run(G, custom_init, previous_pred_ilei)
            
            ari_ilei = leiden_i.get_metrics(previous_pred_ilei, y_pred_ilei)
            ari_fin_ilei = leiden_i.get_metrics (self.y_true_final, y_pred_ilei)
            ari_init_ilei = leiden_i.get_metrics (self.y_true_init, y_pred_ilei)
            communities={}
            for node in y_pred_ilei:
                if not y_pred_ilei[node] in communities: communities[y_pred_ilei[node]]=list()
                communities[y_pred_ilei[node]].append(node)
            modularity_ilei=nx.community.modularity(G, list(communities.values()))
            t['leiden_i']+=time.time()-t0
            
            
            t0=time.time()
            # Run NeLeiden
            if i==0:
                custom_init=None
            else:
                custom_init = NeGMA_init(G, G_prev, previous_pred_nelei,mod_th=0)
                custom_init=_renumber(custom_init)
            y_pred_nelei = leiden_ne.run(G, custom_init, previous_pred_nelei)
            ari_nelei = leiden_ne.get_metrics(previous_pred_nelei, y_pred_nelei)
            ari_fin_nelei = leiden_ne.get_metrics (self.y_true_final, y_pred_nelei)
            ari_init_nelei = leiden_ne.get_metrics (self.y_true_init, y_pred_nelei)
            communities={}
            for node in y_pred_nelei:
                if not y_pred_nelei[node] in communities: 
                    communities[y_pred_nelei[node]]=list()
                communities[y_pred_nelei[node]].append(node)
            modularity_nelei=nx.community.modularity(G, list(communities.values()))
            t['leiden_ne']+=time.time()-t0
            
            
            if pb: 
                progress_bar.update(1)
                print(', '.join([f'{key}:{t[key]/(i+1):.2f}' for key in t]))
                # print(modularity_lei)


            metrics.append((
                ari_base, ari_alei, ari, ari_lei, ari_i, ari_ilei, ari_e, ari_ne,ari_nelei,
                ari_init_base, ari_init_alei, ari_init, ari_init_lei, ari_init_i,ari_init_ilei, ari_init_e, ari_init_ne,ari_init_nelei,
                ari_fin_base, ari_fin_alei, ari_fin, ari_fin_lei, ari_fin_i,ari_fin_ilei, ari_fin_e, ari_fin_ne,ari_fin_nelei,
                modularity_base, modularity_alei, modularity, modularity_lei, modularity_i ,modularity_ilei, modularity_e, modularity_ne,modularity_nelei,
            ))
            if report:
                partitions.append((y_pred_base, y_pred_alei, y_pred, y_pred_lei, y_pred_i, y_pred_ilei, y_pred_e, y_pred_ne, y_pred_nelei))
            
            seed=np.random.randint(10000)
            louvain_base.seed = seed
            louvain.seed = seed
            louvain_i.seed = seed
            louvain_e.seed = seed
            louvain_ne.seed = seed


            aleiden.seed = seed
            leiden.seed = seed
            leiden_i.seed = seed
            leiden_ne.seed = seed
           
        if save:
            nx.set_node_attributes(G,self.y_true_final, name='community')
            nx.write_gexf(G, f'../results/graphs/{self.gname}_moved.gexf')
            
        if report:
            return metrics,partitions,self.y_true_final,self.y_true_init
        return metrics
        
    
    
