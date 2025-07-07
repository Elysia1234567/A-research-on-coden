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

from ..transformations import update_weights

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


def RunCommunityDetection( Num_Graph,seed,mu=0.2, timesteps=300, transformation='split', report=False,instant=False):
    
        if Num_Graph==0 and seed==0 and report==True: 
            pb=True
            progress_bar = tqdm(total=timesteps, desc=f'Run CD {transformation}')
        else: pb=False
            
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
        

        if instant:
            GT=pd.read_csv(f'../results/graphs/mu{int(mu*10):02}/{transformation}/G{Num_Graph:02}_INST/GT.csv.gz')
        else:
            GT=pd.read_csv(f'../results/graphs/mu{int(mu*10):02}/{transformation}/G{Num_Graph:02}/GT.csv.gz')
            
        GT.set_index('Node_id',inplace=True)
        y_true_final=GT.y_true_final.dropna().astype(int).to_dict()
        y_true_init=GT.y_true_init.dropna().astype(int).to_dict()
        
        
        for i in range(timesteps):   

            # Update predictions and communities    
            if i==0: 
                previous_pred_base = y_true_init.copy()
                
                # GMA
                previous_pred = y_true_init.copy()
                
                # iGMA
                previous_pred_i = y_true_init.copy()
                
                # eGMA
                previous_pred_e = y_true_init.copy()
                
                # NeGMA
                previous_pred_ne = y_true_init.copy()
                

                # aLeiden
                previous_pred_alei = y_true_init.copy()
                
                # Leiden
                previous_pred_lei = y_true_init.copy()
                # iLeiden
                
                previous_pred_ilei = y_true_init.copy()

                # NeLei
                previous_pred_nelei = y_true_init.copy()
                
                
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
            if i==0:
                if instant:

                    G=nx.read_weighted_edgelist(
                        f'../results/graphs/mu{int(10*mu):02}/{transformation}/G{Num_Graph:02}_INST/T{i:03}.txt.gz',nodetype=str
                    )
                else:

                    G=nx.read_weighted_edgelist(
                        f'../results/graphs/mu{int(10*mu):02}/{transformation}/G{Num_Graph:02}/T{i:03}.txt.gz',nodetype=str
                    )
                
                G_prev = G.copy()
                G_prev_base = G.copy() 
                
            else:
                G_prev = G.copy()
                G_prev_base = _G.copy()
                if instant: 

                    G=nx.read_weighted_edgelist(
                    f'../results/graphs/mu{int(10*mu):02}/{transformation}/G{Num_Graph:02}_INST/T{i:03}.txt.gz',nodetype=str
                    )
                else:

                    G=nx.read_weighted_edgelist(
                    f'../results/graphs/mu{int(10*mu):02}/{transformation}/G{Num_Graph:02}/T{i:03}.txt.gz',nodetype=str
                    )
                
            
            ###################################################################
            # DETECT COMMUNITIES
            ###################################################################
            
            
            t0=time.time()
            # Run GMA
            zero_init = None #{x:i for i,x in enumerate(G.nodes)}
            y_pred = louvain.run(G, zero_init, previous_pred)
            ari = louvain.get_metrics(previous_pred, y_pred)
            ari_fin = louvain.get_metrics(y_true_final, y_pred)
            ari_init = louvain.get_metrics(y_true_init, y_pred)
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
            ari_fin_base = louvain_base.get_metrics(y_true_final, y_pred_base)
            ari_init_base = louvain_base.get_metrics(y_true_init, y_pred_base)
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
            ari_fin_i = louvain_i.get_metrics(y_true_final, y_pred_i)
            ari_init_i = louvain_i.get_metrics(y_true_init, y_pred_i)
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
            ari_fin_e = louvain_e.get_metrics(y_true_final, y_pred_e)
            ari_init_e = louvain_e.get_metrics(y_true_init, y_pred_e)
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
            ari_fin_ne = louvain_ne.get_metrics(y_true_final, y_pred_ne)
            ari_init_ne = louvain_ne.get_metrics(y_true_init, y_pred_ne)
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
            ari_fin_lei = leiden.get_metrics(y_true_final, y_pred_lei)
            ari_init_lei = leiden.get_metrics(y_true_init, y_pred_lei)
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
            ari_fin_alei = aleiden.get_metrics(y_true_final, y_pred_alei)
            ari_init_alei = aleiden.get_metrics(y_true_init, y_pred_alei)
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
            ari_fin_ilei = leiden_i.get_metrics(y_true_final, y_pred_ilei)
            ari_init_ilei = leiden_i.get_metrics(y_true_init, y_pred_ilei)
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
            ari_fin_nelei = leiden_ne.get_metrics(y_true_final, y_pred_nelei)
            ari_init_nelei = leiden_ne.get_metrics(y_true_init, y_pred_nelei)
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

        if report:
            return metrics,partitions,y_true_final,y_true_init#,t
        return metrics,t
        
    
    
