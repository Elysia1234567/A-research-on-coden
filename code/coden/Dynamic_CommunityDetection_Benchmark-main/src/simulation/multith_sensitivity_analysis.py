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

ths=[-1e-1,-1e-2,-1e-3,-1e-4,1e-4,1e-3,1e-2,1e-1]
def Run_MultiTh( Num_Graph,seed,mu=0.2, timesteps=300, transformation='split', report=False):
    
        if (Num_Graph==0 and seed==0) or report==True: 
            pb=True
            progress_bar = tqdm(total=timesteps, desc=f'Run CD {transformation}')
        else: pb=False
            
        metrics = []
        partitions = []
        t={}
        
        louvain={}
        leiden={}
        for th in ths:
            louvain[th]= Louvain(seed)
            leiden[th]=Leiden(seed)
            
            
        GT=pd.read_csv(f'../results/graphs/mu{int(mu*10):02}/{transformation}/G{Num_Graph:02}/GT.csv.gz',dtype={'Node_id':str})
        GT.set_index('Node_id',inplace=True)
        y_true_final=GT.y_true_final.dropna().to_dict()
        y_true_init=GT.y_true_init.dropna().to_dict()
        y_pred_louvain={}
        y_pred_leiden={}
        
        
        for i in range(timesteps): 
            
            
            # Update predictions and communities  
            previous_pred_louvain={}
            previous_pred_leiden={}
            if i==0:
                for th in ths:
                    previous_pred_louvain[th]= y_true_init.copy()
                    previous_pred_leiden[th]= y_true_init.copy()              

            else: 
                
                for th in ths:
                    previous_pred_louvain[th]= y_pred_louvain[th]
                    previous_pred_leiden[th]= y_pred_leiden[th]            
                    
                         
            # Get the previous graph
            if i==0:
                G=nx.read_weighted_edgelist(f'../results/graphs/mu02/{transformation}/G{Num_Graph:02}/T000.txt.gz',nodetype =str)
                G_prev = G.copy()
                
            else:
                G_prev = G.copy()
                G=nx.read_weighted_edgelist(f'../results/graphs/mu02/{transformation}/G{Num_Graph:02}/T{i:03}.txt.gz',nodetype =str)
            
            ###################################################################
            # DETECT COMMUNITIES
            ###################################################################
            
            ari_louvain={}
            ari_fin_louvain={}
            ari_init_louvain={}
            modularity_louvain={}
            ari_leiden={}
            ari_fin_leiden={}
            ari_init_leiden={}
            modularity_leiden={}
            # Run NeGMA-NeLei
            for th in ths:
                t0=time.time()
                if i==0:
                    t[f'louvain_{th}']=0
                    t[f'leiden_{th}']=0
                    
                    custom_init_louvain=None
                    time_init_GMA=time.time()
                    
                    custom_init_leiden=None
                    time_init_Lei=time.time()
                    
                    
                else:
                    custom_init_louvain = NeGMA_init(G, G_prev, previous_pred_louvain[th],mod_th=th)
                    custom_init_louvain = _renumber(custom_init_louvain)
                    time_init_GMA=time.time()
                    custom_init_leiden = NeGMA_init(G, G_prev, previous_pred_leiden[th],mod_th=th)
                    custom_init_leiden= _renumber(custom_init_leiden)
                    time_init_Lei=time.time()
                    
                
                y_pred_louvain[th] = louvain[th].run(G, custom_init_louvain, previous_pred_louvain[th])
                t[f'louvain_{th}']+=time.time()-time_init_Lei+(time_init_GMA-t0)
                ari_louvain[th] = louvain[th].get_metrics(previous_pred_louvain[th], y_pred_louvain[th])
                ari_fin_louvain[th] = louvain[th].get_metrics(y_true_final, y_pred_louvain[th])
                
                ari_init_louvain[th] = louvain[th].get_metrics(y_true_init, y_pred_louvain[th])
                communities={}
                for node in y_pred_louvain[th]:
                    if not y_pred_louvain[th][node] in communities: communities[y_pred_louvain[th][node]]=list()
                    communities[y_pred_louvain[th][node]].append(node)
                modularity_louvain[th]=nx.community.modularity(G, list(communities.values()))
                
                
                t1=time.time()
                y_pred_leiden[th] = leiden[th].run(G, custom_init_leiden, previous_pred_leiden[th])
                t[f'leiden_{th}']+=time.time()-t1+(time_init_Lei-time_init_GMA)
                
                ari_leiden[th] = leiden[th].get_metrics(previous_pred_leiden[th], y_pred_leiden[th])
                ari_fin_leiden[th] = leiden[th].get_metrics(y_true_final, y_pred_leiden[th])
                ari_init_leiden[th] = leiden[th].get_metrics(y_true_init, y_pred_leiden[th])
                communities={}
                for node in y_pred_leiden[th]:
                    if not y_pred_leiden[th][node] in communities: communities[y_pred_leiden[th][node]]=list()
                    communities[y_pred_leiden[th][node]].append(node)
                modularity_leiden[th]=nx.community.modularity(G, list(communities.values()))
            
            
            if pb: 
                progress_bar.update(1)

            metrics.append((
                [ari_louvain[th] for th in ths]+[ari_leiden[th] for th in ths]+
                [ari_fin_louvain[th] for th in ths]+[ari_fin_leiden[th] for th in ths]+
                [ari_init_louvain[th] for th in ths]+[ari_init_leiden[th] for th in ths]+
                [modularity_louvain[th] for th in ths]+[modularity_leiden[th] for th in ths]  
            ))

            seed=np.random.randint(10000)
            for th in ths:
                louvain[th].seed=seed
                leiden[th].seed=seed
        return metrics,t
        
    
    
