import pandas as pd
import networkx as nx
from tqdm import tqdm
import numpy as np
import time
import copy

from ..graph import generate_weighted_LFR_graph
from ..communities import (
    Leiden,
    Louvain, 
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

alphas=[0.2,0.4,0.6,0.8,0.9,0.95,0.98,0.99,0.99]
def Run_MultiAlpha( Num_Graph,seed,mu=0.2, timesteps=300, transformation='split', report=False,alphas=alphas):
    
        # print(alphas)
    
        if (Num_Graph==0 and seed==0) or report==True: 
            pb=True
            progress_bar = tqdm(total=timesteps, desc=f'Run CD {transformation}')
        else: pb=False
            
        metrics = []
        partitions = []
        t={}
        
        louvain={}
        leiden={}
        for alpha in alphas:
            louvain[alpha]= Louvain(seed)
            leiden[alpha]=Leiden(seed)
            
            
        GT=pd.read_csv(f'../results/graphs/mu{int(mu*10):02}/{transformation}/G{Num_Graph:02}/GT.csv.gz',dtype={'Node_id':str})
        GT.set_index('Node_id',inplace=True)
        y_true_final=GT.y_true_final.dropna().to_dict()
        y_true_init=GT.y_true_init.dropna().to_dict()
        y_pred_louvain={}
        y_pred_leiden={}
        
        G_prev_base ={}
        _G={}
        
        for i in range(timesteps): 
            
            
            # Update predictions and communities  
            previous_pred_louvain={}
            previous_pred_leiden={}
            if i==0:
                for alpha in alphas:
                    previous_pred_louvain[alpha]= y_true_init.copy()
                    previous_pred_leiden[alpha]= y_true_init.copy()              

            else: 
                
                for alpha in alphas:
                    previous_pred_louvain[alpha]= y_pred_louvain[alpha]
                    previous_pred_leiden[alpha]= y_pred_leiden[alpha]            
                    
                         
            # Get the previous graph
            if i==0:
                G=nx.read_weighted_edgelist(f'../results/graphs/mu02/{transformation}/G{Num_Graph:02}/T000.txt.gz',nodetype =str)
                for alpha in alphas:
                    G_prev_base[alpha] = G.copy() 
                
                
            else:
                G=nx.read_weighted_edgelist(f'../results/graphs/mu02/{transformation}/G{Num_Graph:02}/T{i:03}.txt.gz',nodetype =str)
                G_prev_base = copy.deepcopy(_G)
                
                
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
            # Run alphaGMA-alphaLei
            for alpha in alphas:
                t0=time.time()
                
                t[f'louvain_{alpha}']=0
                t[f'leiden_{alpha}']=0
                    
                # For alpha-based framework
                _G[alpha] = update_weights(G_prev_base[alpha], G,ALPHA=alpha)
                    
                time_init=time.time()
                   
                custom_init_louvain=None
                y_pred_louvain[alpha]= louvain[alpha].run(_G[alpha], custom_init_louvain, previous_pred_louvain[alpha])
                y_pred_louvain[alpha]={node:comm for node,comm in y_pred_louvain[alpha].items() if node in G}
                t[f'louvain_{alpha}']+=time.time()-t0
                ari_louvain[alpha] = louvain[alpha].get_metrics(previous_pred_louvain[alpha], y_pred_louvain[alpha])
                ari_fin_louvain[alpha] = louvain[alpha].get_metrics(y_true_final, y_pred_louvain[alpha])
                
                ari_init_louvain[alpha] = louvain[alpha].get_metrics(y_true_init, y_pred_louvain[alpha])
                communities={}
                for node in y_pred_louvain[alpha]:
                    if not y_pred_louvain[alpha][node] in communities: communities[y_pred_louvain[alpha][node]]=list()
                    communities[y_pred_louvain[alpha][node]].append(node)
                modularity_louvain[alpha]=nx.community.modularity(G, list(communities.values()))
                
                
                t1=time.time()
                custom_init_leiden=None
                y_pred_leiden[alpha] = leiden[alpha].run(_G[alpha], custom_init_leiden, previous_pred_leiden[alpha])
                y_pred_leiden[alpha]={node:comm for node,comm in y_pred_leiden[alpha].items() if node in G}
                
                t[f'leiden_{alpha}']+=time.time()-t1+(time_init-t0)
                
                ari_leiden[alpha] = leiden[alpha].get_metrics(previous_pred_leiden[alpha], y_pred_leiden[alpha])
                ari_fin_leiden[alpha] = leiden[alpha].get_metrics(y_true_final, y_pred_leiden[alpha])
                ari_init_leiden[alpha] = leiden[alpha].get_metrics(y_true_init, y_pred_leiden[alpha])
                communities={}
                for node in y_pred_leiden[alpha]:
                    if not y_pred_leiden[alpha][node] in communities: communities[y_pred_leiden[alpha][node]]=list()
                    communities[y_pred_leiden[alpha][node]].append(node)
                modularity_leiden[alpha]=nx.community.modularity(G, list(communities.values()))
            
            
            if pb: 
                progress_bar.update(1)

            metrics.append((
                [ari_louvain[alpha] for alpha in alphas]+[ari_leiden[alpha] for alpha in alphas]+
                [ari_fin_louvain[alpha] for alpha in alphas]+[ari_fin_leiden[alpha] for alpha in alphas]+
                [ari_init_louvain[alpha] for alpha in alphas]+[ari_init_leiden[alpha] for alpha in alphas]+
                [modularity_louvain[alpha] for alpha in alphas]+[modularity_leiden[alpha] for alpha in alphas]  
            ))

            seed=np.random.randint(10000)
            for alpha in alphas:
                louvain[alpha].seed=seed
                leiden[alpha].seed=seed
        return metrics,t
        
    
    
