import sys
sys.path.append('../')
import pandas as pd
from multiprocessing import Pool, cpu_count
from src.simulation import RunCommunityDetection
from tqdm import tqdm
import os
import numpy as np
import gzip
import json


TRANSFORMATIONS=['birth', 'fragment','split','merge',
                  'add_nodes','intermittence_nodes', 'switch',
                  'break_up_communities','remove_nodes']

instant=False
report=True

if report:
    NIT, TSS, DELTA =50, 150, .01
    GIT=NIT
elif instant:
    NIT, TSS, DELTA =1000, 20, 1
    GIT=10
else:
    NIT, TSS, DELTA =1000, 150, .01
    GIT=10
    

 # Number of NIT with the same graph

print('instant?',instant)
print('report?',report)


n_split=3
mu = .2

def wrapper_func(args):
    Num_Graph,seed,mu, timesteps, transformation = args
    global instant
    global report
    run=RunCommunityDetection( Num_Graph,seed,mu, timesteps, transformation,
                             instant=instant,report=report)
    return run

if not os.path.exists('../results/reports/'):
    os.makedirs('../results/reports/')



# for transformation in ['fragment', 'split', 'merge', 'add_nodes', 'remove_nodes', 'on_off_nodes', 'on_off_edges', 'shuffle_edges', 'remove_edges']:
for transformation in TRANSFORMATIONS:

    iterable = [(i//GIT ,i%GIT,mu, TSS, transformation) for i in range(NIT)]
        
    results=[]
    partitions={}
    ts={}
    # wrapper_func(iterable[0])
    with Pool(min(cpu_count(),NIT)) as pool:
        with tqdm(total=len(iterable),desc=f'PRC COUNT {transformation}') as pbar:  # Create a tqdm progress bar
            for i,res in enumerate(pool.imap( wrapper_func, iterable)):
                if report:
                    metrics,part,y_true_final,y_true_init=res
                    partitions[i]={'y_true_final':y_true_final,
                                   'y_true_init':y_true_init}
                    for j,pred in enumerate(['y_pred_base', 
                             'y_pred_alei', 
                             'y_pred', 
                             'y_pred_lei', 
                             'y_pred_i', 
                             'y_pred_ilei', 
                             'y_pred_e', 
                             'y_pred_ne', 
                             'y_pred_nelei'
                            ]):
                        partitions[i][pred]={step:part[step][j] for step in range(TSS)}
                    
                else: 
                    metrics,t=res
                    
                    #,y_true_final,y_true_init,t
                
                    for key in t:
                        if not key in ts: ts[key]=[]
                        ts[key].append(t[key])
                    pbar.set_description(transformation+' PRC COUNT:\n'+', '.join([f'{key}:{np.mean(ts[key])/(TSS):.2f}' for key in ts])+'\n')
                pbar.update(1)
                results.append(metrics)
                                     
    
                
                    
                
                
    print("SAVING")
    # Arrange metrics

    columns = ['ari_base', 'ari_alei', 'ari', 'ari_lei',  'ari_i','ari_ilei', 'ari_e', 'ari_ne', 'ari_nelei',
               'ari_init_base', 'ari_init_alei', 'ari_init', 'ari_init_lei', 'ari_init_i','ari_init_ilei', 'ari_init_e', 'ari_init_ne','ari_init_nelei',
        'ari_fin_base', 'ari_fin_alei', 'ari_fin', 'ari_fin_lei', 'ari_fin_i', 'ari_fin_ilei', 'ari_fin_e', 'ari_fin_ne', 'ari_fin_nelei',
               'modularity_base', 'modularity_alei', 'modularity', 'modularity_lei', 'modularity_i' ,'modularity_ilei', 'modularity_e', 'modularity_ne', 'modularity_nelei',
              ]
    metrics = [pd.DataFrame(x, columns=columns) for x in results]
    metrics = pd.concat(metrics, axis=1)
   
    # Save metrics


    if report:
        
            
        if instant:
            metrics.to_csv(
                f'../results/reports/INST_{transformation}_mu{int(mu*100)}_singlerun.csv.gz', 
                index=False,compression='gzip'
            )
            
            with gzip.open(f'../results/reports/INST_{transformation}_mu{int(mu*100)}_partitions.json.gz', "wt") as json_file:
                json.dump(partitions, json_file,default=lambda x: int(x))


        else:
            metrics.to_csv(
                f'../results/reports/{transformation}_mu{int(mu*100)}_singlerun.csv.gz', 
                index=False,compression='gzip'
            )
            
            with gzip.open(f'../results/reports/{transformation}_mu{int(mu*100)}_partitions.json.gz', "wt") as json_file:
                json.dump(partitions, json_file,default=lambda x: int(x))
            
        

        
        
        
        
    else:
        if not os.path.exists('../results/reports/time/'):
            os.makedirs('../results/reports/time/')

    
        if instant:
            metrics.to_csv(
                f'../results/reports/INST_{transformation}_mu{int(mu*100)}_it{NIT}.csv.gz', 
                index=False,compression='gzip'
            )

            pd.DataFrame(ts).to_csv(
                f'../results/reports/time/INST_time_{transformation}_mu{int(mu*100)}_it{NIT}.csv.gz', 
                index=False,compression='gzip'
            )
        else:
            metrics.to_csv(
                f'../results/reports/{transformation}_mu{int(mu*100)}_it{NIT}.csv.gz', 
                index=False,compression='gzip'
            )

            pd.DataFrame(ts).to_csv(
                f'../results/reports/time/time_{transformation}_mu{int(mu*100)}_it{NIT}.csv.gz', 
                index=False,compression='gzip'
            )
        
    
                

