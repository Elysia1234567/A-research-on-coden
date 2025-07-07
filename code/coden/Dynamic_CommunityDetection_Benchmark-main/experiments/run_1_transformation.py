import sys
sys.path.append('../')
import pandas as pd
from multiprocessing import Pool, cpu_count
from src.simulation import SingelGraphEvolution
from tqdm import tqdm
import copy
import json
import numpy as np
import traceback
import os
 


TRANSFORMATIONS=['birth', 'fragment','split','merge',
                  'add_nodes','intermittence_nodes', 'switch',
                  'break_up_communities','remove_nodes']
inst=True
print('Instant?',inst)
if inst:
    NIT, TSS, DELTA = 1, 20,1
    start_trans=10
    stop_trans=start_trans+1
else:
    NIT, TSS, DELTA = 1, 150,0.1
    start_trans=10
    stop_trans=start_trans+1
GIT=10 # Number of NIT with the same graph
n_split=1
mu = .2


if not os.path.exists('../results/reports/Single_run_details/'):
    os.makedirs('../results/reports/Single_run_details/')
if not os.path.exists('../results/reports/Single_run_details/'):
    os.makedirs('../results/reports/Single_run_details/')



def wrapper_func(args):
    sim, G,seed, transformation, tss, delta = args
    
    if seed == 0: save = True
    else: save = False
    if inst:
        run=sim.run(G, seed=seed, timesteps=tss, delta=delta, transformation=transformation, save=save,report=True,start_trans=start_trans,stop_trans=stop_trans)
    else:
        run=sim.run(G, seed=seed, timesteps=tss, delta=delta, transformation=transformation, save=save,report=True)
        
    return run





# for transformation in ['fragment', 'split', 'merge', 'add_nodes', 'remove_nodes', 'on_off_nodes', 'on_off_edges', 'shuffle_edges', 'remove_edges']:

    
sim_cnt = 1




simulators = {}
seed=np.random.randint(1000)

for transformation in tqdm(TRANSFORMATIONS):
    gname=f'{transformation}_mu{int(mu*100)}'
    if transformation in ['remove_nodes','remove_edges']: 
        n=300
    else: 
        n=200
    while True:
        try:
            print(sim_cnt)
            
            sim = SingelGraphEvolution(n=n, mu=mu, gname=f'{gname}', seed=seed)
            G = sim.setup_transformation(transformation, n_splits=n_split, save=False)

            break
        except Exception as e:
            
            sim_cnt+=1
            print(transformation)
            print(sim_cnt,e)
            print(print(traceback.format_exc()))
            seed=np.random.randint(1000)
            

    simulators[transformation]=(sim,G)

# Run experiments in multiprocessing
iterable = [(transformation, 
             42, 
             simulators[transformation][0],simulators[transformation][1]) for transformation in TRANSFORMATIONS]

with Pool(len(iterable)) as pool:
    # Map the wrapper function to the iterable using the pool
    results = pool.map(
        wrapper_func, 
        [(sim,G, i, transformation, TSS, DELTA,) for transformation, i, sim,G in iterable
        ]
    )

# Arrange metrics

columns = ['ari_base', 'ari_alei', 'ari', 'ari_lei',  'ari_i','ari_ilei', 'ari_e', 'ari_ne','ari_nelei',
           'ari_init_base', 'ari_init_alei', 'ari_init', 'ari_init_lei', 'ari_init_i','ari_init_ilei', 'ari_init_e', 'ari_init_ne','ari_init_nelei',
    'ari_fin_base', 'ari_fin_alei', 'ari_fin', 'ari_fin_lei', 'ari_fin_i', 'ari_fin_ilei', 'ari_fin_e', 'ari_fin_ne',  'ari_fin_nelei',
           'modularity_base', 'modularity_alei', 'modularity', 'modularity_lei', 'modularity_i' ,'modularity_ilei', 'modularity_e', 'modularity_ne', 'modularity_nelei',
          ]

for i,transformation in enumerate(tqdm(TRANSFORMATIONS)):
    metrics = pd.DataFrame(results[i][0], columns=columns) 
    if inst:
        dest=f'../results/reports/Single_run_details/INST_metrics_{transformation}_mu{int(mu*100)}_it{NIT}.csv.gz'
        
    # Save metrics
    else:
        dest=f'../results/reports/Single_run_details/metrics_{transformation}_mu{int(mu*100)}_it{NIT}.csv.gz',
    metrics.to_csv(
        dest, 
        index=False,compression='gzip'
    )
    
    partitions = {str(j):{'y_pred_base':y_pred_base, 
                     'y_pred_alei':y_pred_alei, 
                     'y_pred':y_pred, 
                     'y_pred_lei':y_pred_lei, 
                     'y_pred_i':y_pred_i, 
                     'y_pred_ilei':y_pred_ilei, 
                     'y_pred_e': y_pred_e,
                     'y_pred_ne':y_pred_ne,
                     'y_pred_nelei':y_pred_nelei
                         }

                  for j, (y_pred_base, y_pred_alei, y_pred, y_pred_lei, y_pred_i, y_pred_ilei, y_pred_e, y_pred_ne, y_pred_nelei) in enumerate(results[i][1])
                         }
    
    partitions["y_true_final"]={key:int(value) for key,value in results[i][2].items()}
    partitions["y_true_init"]= {key:int(value) for key,value in results[i][3].items()}


    # Save metrics
    if inst:
        dest=f'../results/reports/Single_run_details/INST_partitions_{transformation}_mu{int(mu*100)}_it{NIT}.json'
        
    # Save metrics
    else:
        dest=f'../results/reports/Single_run_details/partitions_{transformation}_mu{int(mu*100)}_it{NIT}.json'
    print(partitions)
    with open(dest, "w") as json_file:
        json.dump(partitions, json_file,default=lambda x: int(x))
        
