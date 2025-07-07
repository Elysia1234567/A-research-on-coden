import sys
sys.path.append('../')
import pandas as pd
from multiprocessing import Pool, cpu_count
from src.simulation import GraphMovementSimulation
from tqdm import tqdm
import os



TRANSFORMATIONS=['birth', 'fragment','split','merge',
                  'add_nodes','intermittence_nodes', 'switch',
                  'break_up_communities','remove_nodes']

NIT, TSS, DELTA =1000, 150, .01

GIT=10 # Number of NIT with the same graph
n_split=3
mu = .2

def wrapper_func(args):
    sim,G, i, transf, tss, delta = args
    
    run=sim.run( G, N_Graph=i, delta=delta, transformation=transf,timesteps=tss)
    return run








for transformation in TRANSFORMATIONS:
    
    sim_cnt = 1
    
    if transformation in ['remove_nodes','remove_edges']: 
        n=3000
    else: 
        n=1000
    
    
    directory_path=f'../results/graphs/mu0{int(mu*10)}/{transformation}/'
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    gname=f'{transformation}_mu0{int(mu*10)}'
    simulators = list()
    
    seeds=[]


    for i in tqdm(range(NIT//GIT), desc = f'Generete Graphs {gname}'):
        while True:
            try:
                seed=i+sim_cnt
                sim = GraphMovementSimulation(n=n, mu=mu, gname=f'{gname}_{i}', seed=i+sim_cnt)
                G = sim.setup_transformation(transformation, n_splits=n_split, save=True)
                sim_cnt+=1
                break
            except Exception as e:
                sim_cnt+=1
                # print(f"Error: {e}")
                # print(sim_cnt)
                
        seeds.append((i,seed))
        simulators.append((sim,G))
        
       
        # Check if the directory exists
        if not os.path.exists(directory_path+ f'G{i:02}/'):
            # If it doesn't exist, create the directory
            os.makedirs(directory_path+ f'G{i:02}/')
            
    with open(directory_path+'seeds.csv','w') as f:
        f.write('num_graph,seed\n')
        lines= '\n'.join([",".join(map(str, item)) for item in seeds])
        f.write(lines)
    # Run experiments in multiprocessing
    iterable = [(transformation, 
                 i, 
                 simulators[i][0],simulators[i][1]) for i in range(NIT//GIT)]
    with Pool(cpu_count()) as pool:
        with tqdm(total=len(iterable),desc=f'PRC COUNT {transformation}') as pbar:  # Create a tqdm progress bar
            for num_graph,GT in pool.imap( wrapper_func, 
                                  [(sim,G, i, transformation, TSS, DELTA) for transformation, i, sim,G in iterable]
                                ):
                pbar.update(1)
                pd.DataFrame({'y_true_init':GT['y_true_init'],'y_true_final':GT[ 'y_true_final'] }).fillna('')\
                .to_csv(f'../results/graphs/mu0{int(mu*10)}/{transformation}/G{num_graph:02}/GT.csv.gz',sep=',',index=True,index_label='Node_id')
                

