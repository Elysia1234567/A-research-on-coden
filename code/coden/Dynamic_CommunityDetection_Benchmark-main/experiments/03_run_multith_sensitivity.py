import sys
sys.path.append('../')
import pandas as pd
from multiprocessing import Pool, cpu_count
from src.simulation import Run_MultiTh,ths #导入多阈值模拟器和预定义阈值列表
from tqdm import tqdm
import os
import numpy as np
 

TRANSFORMATIONS=['birth', 'fragment','split','merge',
                  'add_nodes','intermittence_nodes', 'switch',
                  'break_up_communities','remove_nodes']

NIT, TSS, DELTA =1000, 150, .01

GIT=10 # Number of NIT with the same graph
n_split=3
mu = .2

def wrapper_func(args):
    Num_Graph,seed,mu, timesteps, transformation = args
    
    run=Run_MultiTh( Num_Graph,seed,mu, timesteps, transformation)
    return run

# 输出目录创建
if not os.path.exists('../results/reports/Threshold_sensitivity/'):
    os.makedirs('../results/reports/Threshold_sensitivity/')
if not os.path.exists('../results/reports/Threshold_sensitivity/time/'):
    os.makedirs('../results/reports/Threshold_sensitivity/time/')

# for transformation in ['fragment', 'split', 'merge', 'add_nodes', 'remove_nodes', 'on_off_nodes', 'on_off_edges', 'shuffle_edges', 'remove_edges']:
for transformation in TRANSFORMATIONS:
    
    iterable = [(i//GIT,i%GIT,mu, TSS, transformation) for i in range(NIT)]
    results=[]
    ts={}
    with Pool(cpu_count()) as pool:
        with tqdm(total=len(iterable)) as pbar:  # Create a tqdm progress bar
            for res,t in pool.imap( wrapper_func, iterable):
                for key in t:
                    if not key in ts: ts[key]=[]
                    ts[key].append(t[key])
                pbar.set_description(transformation+' PRC COUNT:\n'+', '.join([f'{key}:{np.mean(ts[key])/(TSS):.2f}' for key in ts])+'\n')
                pbar.update(1)
                results.append(res)
                
                                        
                
                
    print("SAVING")
    # Arrange metrics
    # 评估了Louvain和Leiden两种社区检测算法，每个算法应用ths中的所有阈值
    # ari_*:演化过程平均ARI,ari_init_*:初始状态ARI,ari_fin_*：最终状态ARI,modularity_*模块度分数
    columns = [[f'ari_louvain_{th}' for th in ths]+[f'ari_leiden_{th}' for th in ths]+
                [f'ari_fin_louvain_{th}' for th in ths]+[f'ari_fin_leiden_{th}' for th in ths]+
                [f'ari_init_louvain_{th}' for th in ths]+[f'ari_init_leiden_{th}' for th in ths]+
                [f'modularity_louvain_{th}' for th in ths]+[f'modularity_leiden_{th}' for th in ths] 
              ]
    
    
    
    
    
    metrics = [pd.DataFrame(x, columns=columns) for x in results]
    metrics = pd.concat(metrics, axis=1)
    # Save metrics
    metrics.to_csv( #多阈值分裂演化结果
        f'../results/reports/Threshold_sensitivity/multith_{transformation}_mu{int(mu*100)}_it{NIT}.csv.gz', 
        index=False,compression='gzip'
    )
    
    pd.DataFrame(ts).to_csv( #各算法耗时记录
            f'../results/reports/Threshold_sensitivity/time/time_{transformation}_mu{int(mu*100)}_it{NIT}.csv.gz', 
            index=False,compression='gzip'
        )


