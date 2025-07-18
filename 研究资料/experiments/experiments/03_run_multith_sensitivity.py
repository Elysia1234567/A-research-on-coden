import sys
sys.path.append('../')
import pandas as pd
from multiprocessing import Pool, cpu_count, freeze_support  # 添加了 freeze_support
from src.simulation import Run_MultiTh, ths
from tqdm import tqdm
import os
import numpy as np

TRANSFORMATIONS = ['birth', 'fragment', 'split', 'merge',
                  'add_nodes', 'intermittence_nodes', 'switch',
                  'break_up_communities', 'remove_nodes']

NIT, TSS, DELTA = 5, 5, .01
GIT = 2  # Number of NIT with the same graph
n_split = 3
mu = .2

def wrapper_func(args):
    Num_Graph, seed, mu, timesteps, transformation = args
    run = Run_MultiTh(Num_Graph, seed, mu, timesteps, transformation)
    return run

def main():  # 将主要逻辑封装到 main 函数中
    if not os.path.exists('../results/reports/Threshold_sensitivity/'):
        os.makedirs('../results/reports/Threshold_sensitivity/')
    if not os.path.exists('../results/reports/Threshold_sensitivity/time/'):
        os.makedirs('../results/reports/Threshold_sensitivity/time/')

    for transformation in TRANSFORMATIONS:
        iterable = [(i//GIT, i%GIT, mu, TSS, transformation) for i in range(NIT)]
        results = []
        ts = {}
        with Pool(cpu_count()) as pool:
            with tqdm(total=len(iterable)) as pbar:
                for res, t in pool.imap(wrapper_func, iterable):
                    for key in t:
                        if key not in ts:
                            ts[key] = []
                        ts[key].append(t[key])
                    pbar.set_description(transformation + ' PRC COUNT:\n' + 
                                       ', '.join([f'{key}:{np.mean(ts[key])/(TSS):.2f}' 
                                                for key in ts]) + '\n')
                    pbar.update(1)
                    results.append(res)

        print("SAVING")
        columns = [[f'ari_louvain_{th}' for th in ths] + 
                  [f'ari_leiden_{th}' for th in ths] +
                  [f'ari_fin_louvain_{th}' for th in ths] + 
                  [f'ari_fin_leiden_{th}' for th in ths] +
                  [f'ari_init_louvain_{th}' for th in ths] + 
                  [f'ari_init_leiden_{th}' for th in ths] +
                  [f'modularity_louvain_{th}' for th in ths] + 
                  [f'modularity_leiden_{th}' for th in ths]]

        metrics = [pd.DataFrame(x, columns=columns) for x in results]
        metrics = pd.concat(metrics, axis=1)
        metrics.to_csv(
            f'../results/reports/Threshold_sensitivity/multith_{transformation}_mu{int(mu*100)}_it{NIT}.csv.gz', 
            index=False, compression='gzip'
        )
        
        pd.DataFrame(ts).to_csv(
            f'../results/reports/Threshold_sensitivity/time/time_{transformation}_mu{int(mu*100)}_it{NIT}.csv.gz', 
            index=False, compression='gzip'
        )

if __name__ == '__main__':  # 保护主模块执行
    freeze_support()  # Windows 多进程必需
    main()