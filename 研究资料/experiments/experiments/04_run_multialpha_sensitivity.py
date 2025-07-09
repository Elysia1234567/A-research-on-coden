import sys
# 将项目根目录添加到Python路径的最前面
sys.path.insert(0, "D:\srtq项目\Dynamic_CommunityDetection_Benchmark-main")
import os
import pandas as pd
from multiprocessing import Pool, cpu_count, freeze_support
from src.simulation import Run_MultiAlpha, alphas
from tqdm import tqdm
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
    run = Run_MultiAlpha(Num_Graph, seed, mu, timesteps, transformation)
    return run

def main():
    if not os.path.exists('../results/reports/Alpha_sensitivity/'):
        os.makedirs('../results/reports/Alpha_sensitivity/')
    if not os.path.exists('../results/reports/Alpha_sensitivity/time/'):
        os.makedirs('../results/reports/Alpha_sensitivity/time/')

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
                    pbar.update(1)
                    results.append(res)

        print("SAVING")
        columns = [
            [f'ari_louvain_{alpha}' for alpha in alphas] +
            [f'ari_leiden_{alpha}' for alpha in alphas] +
            [f'ari_fin_louvain_{alpha}' for alpha in alphas] +
            [f'ari_fin_leiden_{alpha}' for alpha in alphas] +
            [f'ari_init_louvain_{alpha}' for alpha in alphas] +
            [f'ari_init_leiden_{alpha}' for alpha in alphas] +
            [f'modularity_louvain_{alpha}' for alpha in alphas] +
            [f'modularity_leiden_{alpha}' for alpha in alphas]
        ]
        
        metrics = [pd.DataFrame(x, columns=columns) for x in results]
        metrics = pd.concat(metrics, axis=1)
        
        metrics.to_csv(
            f'../results/reports/Alpha_sensitivity/multialpha_{transformation}_mu{int(mu*100)}_it{NIT}.csv.gz',
            index=False, compression='gzip'
        )
        
        pd.DataFrame(ts).to_csv(
            f'../results/reports/Alpha_sensitivity/time/time_{transformation}_mu{int(mu*100)}_it{NIT}.csv.gz',
            index=False, compression='gzip'
        )

if __name__ == '__main__':
    freeze_support()  # Required for Windows support
    main()