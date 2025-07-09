import sys
from pathlib import Path
import pandas as pd
from multiprocessing import Pool, cpu_count, freeze_support
from tqdm import tqdm
import os
import numpy as np

# 设置项目路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.simulation import GraphMovementSimulation

# 定义社区演化类型
TRANSFORMATIONS = ['birth', 'fragment', 'split', 'merge',
                  'add_nodes', 'intermittence_nodes', 'switch',
                  'break_up_communities', 'remove_nodes']

# 测试参数
NIT, TSS, DELTA = 5, 20, .1  # 总实验次数, 时间步数, 演化步长
GIT = 2  # 每组图重复次数
n_split = 1  # 受演化影响的社区数量
mu = .2  # 社区混合参数

def wrapper_func(args):
    """封装模拟任务的运行逻辑"""
    sim, G, i, transf, tss, delta = args
    run = sim.run(G, N_Graph=i, delta=delta, transformation=transf, timesteps=tss)
    return run

def main():
    """主函数"""
    # 主循环：生成图数据
    for transformation in TRANSFORMATIONS:
        sim_cnt = 1
        
        # 设置节点数
        n = 300 if transformation in ['remove_nodes', 'remove_edges'] else 100
        
        # 创建输出目录
        directory_path = f'../results/graphs/mu0{int(mu*10)}/{transformation}/'
        os.makedirs(directory_path, exist_ok=True)
        gname = f'{transformation}_mu0{int(mu*10)}'
        
        simulators = []
        seeds = []

        # 图生成与模拟
        for i in tqdm(range(NIT//GIT), desc=f'Generate Graphs {gname}'):
            while True:
                try:
                    seed = i + sim_cnt
                    sim = GraphMovementSimulation(n=n, mu=mu, gname=f'{gname}_{i}', seed=seed)
                    G = sim.setup_transformation(transformation, n_splits=n_split, save=True)
                    sim_cnt += 1
                    break
                except Exception as e:
                    sim_cnt += 1
                    continue
            
            seeds.append((i, seed))
            simulators.append((sim, G))
            os.makedirs(directory_path + f'G{i:02}/', exist_ok=True)
        
        # 保存种子
        with open(directory_path + 'seeds.csv', 'w') as f:
            f.write('num_graph,seed\n')
            lines = '\n'.join([",".join(map(str, item)) for item in seeds])
            f.write(lines)
        
        # 准备多进程任务
        iterable = [(transformation, i, simulators[i][0], simulators[i][1]) 
                   for i in range(NIT//GIT)]
        
        # 多进程运行模拟
        with Pool(processes=min(4, cpu_count())) as pool:  # 限制进程数
            with tqdm(total=len(iterable), desc=f'PRC COUNT {transformation}') as pbar:
                results = []
                for num_graph, GT in pool.imap(
                    wrapper_func,
                    [(sim, G, i, transformation, TSS, DELTA) 
                     for transformation, i, sim, G in iterable]
                ):
                    pbar.update(1)
                    # 结果保存
                    pd.DataFrame({
                        'y_true_init': GT['y_true_init'],
                        'y_true_final': GT['y_true_final']
                    }).fillna('').to_csv(
                        f'../results/graphs/mu0{int(mu*10)}/{transformation}/G{num_graph:02}/GT.csv.gz',
                        sep=',', index=True, index_label='Node_id'
                    )

if __name__ == '__main__':
    freeze_support()  # Windows多进程必需
    main()