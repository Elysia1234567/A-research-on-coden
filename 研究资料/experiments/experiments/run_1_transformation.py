import sys
# 将项目根目录添加到Python路径的最前面
sys.path.insert(0, "D:\srtq项目\Dynamic_CommunityDetection_Benchmark-main")
import os
import pandas as pd
from multiprocessing import Pool, cpu_count, freeze_support
from src.simulation import SingelGraphEvolution
from tqdm import tqdm
import numpy as np
import traceback
import json

# 设置项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TRANSFORMATIONS = ['birth', 'fragment', 'split', 'merge',
                 'add_nodes', 'intermittence_nodes', 'switch',
                 'break_up_communities', 'remove_nodes']

# 全局配置
inst = True
print('Instant?', inst)
if inst:
    NIT, TSS, DELTA = 1, 20, 1
    start_trans, stop_trans = 10, 11
else:
    NIT, TSS, DELTA = 1, 150, 0.1
    start_trans, stop_trans = 10, 11

GIT = 10  # Number of NIT with the same graph
n_split = 1
mu = 0.2

# 确保输出目录存在
output_dir = '../results/reports/Single_run_details/'
os.makedirs(output_dir, exist_ok=True)

def wrapper_func(args):
    """可序列化的包装函数，处理单个任务"""
    sim, G, seed, transformation, tss, delta, inst, start_trans, stop_trans = args
    save = (seed == 0)
    
    try:
        if inst:
            return sim.run(G, seed=seed, timesteps=tss, delta=delta, 
                         transformation=transformation, save=save, report=True,
                         start_trans=start_trans, stop_trans=stop_trans)
        else:
            return sim.run(G, seed=seed, timesteps=tss, delta=delta,
                         transformation=transformation, save=save, report=True)
    except Exception as e:
        print(f"Error in {transformation} with seed {seed}: {str(e)}")
        traceback.print_exc()
        return (None, None, None, None)  # 保持一致的返回结构

def initialize_simulators():
    """初始化所有模拟器"""
    simulators = {}
    seed = np.random.randint(1000)
    sim_cnt = 1

    for transformation in tqdm(TRANSFORMATIONS, desc="Initializing simulators"):
        gname = f'{transformation}_mu{int(mu*100)}'
        n = 300 if transformation in ['remove_nodes', 'remove_edges'] else 200
        
        while True:
            try:
                sim = SingelGraphEvolution(n=n, mu=mu, gname=gname, seed=seed)
                G = sim.setup_transformation(transformation, n_splits=n_split, save=False)
                simulators[transformation] = (sim, G)
                break
            except Exception as e:
                sim_cnt += 1
                print(f"Retry {transformation} (attempt {sim_cnt}): {str(e)}")
                seed = np.random.randint(1000)
    
    return simulators

def save_results(results):
    """保存结果到文件"""
    columns = ['ari_base', 'ari_alei', 'ari', 'ari_lei', 'ari_i', 'ari_ilei', 
              'ari_e', 'ari_ne', 'ari_nelei', 'ari_init_base', 'ari_init_alei', 
              'ari_init', 'ari_init_lei', 'ari_init_i', 'ari_init_ilei', 
              'ari_init_e', 'ari_init_ne', 'ari_init_nelei', 'ari_fin_base', 
              'ari_fin_alei', 'ari_fin', 'ari_fin_lei', 'ari_fin_i', 
              'ari_fin_ilei', 'ari_fin_e', 'ari_fin_ne', 'ari_fin_nelei',
              'modularity_base', 'modularity_alei', 'modularity', 
              'modularity_lei', 'modularity_i', 'modularity_ilei', 
              'modularity_e', 'modularity_ne', 'modularity_nelei']

    for i, transformation in enumerate(tqdm(TRANSFORMATIONS, desc="Saving results")):
        if results[i] is None or results[i][0] is None:
            continue
            
        # 保存指标
        metrics = pd.DataFrame(results[i][0], columns=columns)
        dest = os.path.join(output_dir, 
                          f'{"INST_" if inst else ""}metrics_{transformation}_mu{int(mu*100)}_it{NIT}.csv.gz')
        metrics.to_csv(dest, index=False, compression='gzip')

        # 保存分区数据
        if len(results[i]) >= 4:  # 确保有足够的结果数据
            partitions = {
                str(j): {
                    'y_pred_base': y_pred_base,
                    'y_pred_alei': y_pred_alei,
                    'y_pred': y_pred,
                    'y_pred_lei': y_pred_lei,
                    'y_pred_i': y_pred_i,
                    'y_pred_ilei': y_pred_ilei,
                    'y_pred_e': y_pred_e,
                    'y_pred_ne': y_pred_ne,
                    'y_pred_nelei': y_pred_nelei
                } for j, (y_pred_base, y_pred_alei, y_pred, y_pred_lei, 
                         y_pred_i, y_pred_ilei, y_pred_e, y_pred_ne, y_pred_nelei) in enumerate(results[i][1])
            }
            
            partitions.update({
                "y_true_final": {key: int(value) for key, value in results[i][2].items()},
                "y_true_init": {key: int(value) for key, value in results[i][3].items()}
            })

            dest = os.path.join(output_dir,
                              f'{"INST_" if inst else ""}partitions_{transformation}_mu{int(mu*100)}_it{NIT}.json')
            with open(dest, "w") as f:
                json.dump(partitions, f, default=lambda x: int(x))

def main():
    # 初始化所有模拟器
    simulators = initialize_simulators()

    # 准备多进程任务
    iterable = [(simulators[trans][0], simulators[trans][1], 42, trans, TSS, DELTA, inst, start_trans, stop_trans)
               for trans in TRANSFORMATIONS]

    # 运行多进程
    with Pool(processes=min(len(iterable), cpu_count())) as pool:
        results = list(tqdm(pool.imap(wrapper_func, iterable), total=len(iterable)))

    # 保存结果
    save_results(results)

if __name__ == '__main__':
    freeze_support()  # Windows多进程必需
    main()