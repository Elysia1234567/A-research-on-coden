from pathlib import Path
import sys
import os
import pandas as pd
from multiprocessing import Pool, cpu_count, freeze_support
from tqdm import tqdm
import numpy as np
import gzip
import json

# 添加项目根目录到Python路径
sys.path.append('../')
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from src.simulation import RunCommunityDetection

# 全局配置
TRANSFORMATIONS = ['birth', 'fragment', 'split', 'merge',
                  'add_nodes', 'intermittence_nodes', 'switch',
                  'break_up_communities', 'remove_nodes']

# 全局变量（需要在wrapper_func中访问）
instant = False
report = True

def wrapper_func(args):
    """可序列化的包装函数，必须在模块级别定义"""
    Num_Graph, seed, mu, timesteps, transformation = args
    return RunCommunityDetection(
        Num_Graph, seed, mu, timesteps, transformation,
        instant=instant, report=report
    )

def process_results(transformation, NIT, GIT, mu, TSS):
    """处理单个变换类型的所有结果"""
    iterable = [(i//GIT, i%GIT, mu, TSS, transformation) for i in range(NIT)]
    results = []
    partitions = {}
    ts = {}

    with Pool(min(cpu_count(), NIT)) as pool:
        with tqdm(total=len(iterable), desc=f'Processing {transformation}') as pbar:
            for i, res in enumerate(pool.imap(wrapper_func, iterable)):
                if report:
                    metrics, part, y_true_final, y_true_init = res
                    partitions[i] = {
                        'y_true_final': y_true_final,
                        'y_true_init': y_true_init
                    }
                    for j, pred in enumerate([
                        'y_pred_base', 'y_pred_alei', 'y_pred', 
                        'y_pred_lei', 'y_pred_i', 'y_pred_ilei',
                        'y_pred_e', 'y_pred_ne', 'y_pred_nelei'
                    ]):
                        partitions[i][pred] = {step: part[step][j] for step in range(TSS)}
                else:
                    metrics, t = res
                    for key in t:
                        ts.setdefault(key, []).append(t[key])
                    pbar.set_description(
                        f"{transformation} Progress: " + 
                        ", ".join(f"{k}:{np.mean(v)/TSS:.2f}" for k, v in ts.items())
                    )
                pbar.update(1)
                results.append(metrics)
    return results, partitions, ts

def save_results(transformation, mu, NIT, results, partitions, ts):
    """保存结果到文件"""
    columns = [
        'ari_base', 'ari_alei', 'ari', 'ari_lei', 'ari_i', 'ari_ilei', 
        'ari_e', 'ari_ne', 'ari_nelei', 'ari_init_base', 'ari_init_alei', 
        'ari_init', 'ari_init_lei', 'ari_init_i', 'ari_init_ilei', 
        'ari_init_e', 'ari_init_ne', 'ari_init_nelei', 'ari_fin_base', 
        'ari_fin_alei', 'ari_fin', 'ari_fin_lei', 'ari_fin_i', 
        'ari_fin_ilei', 'ari_fin_e', 'ari_fin_ne', 'ari_fin_nelei',
        'modularity_base', 'modularity_alei', 'modularity', 
        'modularity_lei', 'modularity_i', 'modularity_ilei', 
        'modularity_e', 'modularity_ne', 'modularity_nelei'
    ]
    
    metrics_df = pd.concat([pd.DataFrame(x, columns=columns) for x in results], axis=1)

    if report:
        prefix = f"INST_" if instant else ""
        metrics_df.to_csv(
            f"../results/reports/{prefix}{transformation}_mu{int(mu*100)}_singlerun.csv.gz", 
            index=False, compression='gzip'
        )
        with gzip.open(
            f"../results/reports/{prefix}{transformation}_mu{int(mu*100)}_partitions.json.gz", 
            "wt"
        ) as f:
            json.dump(partitions, f, default=lambda x: int(x))
    else:
        os.makedirs("../results/reports/time/", exist_ok=True)
        prefix = f"INST_" if instant else ""
        metrics_df.to_csv(
            f"../results/reports/{prefix}{transformation}_mu{int(mu*100)}_it{NIT}.csv.gz", 
            index=False, compression='gzip'
        )
        pd.DataFrame(ts).to_csv(
            f"../results/reports/time/{prefix}time_{transformation}_mu{int(mu*100)}_it{NIT}.csv.gz",
            index=False, compression='gzip'
        )

def main():
    """主执行函数"""
    global instant, report
    
    # 参数配置
    if report:
        NIT, TSS, DELTA = 5, 10, .01
        GIT = NIT
    elif instant:
        NIT, TSS, DELTA = 5, 20, 1
        GIT = 2
    else:
        NIT, TSS, DELTA = 5, 10, .01
        GIT = 2

    print(f"Configuration: instant={instant}, report={report}")
    print(f"Parameters: NIT={NIT}, TSS={TSS}, DELTA={DELTA}")

    n_split = 3
    mu = 0.2

    # 确保输出目录存在
    os.makedirs("../results/reports/", exist_ok=True)

    # 处理每种变换类型
    for transformation in TRANSFORMATIONS:
        print(f"\nStarting processing for {transformation}...")
        results, partitions, ts = process_results(transformation, NIT, GIT, mu, TSS)
        save_results(transformation, mu, NIT, results, partitions, ts)
        print(f"Completed processing for {transformation}")

if __name__ == '__main__':
    freeze_support()  # Windows多进程必需
    main()