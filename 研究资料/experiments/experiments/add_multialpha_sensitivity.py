import sys
# 将项目根目录添加到Python路径的最前面
sys.path.insert(0, "D:\srtq项目\Dynamic_CommunityDetection_Benchmark-main")
import os
import pandas as pd
from multiprocessing import Pool, cpu_count, freeze_support
from src.simulation import Run_MultiAlpha
from tqdm import tqdm
import numpy as np

# 设置项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 全局配置
TRANSFORMATIONS = ['birth', 'fragment', 'split', 'merge',
                  'add_nodes', 'intermittence_nodes', 'switch',
                  'break_up_communities', 'remove_nodes']

NIT, TSS, DELTA = 5, 5, .01
GIT = 2  # Number of NIT with the same graph
n_split = 3
mu = .2
alphas = [1e-2, 0.05, 0.1, 0.9]

# 确保输出目录存在
os.makedirs('../results/reports/Alpha_sensitivity/', exist_ok=True)
os.makedirs('../results/reports/Alpha_sensitivity/time/', exist_ok=True)

def wrapper_func(args):
    """
    可序列化的包装函数
    参数需要包含所有必要的信息，不能依赖全局变量
    """
    Num_Graph, seed, mu, timesteps, transformation, alphas = args
    run = Run_MultiAlpha(Num_Graph, seed, mu, timesteps, transformation, alphas=alphas)
    return run

def process_transformation(transformation):
    """处理单个变换类型的实验"""
    iterable = [(i//GIT, i%GIT, mu, TSS, transformation, alphas) 
               for i in range(NIT)]
    results = []
    ts = {}
    
    with Pool(cpu_count()) as pool:
        with tqdm(total=len(iterable), desc=f'Processing {transformation}') as pbar:
            for res, t in pool.imap(wrapper_func, iterable):
                # 收集时间统计信息
                for key in t:
                    ts.setdefault(key, []).append(t[key])
                pbar.update(1)
                results.append(res)
    
    # 保存指标结果
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
    
    metrics = pd.concat([pd.DataFrame(x, columns=columns) for x in results], axis=1)
    metrics.to_csv(
        f'../results/reports/Alpha_sensitivity/multialpha_add_{transformation}_mu{int(mu*100)}_it{NIT}.csv.gz',
        index=False, compression='gzip'
    )
    
    # 保存时间统计
    pd.DataFrame(ts).to_csv(
        f'../results/reports/Alpha_sensitivity/time/time_add_{transformation}_mu{int(mu*100)}_it{NIT}.csv.gz',
        index=False, compression='gzip'
    )

def main():
    """主执行函数"""
    for transformation in TRANSFORMATIONS:
        try:
            process_transformation(transformation)
            print(f"Completed processing for {transformation}")
        except Exception as e:
            print(f"Error processing {transformation}: {str(e)}")
            traceback.print_exc()

if __name__ == '__main__':
    freeze_support()  # Windows多进程必需
    main()