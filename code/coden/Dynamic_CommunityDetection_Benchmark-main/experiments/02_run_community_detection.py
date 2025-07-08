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

if report:                          #完整报告模式（轻量级测试）
    NIT, TSS, DELTA =50, 150, .01
    GIT=NIT
elif instant:                       #瞬时突变模式（压力测试）
    NIT, TSS, DELTA =1000, 20, 1
    GIT=10
else:                               #渐进演化模式（常规分析）
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

    iterable = [(i//GIT ,i%GIT,mu, TSS, transformation) for i in range(NIT)] #参数矩阵构建  前两个参数分别是实验组编号和组内实验编号
        
    results=[]      #存储各实验的指标数据（ARI/模块度等），用于后续生成CSV统计文件
    partitions={}   #存储社区标签（report模式专用），保存完整的社区演化轨迹
    ts={}           #存储算法耗时数据（非report模式专用），记录算法计算效率
    # wrapper_func(iterable[0])
    with Pool(min(cpu_count(),NIT)) as pool:    #多进程任务执行
        with tqdm(total=len(iterable),desc=f'PRC COUNT {transformation}') as pbar:  # Create a tqdm progress bar 创建进度条
            for i,res in enumerate(pool.imap( wrapper_func, iterable)):
                if report:  #报告模式数据处理
                    metrics,part,y_true_final,y_true_init=res
                    partitions[i]={'y_true_final':y_true_final, #真实社区标签记录
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
                            ]):             #这应该是RunCommunityDetection集成的9种算法，运行run的时候会对这9种算法进行评估
                                            #9种算法应该分别是“静态社区检测基准算法、带自适应阈值的标签传播、迭代式标签扩展、增量更新算法
                                            # 增量+标签传播混合、基于历史轨迹预测、无回溯的演化算法、非回溯与标签传播混合、默认动态算法”
                        partitions[i][pred]={step:part[step][j] for step in range(TSS)} #各算法预测标签记录
                    
                else:       #非报告模式耗时记录
                    metrics,t=res
                    
                    #,y_true_final,y_true_init,t
                
                    for key in t:
                        if not key in ts: ts[key]=[]
                        ts[key].append(t[key]) #累计所有实验的耗时数据
                    pbar.set_description(transformation+' PRC COUNT:\n'+', '.join([f'{key}:{np.mean(ts[key])/(TSS):.2f}' for key in ts])+'\n')
                    #更新进度条描述，显示各算法平均单步耗时
                pbar.update(1)                  
                results.append(metrics) #收集当前实验指标
                                        #metrics：包含ARI、模块度等9x3=27个指标（9算法x3类指标）
    
                
                    
                
                
    print("SAVING") #提示用户进入数据持久化阶段
    # Arrange metrics

    #指标列，9个初始状态的ARI（初始时间步的算法准确性）,9个最终状态的ARI（最终时间步的算法准确性）,9个模块度指标（各算法检测结果的模块度分数）
    columns = ['ari_base', 'ari_alei', 'ari', 'ari_lei',  'ari_i','ari_ilei', 'ari_e', 'ari_ne', 'ari_nelei',
               'ari_init_base', 'ari_init_alei', 'ari_init', 'ari_init_lei', 'ari_init_i','ari_init_ilei', 'ari_init_e', 'ari_init_ne','ari_init_nelei',
        'ari_fin_base', 'ari_fin_alei', 'ari_fin', 'ari_fin_lei', 'ari_fin_i', 'ari_fin_ilei', 'ari_fin_e', 'ari_fin_ne', 'ari_fin_nelei',
               'modularity_base', 'modularity_alei', 'modularity', 'modularity_lei', 'modularity_i' ,'modularity_ilei', 'modularity_e', 'modularity_ne', 'modularity_nelei',
              ]
    #数据整合成DataFrame
    metrics = [pd.DataFrame(x, columns=columns) for x in results]
    metrics = pd.concat(metrics, axis=1)
   
    # Save metrics

    # 根据是是否是报告模式，是否是突变模式进行存储
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
        
    
                

