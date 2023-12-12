import pandas as pd
import numpy as np
import time
import sys


def ARI_val(exact,pred):

    from sklearn.metrics.cluster import adjusted_rand_score


    ari = adjusted_rand_score(exact,pred)

    return ari


def write_df(df,index,info):

    for i in info.keys():
        if i != 'gpu_res' and i != 'data':
            df.loc[index,i] = info[i]

    return





# Valor de K
mpts = 32

D = 2
N = int(1e4)
dataset = 'standart'
iter = 0

args = sys.argv[1:]



while args:
    a = args.pop(0)
    if a == '-name':      dataset = args.pop(0)
    elif a == '-D':       D = int(args.pop(0)) 
    elif  a == '-N':      N = int(args.pop(0)) 
    elif a == '-iter':    iter = args.pop(0)
    elif a == '-mpts':    mpts = int(args.pop(0)) 
    else:
        print("argument %s unknown" % a, file=sys.stderr)
        sys.exit(1)


# Read EXACT FILE
exact = '/nndescent/GPU_HDBSCAN/HDBSCAN/groundtruth/groundtruth_1.0.txt'
with open(exact) as f:
    lines = f.readlines()

exact_labels = []
for i in lines:
    insert = int(i.split(".")[0])
    exact_labels.append(insert)

# Read Approximate FILE
pred = "/nndescent/GPU_HDBSCAN/HDBSCAN/groundtruth/approximate_result.txt"
with open(pred) as f:
    lines = f.readlines()
pred_labels = []
for i in lines:
    insert = int(i)
    pred_labels.append(insert)

df_gpu = None

file_name = '/nndescent/GPU_HDBSCAN/HDBSCAN/results/fapesp.csv'

try:   
    df_gpu = pd.read_csv(file_name)
except:
    print("DF_GPU ainda nao existe, logo vai ser criado")
    df_gpu = pd.DataFrame()

index_ = df_gpu.shape[0]

info = {}
info['Name'] = dataset
info['Num_Sample'] = N
info['Dim'] = D
info['mpts'] = mpts
info['Iter'] = iter
info['Time'] = 0

info['ARI'] = ARI_val(exact_labels,pred_labels)

write_df(df_gpu,index_,info)
df_gpu.to_csv(file_name, index=False)










    
