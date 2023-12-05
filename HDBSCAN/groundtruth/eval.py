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
exact = NULL

# Read Approximate FILE
pred = NULL


df_gpu = None

file_name = 'fapesp.csv'

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

info['ARI'] = ARI_val(exact,pred)

write_df(df_gpu,index_,info)
df_gpu.to_csv(file_name, index=False)










    