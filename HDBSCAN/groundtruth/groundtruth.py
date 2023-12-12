import numpy as np
import pandas as pd
import hdbscan
import sys
import time

np.random.seed(0)



def set_colors(rows,N):
    colors = np.zeros(N)
    
    cores = rows.shape[0]
    sample = rows.shape[1]
    
    for k in range(cores):
        for i in range(sample):
            if rows[k,i]:
                colors[i] = k + 1
    return colors

#Concatena as distribuições probabilísticas
def make_sample(data):
    
    
    sample = data[0]
    for i in range(1,len(data)):
        sample = np.concatenate((sample,data[i]))
    
    
    return sample

def make_colors(colors):
    
    sample = colors[0]
    max_c = max(colors[0])
    
    for i in range(1,len(colors)):
        colors[i] = colors[i] + max_c + 1   
        max_c = max(colors[i])
        sample = np.concatenate((sample,colors[i]))
    return sample

def biclust_dataset(N,dim):
    #Building make_bicluster dataset
    from sklearn.datasets import make_biclusters
    X0, rows,_ = make_biclusters(
    shape=(N, dim), n_clusters=2, noise=.4,minval=-12,maxval=10, shuffle=False, random_state=10)
    y0 = set_colors(rows,N) #Colors
    
    return X0,y0

def blobs_dataset(N,dim):
    #Building make_blobs dataset
    from sklearn.datasets import make_blobs
    X1, y1 = make_blobs(n_samples=N, centers=5, n_features=dim,
                   random_state=10,cluster_std=.6)
    return X1,y1

def normalize_dataset(data):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    norm_data = scaler.fit_transform(data)
    
    return norm_data

def get_artifical_db(N,dim):
    
    old_n = N
    N = N//2
    
    x0,y0 = biclust_dataset(N,dim)
    
    x1,y1 = blobs_dataset(N,dim)
    
    data = [x0,x1]
    colors = [y0,y1]
    
    sample = make_sample(data)
    col_list = make_colors(colors)
    
    #É preciso normalizar o conjunto de dados, visto que a distância utilizada é a euclidiana
    np.random.shuffle(sample)
    return sample,col_list

def create_dataset(N,dim):
    
    sample,col_list = get_artifical_db(N,dim)
    colors = col_list
    N = sample.shape[0]
    i0 = 0
    for i in range(N//2,len(colors),N):
        
        c_unique = colors[i0:i]
        c_out = colors[i:]
        

      
    return sample.astype(np.float32),col_list



class ClusteringHDBSCAN:

    def __init__(self,args):
        self.min_cluster_size = args['min_clust']
        self.exact = args['exact']

        
    def cluster_lelland(self,data):
    

        self.running_time = time.time()
        self.clusterer = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size, algorithm='prims_kdtree', metric='euclidean' , approx_min_span_tree=True, 
                                            match_reference_implementation=False,cluster_selection_method="eom",
                            allow_single_cluster=False,
                            cluster_selection_epsilon=0.0,
                            max_cluster_size=0)
                                        

        self.clusterer.fit(data)

        self.running_time = time.time() - self.running_time
        
        self.labels = self.clusterer.labels_


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


df_gpu = None

file_name = 'fapesp.csv'

try:   
    df_gpu = pd.read_csv(file_name)
except:
    print("DF_GPU ainda nao existe, logo vai ser criado")
    df_gpu = pd.DataFrame()

index_ = df_gpu.shape[0]


db = create_dataset(int(N),12)[0]

params = {'min_clust':mpts,'exact':False}
clus = ClusteringHDBSCAN(params)
clus.cluster_lelland(db)



info = {}
info['Name'] = dataset
info['Num_Sample'] = N
info['Dim'] = D
info['mpts'] = mpts
info['Iter'] = iter
info['Time'] = clus.running_time

info['ARI'] = 0.0

write_df(df_gpu,index_,info)
df_gpu.to_csv(file_name, index=False)

if int(iter) == 0:
    #Save labels
    np.savetxt(f'/nndescent/GPU_HDBSCAN/experiments/HDBSCAN/groundtruth/groundtruth_{N//1e6}.txt', clus.labels, delimiter='    ')  
