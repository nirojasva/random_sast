import os
import numpy as np
import pandas as pd
from operator import itemgetter
import matplotlib.pyplot as plt
from scipy.io.arff import loadarff
from sast import znormalize_array 
from sklearn.model_selection import train_test_split


def load_arff_2_dataframe(fname):
    data = loadarff(fname)
    data = pd.DataFrame(data[0])
    data = data.astype(np.float64)
    return data
    
def load_dataset(ds_folder, ds_name, shuffle=False ):
    # dataset path
    ds_path = os.path.join(ds_folder, ds_name)
    
    # load train and test set from arff
    train_ds = load_arff_2_dataframe(os.path.join(ds_path, f'{ds_name}_TRAIN.arff'))
    test_ds = load_arff_2_dataframe(os.path.join(ds_path, f'{ds_name}_TEST.arff'))
    if shuffle:
        ntrain=train_ds.shape[0]
        ntest=test_ds.shape[0]
        ds_concat=pd.concat([train_ds, test_ds],ignore_index=True)
        
        #print("train_ds.shape",str(train_ds.shape))
        #print("test_ds.shape",str(test_ds.shape))
        #print("ds_concat.shape",str(ds_concat.shape))
        #np.random.shuffle(ds_concat)
        train_ds, test_ds=train_test_split(ds_concat,test_size=ntest, train_size=ntrain,shuffle=True)

    return train_ds, test_ds

def format_dataset(data, shuffle=True):
    X = data.values.copy()
    if shuffle:
        np.random.shuffle(X)
    X, y = X[:, :-1], X[:, -1]

    return X, y.astype(int)

def plot(h, h_val, title):
    plt.title(title)
    plt.plot(h, label='Train')
    plt.plot(h_val, label='Validation')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()

def plot_most_important_features(kernels, scores, dilations=[], limit = 4, scale_color=True):
    if len(dilations)==0:
        dilations=[1]*len(kernels)
    features = zip(kernels, scores, dilations)
    sorted_features = sorted(features, key=lambda sublist: abs(sublist[1]), reverse=True)
    #sorted_features =sorted(features, key=itemgetter(1), reverse=True)
    for l, sf in enumerate(sorted_features[:limit]):
        kernel, score, dilation = sf
        kernel = kernel[~np.isnan(kernel)]
        kernel_d=[]
        for value in kernel:
            for j in range(dilation):
                if j==0:
                    kernel_d.append(value)        
                else:
                    kernel_d.append(None)
        kernel_d=np.array(kernel_d)
        dmask = np.isfinite(kernel_d.astype(np.double))
        shp_range=np.arange(kernel_d.size)
        if scale_color:
            #plt.scatter(range(kernel_d.size), kernel_d, linewidth=1.5)
            plt.plot(shp_range[dmask], kernel_d[dmask], linewidth=50*score, label="feature"+str(l+1)+": "+"d="+str(dilation)+" coef="+str(f'{score:.5}'), linestyle='-', marker='o')
        else:
            #plt.scatter(range(kernel_d.size), kernel_d, linewidth=1.5)
            plt.plot(shp_range[dmask], kernel_d[dmask], label="feature"+str(l+1)+": "+"d="+str(dilation)+" coef="+str(f'{score:.5}'), linestyle='-', marker='o')
    plt.legend()
    plt.show()
    
def plot_most_important_feature_on_ts(  features, scores,set_ts,labels, dilations=[],type_features=[], offset=0, limit = 5, fname=None, znormalized=False):
    '''Plot the most important features on ts'''
    print('Plot the most important features on ts')
    if len(dilations)==0:
        dilations=[1]*len(features)
    
    if len(type_features)==0:
        type_features=["min"]*len(features)

    
    features = zip(features, scores, dilations,type_features, set_ts, labels)
    
    
    sorted_features = sorted(features, key=lambda sublist: abs(sublist[1]), reverse=True)
    max_ = min(limit, len(sorted_features) - offset)    
    #sorted_features = sorted(features, key=itemgetter(1), reverse=True)
    
    
    
    
    
    if max_ <= 0:
        print('Nothing to plot')
        return        
    
    
    for s, l in enumerate(np.unique(labels)):
        fig, axes = plt.subplots(1, max_, sharey=True, figsize=(3*max_, 3), tight_layout=True, clear=True)
        
        print(f"s+1 {s+1} max_ {max_} label {l}")            
        for f in range(max_):
            
            kernel, score, dilation, type_f, ts, label = sorted_features[f+offset]
            print(f"ts {ts} kernel {kernel}") 
            if label!=l:
                *_, ts, _=list(filter(lambda x: x[5] == l,sorted_features))[0]
                print(f"label {label} l {l}") 
                print(f"diff ts {ts} kernel {kernel}") 
                
                

            kernel_d=[]
            for value in kernel:
                for j in range(dilation):
                    if j==0:
                        kernel_d.append(value)        
                    else:
                        kernel_d.append(None)
            kernel_d=np.array(kernel_d)
            
            if znormalized:
                kernel_d = znormalize_array(kernel_d)
                ts = znormalize_array(ts)            
            
            d_best = np.inf

            
            for i in range(ts.size - kernel_d.size + 1):

                d=0
                for k, value in enumerate(kernel_d):
                   

                    
                    if kernel_d[k] is not None:
                        d = d+(ts[i:i+kernel_d.size][k] - kernel_d[k])**2
                    else:
                        break
                if d < d_best:
                    d_best = d
                    start_pos = i
            dmask = np.isfinite(kernel_d.astype(np.double))
            shp_range=np.arange(start_pos, start_pos + kernel_d.size)
            axes[f].plot(shp_range[dmask], kernel_d[dmask], linewidth=4,color="darkred", linestyle='-', marker='o')
            axes[f].plot(range(ts.size), ts, linewidth=2,color='darkorange')
            axes[f].set_title(f'feature: {f+1+offset}, type: {type_f}')
            #print('gph shapelet values:',str(f+1),' start_pos:',start_pos,' shape:', kernel_d.size,' dilation:', str(dilation))
            #print(" shapelet:", kernel_d )

        fig.suptitle(f'Ground truth class: {l}', fontsize=15)

        plt.show();

        if fname is not None:
            fig.savefig(fname)


def plot_kernel_generators(sastClf):
    ''' This herper function is used to plot the reference time series used by a SAST'''
    for c, ts in sastClf.kernels_generators_.items():
        print("c:", c ," ts:", ts)
        plt.figure(figsize=(5, 3))
        plt.title(f'Class {c}')
        for t in ts:
            plt.plot(t)
        plt.show()