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
    sorted_features = sorted(features, key=itemgetter(1), reverse=True)
    for l, sf in enumerate(sorted_features[:limit]):
        kernel, score, dilation = sf
        kernel = kernel[~np.isnan(kernel)]
        kernel_d=[]
        for i, value in enumerate(kernel):
            for j in range(dilation):
                if j==0:
                    kernel_d.append(value)        
                else:
                    kernel_d.append(None)
        kernel_d=np.array(kernel_d)
        if scale_color:
            plt.scatter(range(kernel_d.size), kernel_d, linewidth=1.5)
            plt.plot(range(kernel_d.size), kernel_d, linewidth=50*score, label="feature"+str(l+1)+": "+"d="+str(dilation)+" alpha:"+str(f'{score:.5}'))
        else:
            plt.scatter(range(kernel_d.size), kernel_d, linewidth=1.5)
            plt.plot(range(kernel_d.size), kernel_d, label="feature"+str(l+1)+": "+"d="+str(dilation)+" alpha:"+str(f'{score:.5}'))
    plt.legend()
    plt.show()
    
def plot_most_important_feature_on_ts(ts, label, features, scores, dilations=[], offset=0, limit = 5, fname=None, znormalized=True):
    '''Plot the most important features on ts'''
    print('Plot the most important features on ts')
    if len(dilations)==0:
        dilations=[1]*len(features)
    features = zip(features, scores, dilations)
    sorted_features = sorted(features, key=itemgetter(1), reverse=True)
    
    max_ = min(limit, len(sorted_features) - offset)

    if max_ <= 0:
        print('Nothing to plot')
        return
    fig, axes = plt.subplots(1, max_, sharey=True, figsize=(3*max_, 3), tight_layout=True)
    
    for f in range(max_):
        if znormalized:
            kernel, score, dilation = sorted_features[f+offset]
            kernel_d=[]
            for i, value in enumerate(kernel):
                for j in range(dilation):
                    if j==0:
                        kernel_d.append(value)        
                    else:
                        kernel_d.append(None)
            kernel_d=np.array(kernel_d)
            kernel_normalized = znormalize_array(kernel_d)
            d_best = np.inf
            for i in range(ts.size - kernel_d.size):
                ts[i:i+kernel_d.size] = znormalize_array(ts[i:i+kernel_d.size])
                d=0
                for k, value in enumerate(kernel_d):
                    if kernel_d[k] is not None:
                        d = d+(ts[i:i+kernel_d.size][k] - kernel_d[k])**2
                    else:
                        break
                if d < d_best:
                    d_best = d
                    start_pos = i
            axes[f].scatter(range(start_pos, start_pos + kernel_d.size), kernel_d, linewidth=1.5,color="darkred")
            axes[f].plot(range(ts.size), ts, linewidth=2,color='darkorange')
            axes[f].set_title(f'feature: {f+1+offset}')
            print('gph shapelet values:',str(f+1),' start_pos:',start_pos,' shape:', kernel_d.size,' dilation:', str(dilation))
            print(" shapelet:", kernel_d )
            
        else:
            kernel, score, dilation = sorted_features[f+offset]
            kernel_d=[]
            for i, value in enumerate(kernel):
                for j in range(dilation):
                    if j==0:
                        kernel_d.append(value)        
                    else:
                        kernel_d.append(None)
            kernel_d=np.array(kernel_d)

            d_best = np.inf
            for i in range(ts.size - kernel_d.size):
                d=0
                for k, value in enumerate(kernel_d):
                    
                    if kernel_d[k] is not None:
                        d = d+(ts[i:i+kernel_d.size][k] - kernel_d[k])**2
                    else:
                        break

                if d < d_best:
                    d_best = d
                    start_pos = i
            axes[f].scatter(range(start_pos, start_pos + kernel_d.size), kernel_d, linewidth=1.5,color="darkred")
            axes[f].plot(range(ts.size), ts, linewidth=2,color="darkorange")
            axes[f].set_title(f'feature: {f+1+offset}')
            print('gph shapelet values:',str(f+1),' start_pos:',start_pos,' shape:', kernel_d.size,' dilation:', str(dilation))
            print(" shapelet:", kernel_d )
            
    
    fig.suptitle(f'Ground truth class: {label}', fontsize=15)
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