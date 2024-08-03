import numpy as np
import torch

def preprocess(inp):
    ''' 
    Preprocess images from dataset for plt.imshow
    '''
    if isinstance(inp, np.ndarray):
        img = inp.copy()
        img/=img.max()
        return img.transpose(1, 2, 0)
    elif torch.is_tensor(inp):
        img = inp.clone()
        img/=img.max()
        return img.permute(1, 2, 0)
    else:
        raise ValueError("Invalid input type")
    

def stats(*inp):
    ''' 
    Print image statistics
    '''
    def print_stats(x):
        print(f'\nShape : {x.shape}\n\nMin : {x.min()}\n\nMax : {x.max()}\n\nSum : {x.sum()}\n\nMean : {x.mean()}\n')
    
    for x in inp:
        print_stats(x)