import numpy as np
import torch
import cv2

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

def stack_video(stack, filename):
    if len(stack[0].shape) == 3:
        _, H, W = stack[0].shape
        framerate = 5
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        out = cv2.VideoWriter(filename, fourcc, framerate, (W, H))
        for frame in stack:
            out.write(cv2.cvtColor((frame.transpose(1, 2, 0) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        out.release()
    elif len(stack[0].shape) == 2:
        H, W = stack[0].shape
        framerate = 3
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        out = cv2.VideoWriter(filename, fourcc, framerate, (W, H), isColor= True)
        for frame in stack:
            frame_normalized = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            frame_color = cv2.applyColorMap(frame_normalized, cv2.COLORMAP_INFERNO)
            out.write(frame_color)
        out.release()
    else:
        raise ValueError('stacked images have dim > 3')
    print(f'Video saved at {filename} + v4')