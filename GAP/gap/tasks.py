'''
Inpainting 
mask credits : Repaint, Palette.
'''
import numpy as np
import torch

class inpainting:
    def __init__(self, imgsize = 256, masksize = 128):
        self.imgsize = imgsize
        self.masksize = masksize
    
    def randombbox(self):
        imgsize = self.imgsize
        maxr = imgsize - self.masksize
        t = int(np.random.uniform(size = [], low=0, high=maxr).tolist())
        l = int(np.random.uniform(size = [], low=0, high=maxr).tolist())
        h = int(self.masksize)
        w = int(self.masksize)
        return (t, l, h, w)
    
    def generate_mask(self):
        bbox = self.randombbox()
        imgsize = self.imgsize
        mask = np.zeros((1, imgsize, imgsize), np.float32)
        delta = 15
        h = np.random.randint(delta)
        w = np.random.randint(delta)
        mask[:, bbox[0]+h:bbox[0]+bbox[2]-h,          
                bbox[1]+w:bbox[1]+bbox[3]-w] = 1.
        return torch.from_numpy(mask)