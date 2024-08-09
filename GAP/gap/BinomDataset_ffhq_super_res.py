import os
import torch as torch
import numpy as np
from torchvision import transforms
from PIL import Image
from pathlib import Path
from typing import Union

class BinomDataset(torch.utils.data.Dataset):
    '''
    Returns a BinomDataset that will randomly split an image into input and target using a binomial distribution for each pixel.
            
            Parameters:
                    root (path): path to the 'directory' eg. /../../directory/child_dir/img.png 
                    windowSize (int): size of window to be used in random crops
                    minPSNR (float): minimum pseudo PSNR of sampled data (see supplement)
                    maxPSNR (float): maximum pseudo PSNR of sampled data (see supplement)
                    virtSize (int): virtual size of dataset (default is None, i.e., the real size)
                    augment (bool): use 8-fold data augmentation (default is False) 
                    maxProb (float): the maximum success probability for binomial splitting
            Returns:
                    dataset
    '''
    def __init__(
        self,
        root: Union[str, Path],
        windowSize,
        minPSNR,
        maxPSNR,
        virtSize = None,
        scale = 100,
        augment = True,
        maxProb = 0.99,
        in_size = 64,
        out_size = 512
    ):
        self.flipH = transforms.RandomHorizontalFlip()
        self.minPSNR = minPSNR
        self.maxPSNR = maxPSNR
        self.windowSize = windowSize
        self.maxProb = maxProb
        self.virtSize = virtSize
        self.augment = augment
        self.scale = scale
        self.in_size = in_size
        self.out_size = out_size
        self.samples = self.make_dataset(root = root)
    
    @staticmethod
    def make_dataset(root : Union[str, Path]):
        samples = []
        for fnames in os.listdir(root):
            path = os.path.join(root, fnames)
            samples.append(path)
        return samples


    def loader(self, path : Union[str, Path]) -> Image.Image:
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

    def convert2lowres(self, img, in_size, out_size):
        img = transforms.functional.resize(img = img, size = in_size, interpolation = Image.BICUBIC, antialias = True)
        img = transforms.functional.center_crop(img  = img, output_size = in_size)
        img = transforms.functional.resize(img = img, size = out_size, interpolation = Image.BICUBIC, antialias = True)
        return img
    
    def resize(self, img, out_size):
        img = transforms.functional.resize(img = img, size = out_size, interpolation = Image.BICUBIC, antialias = True)
        img = img - img.min()
        img = torch.from_numpy(img.numpy().astype(np.int32))
        return img

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        idx_ = idx 
        if self.virtSize is not None:
            idx_ = np.random.randint(len(self.samples)) # get random sample
        data = self.loader(self.samples[idx_])
        data = np.array(data).astype(np.int32).transpose(2, 0 ,1)
        img = torch.from_numpy(data) 

        gt = img.clone().type(torch.float)
        sr = self.convert2lowres(img = gt, in_size = self.in_size, out_size = self.out_size)
        sr = sr / (sr.mean()+1e-8)
    
        img = self.resize(img = img, out_size = self.out_size)
        img = img * self.scale
        
        uniform = np.random.rand()*(self.maxPSNR-self.minPSNR)+self.minPSNR

        level = (10**(uniform/10.0))/ (img.type(torch.float).mean().item()+1e-5)
        level = min(level, self.maxProb)

        binom = torch.distributions.binomial.Binomial(total_count=img, probs=torch.tensor([level]))
        imgNoise = binom.sample()
        
        img = (img).type(torch.float)
        img = img / (img.mean()+1e-8)
        
        imgNoise = imgNoise.type(torch.float)

        out = torch.cat((img, sr, imgNoise),dim = 0)
        
        
        if np.random.rand()<0.5:
            out = self.flipH(out)

        return out
