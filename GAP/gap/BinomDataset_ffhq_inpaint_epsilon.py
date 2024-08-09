import os
import torch as torch
import numpy as np
from torchvision import transforms
from PIL import Image
from pathlib import Path
from typing import Union
from tasks import inpainting

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
        scale = 50,
        augment = True,
        maxProb = 0.99,
    ):
        self.crop = transforms.RandomCrop(windowSize)
        self.flipH = transforms.RandomHorizontalFlip()
        self.flipV = transforms.RandomVerticalFlip()
        self.minPSNR = minPSNR
        self.maxPSNR = maxPSNR
        self.windowSize = windowSize
        self.maxProb = maxProb
        self.virtSize = virtSize
        self.augment = augment
        self.scale = scale
        self.mask_gen = inpainting(imgsize = 512, masksize = 256)
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
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        idx_ = idx 
        if self.virtSize is not None:
            idx_ = np.random.randint(len(self.samples)) # get random sample
        data = self.loader(self.samples[idx_])
        data = np.array(data).astype(np.int32).transpose(2, 0 ,1)
        img = torch.from_numpy(data) * self.scale
        
        
        uniform = np.random.rand()*(self.maxPSNR-self.minPSNR)+self.minPSNR
        # print(f'clean uniform: {uniform}')
        level = (10**(uniform/10.0))/ (img.type(torch.float).mean().item()+1e-5)
        level = min(level, self.maxProb)

        gt = img.clone().type(torch.float)
        gt = gt / (gt.mean()+1e-8)

        binom_target = torch.distributions.binomial.Binomial(total_count=img, probs=torch.tensor([level]))
        imgTarget = binom_target.sample()
        norm = imgTarget.clone()
        norm = norm / (norm.sum(dim=(-1,-2,-3), keepdim = True))
        photnum = max(0.99 * imgTarget.sum(),1) 
        img = (norm*(photnum)).to(torch.int32)

        level = (10**(uniform/10.0))/ (img.type(torch.float).mean().item()+1e-5)
        level = min(level, self.maxProb)
        
        binom_noise = torch.distributions.binomial.Binomial(total_count=img, probs=torch.tensor([0.90]))
        imgNoise = binom_noise.sample()
        
        imgTarget = imgTarget.type(torch.float)
        imgNoise = imgNoise.type(torch.float)

        mask = self.mask_gen.generate_mask()

        out = torch.cat((imgTarget, mask, gt * (1 - mask) ,imgNoise),dim = 0)
        # print('version 10')
        if np.random.rand()<0.5:
            out = self.flipH(out)
        return out