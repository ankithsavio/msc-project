import torch as torch
import numpy as np
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from pathlib import Path
from typing import Any, Callable, Optional, Union


class DataFolder(ImageFolder):
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
        minPSNR,
        maxPSNR,
        virtSize = None,
        scale = 50,
        maxProb = 0.99,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        allow_empty: bool = False,
    ):
        self.virtSize = virtSize
        self.minPSNR = minPSNR
        self.maxPSNR = maxPSNR
        self.scale = scale
        self.maxProb = maxProb
        super().__init__(root, transform, target_transform, loader, is_valid_file, allow_empty)
        
    def __getitem__(self, idx):
        idx_ = idx 
        if self.virtSize is not None:
            idx_ = np.random.randint(super().__len__()) # get random sample
        data, _ = super().__getitem__(index = idx_)
        data = np.array(data).astype(np.int32).transpose(2, 0 ,1)
        img = torch.from_numpy(data)
        gt = (img.clone()).type(torch.float)
        img = img * self.scale
        uniform = np.random.rand()*(self.maxPSNR-self.minPSNR)+self.minPSNR

        level = (10**(uniform/10.0))/ (img.type(torch.float).mean().item()+1e-5)
        level = min(level, self.maxProb)

        binom = torch.distributions.binomial.Binomial(total_count=img, probs=torch.tensor([level]))
        imgNoise = binom.sample()
        gt = (gt / (gt.mean() + 1e-9))
        img = (img - imgNoise).type(torch.float)
        img = img / (img.mean()+1e-8)
        
        imgNoise = imgNoise.type(torch.float)
        return gt, img, imgNoise
