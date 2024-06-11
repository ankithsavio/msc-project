import torch as torch
import numpy as np
from torchvision import transforms
from tifffile import imread

class BinomDatasetv2(torch.utils.data.Dataset):
    '''
    Returns a BinomDataset that will randomly split an image into input and target using a binomial distribution for each pixel.
            
            Parameters:
                    # data (numpy array): a 3D numpy array (image_index, y, x) with integer photon counts. 
                    data (List): list contatining the paths to variable size tiff images
                    windowSize (int): size of window to be used in random crops
                    minPSNR (float): minimum pseudo PSNR of sampled data (see supplement)
                    maxPSNR (float): maximum pseudo PSNR of sampled data (see supplement)
                    virtSize (int): virtual size of dataset (default is None, i.e., the real size)
                    augment (bool): use 8-fold data augmentation (default is False) 
                    maxProb (float): the maximum success probability for binomial splitting
            Returns:
                    dataset
    '''
    def __init__(self, data, windowSize, minPSNR, maxPSNR, virtSize=None, augment = True, maxProb = 0.99, channelSize = None):
        # self.data = torch.from_numpy(data.astype(np.int32))
        self.data = data
        self.crop = transforms.RandomCrop(windowSize)
        self.flipH = transforms.RandomHorizontalFlip()
        self.flipV = transforms.RandomVerticalFlip()
        self.minPSNR = minPSNR
        self.maxPSNR = maxPSNR
        self.windowSize = windowSize
        self.maxProb = maxProb
        # self.std = data.std()
        self.virtSize = virtSize
        self.augment = augment
        self.channelSize = channelSize
    
    def __len__(self): 
        if self.virtSize is not None:
            return self.virtSize
        else:
            # return self.data.shape[0]
            return len(self.data)
    
    def __getitem__(self, idx):
        idx_ = idx 
        if self.virtSize is not None:
            # idx_ = np.random.randint(self.data.shape[0]) # get random sample
            idx_ = np.random.randint(len(self.data))
        
        if self.channelSize is not None:
            idx2_ = np.random.randint(self.channelSize)
        # img = self.crop(self.data[idx_]) # crop the sample 

        img = self.crop(torch.from_numpy(imread(self.data[idx_]).astype(np.int32))[idx2_])
        
        
        uniform = np.random.rand()*(self.maxPSNR-self.minPSNR)+self.minPSNR
    
        
        level = (10**(uniform/10.0))/ (img.type(torch.float).mean().item()+1e-5)
        level = min(level, self.maxProb)

        
        binom = torch.distributions.binomial.Binomial(total_count=img, probs=torch.tensor([level]))
        imgNoise = binom.sample()
        
        img = (img - imgNoise)[None,...].type(torch.float)
        img = img / (img.mean()+1e-8)
        
        imgNoise = imgNoise[None,...].type(torch.float)
        out = torch.cat((img, imgNoise),dim = 0)
        
        if not self.augment:
            return out
        else:    
            if np.random.rand()<0.5:
                out = torch.transpose(out,-1,-2)

            return self.flipV(self.flipH(out))
