import numpy as np
import torch

class PSNRSample():
    def __init__(self, minPSNR, maxPSNR):
        self.minPSNR = minPSNR
        self.maxPSNR = maxPSNR
    
    def sample(self,  data, value):
        data = torch.from_numpy((data.numpy()).astype(np.int32)) * 50
        uniform = value*(self.maxPSNR-self.minPSNR)+self.minPSNR

        level = (10**(uniform/10.0))/ (data.type(torch.float).mean().item()+1e-5)
        level = min(level, 0.99)
        
        binom = torch.distributions.binomial.Binomial(total_count=data, probs=torch.tensor([level]))
        imgNoise = binom.sample()
        imgNoise = imgNoise.type(torch.float)

        return imgNoise

