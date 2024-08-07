import torch as torch
import numpy as np

'''
Samples an image using Generative Accumulation of Photons (GAP) based on an initial photon image.
If the initial photon image contains only zeros the model samples from scratch. 
If it contains photon numbers, the model performs diversity denoising.

        Parameters:
                input_image (torch tensor): the initial photon image, containing integers (batch, channel, y, x).  
                model: the network used to predict the next phton location.
                max_photons (int): stop sampling when image contains more photons. 
                max_its (int): stop sampling after max_its iterations. 
                max_psnr (float): stop sampling when pseudo PSNR is larger max_psnr
                save_every_n (int): store and return images at every nth step. 
                augment (bool): use 8-fold data augmentation (default is False) 
                beta (float): photon number is increased exponentially by factor beta in each step.
        Returns:
                denoised (numpy array): denoised image at the end of that sampling process.
                photons (numpy array): photon image at the end of the sampling process.
                stack (list): list of numpy arrays containing intermediate results.
                i (int) number of executed iterations.
'''
def stats(img):
    print(f'\nMax : {img.max()}\nMin : {img.min()}\nMean : {img.mean()}\nSum : {img.sum()}\n')


def sample_image(input_image,
                 model,
                 max_photons = None,
                 max_its = 500000,
                 max_psnr = -15,
                 save_every_n = 5,
                 beta = 0.1,
                 channels = 1,
                 use_mask = False,
                 const_photon = None,
                ):

    start = input_image[:,-channels:, :, :].clone()
    cond_input = input_image[:,:-channels, :, :].clone()
    mask = cond_input[:, :1, :, :].clone()
    photons = start
    photnum = 1

    denoised = None
    stack = []
    
    sumDenoised = start
    region = 64


    for i in range(max_its):
   
        # compute the pseudo PSNR
        psnr = np.log10( photons.mean().item() + 1e-50) * 10
        psnr = max(-40, psnr)
            
        if (max_photons is not None) and (photons.sum().item() > max_photons):
            break
            
        if psnr > max_psnr:
            break
        print(f'cond_input shape : {cond_input.shape}\nphotons shape: {photons.shape}\n\n\n')
        input = torch.cat((cond_input, photons),1)
        denoised = model(input).detach()
        denoised = denoised - denoised.max()
        denoised = torch.exp(denoised)   
        denoised = denoised / (denoised.sum(dim=(-1,-2,-3), keepdim = True))
        print(denoised.shape)
        stats(denoised)
        stats(photons)
        

        # here we save an image into our stack
        if (save_every_n is not None) and (i%save_every_n == 0):  

            imgsave = denoised[0,0,:,...].detach().cpu()
            imgsave = imgsave/imgsave.max()
            photsave = photons[0,0,:,...].detach().cpu()
            photsave = photsave / max(photsave.max(),1)      
            combi = torch.cat((photsave,imgsave),1)
            stack.append(combi.numpy())

        # increase photon number    
        photnum = max(beta* photons.sum(),1)
        
        # draw new photons
        if const_photon:
            new_photons = torch.poisson(denoised*(const_photon)) 
        else:
            new_photons = torch.poisson(denoised*(photnum)) 
        
        # add new photons
        if use_mask:
            photons = (photons + new_photons) * mask
        else: 
            photons = photons + new_photons
        print(photons.shape) 
        
    
    return denoised[...].detach().cpu().numpy(), photons[...].detach().cpu().numpy(), stack, i