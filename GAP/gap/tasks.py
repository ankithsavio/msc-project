'''
Inpainting 
mask credits : Repaint, Palette, Free-Form.
'''
import numpy as np
import torch
from PIL import Image, ImageDraw
import math

class inpainting:
    def __init__(self, imgsize = 256, masksize = 128):
        self.imgsize = imgsize
        self.masksize = masksize
    
    def randombbox(self):
        imgsize = self.imgsize
        masksize = np.random.randint(low= self.masksize - (self.masksize * 0.1), high= self.masksize + (self.masksize * 0.3))
        maxr = imgsize - masksize
        t = int(np.random.uniform(size = [], low=0, high=maxr).tolist())
        l = int(np.random.uniform(size = [], low=0, high=maxr).tolist())
        h = int(masksize)
        w = int(masksize)
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
    
    def generate_stroke_mask(self):
        min_num_vertex = 4
        max_num_vertex = 12
        mean_angle = 2*math.pi / 5
        angle_range = 2*math.pi / 15
        min_width = 12
        max_width = 40
        def generate_mask(H, W):
            average_radius = math.sqrt(H*H+W*W) / 8
            mask = Image.new('L', (W, H), 0)

            for _ in range(np.random.randint(1, 4)):
                num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
                angle_min = mean_angle - np.random.uniform(0, angle_range)
                angle_max = mean_angle + np.random.uniform(0, angle_range)
                angles = []
                vertex = []
                for i in range(num_vertex):
                    if i % 2 == 0:
                        angles.append(2*math.pi - np.random.uniform(angle_min, angle_max))
                    else:
                        angles.append(np.random.uniform(angle_min, angle_max))

                h, w = mask.size
                vertex.append((int(np.random.randint(0, w)), int(np.random.randint(0, h))))
                for i in range(num_vertex):
                    r = np.clip(
                        np.random.normal(loc=average_radius, scale=average_radius//2),
                        0, 2*average_radius)
                    new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
                    new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
                    vertex.append((int(new_x), int(new_y)))

                draw = ImageDraw.Draw(mask)
                width = int(np.random.uniform(min_width, max_width))
                draw.line(vertex, fill=1, width=width)
                for v in vertex:
                    draw.ellipse((v[0] - width//2,
                                v[1] - width//2,
                                v[0] + width//2,
                                v[1] + width//2),
                                fill=1)

            if np.random.normal() > 0:
                mask.transpose(Image.FLIP_LEFT_RIGHT)
            if np.random.normal() > 0:
                mask.transpose(Image.FLIP_TOP_BOTTOM)
            mask = np.asarray(mask, np.float32)
            mask = np.reshape(mask, (1, H, W, 1))
            return mask
        h, w = [self.imgsize] * 2
        mask = generate_mask(h, w).reshape(([1] + [h, w]))
        return torch.from_numpy(mask)

        
