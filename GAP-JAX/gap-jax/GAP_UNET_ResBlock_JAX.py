'''
Modified GAP Architecture for JAX
'''

import jax
import jax.numpy as jnp
import flax.linen as nn

def conv3x3(in_channels, out_channels, stride = 1, padding = 1, bias = True):
    return nn.Conv(features= out_channels,
                   kernel_size= (3, 3), 
                   strides= (stride, stride),
                   padding= padding,
                   use_bias= bias,
                   kernel_init= nn.initializers.xavier_normal(),
                   bias_init= nn.initializers.constant(0))

def conv1x1(in_channels, out_channels):
    return nn.Conv(features= out_channels, 
                   kernel_size= 1,
                   strides= 1,
                   kernel_init= nn.initializers.xavier_normal(),
                   bias_init= nn.initializers.constant(0))

class Upsample(nn.Module):
    method : str = 'bilinear'
    scale : int = 2

    @nn.compact
    def __call__(self, x):
        shape = x.shape
        shape = (shape[0], shape[1]*self.scale, shape[2]*self.scale, shape[3])
        return jax.image.resize(image = x, shape = shape, method = self.method)

def upconv2x2(in_channels, out_channels, mode='transpose'):
    '''
    Upsample not implemented, need to find alternative
    '''
    if mode == 'transpose':
        return nn.ConvTranspose(features= out_channels,
                            kernel_size= (2, 2),
                            strides= (2, 2))
    else:
        return nn.Sequential([Upsample(method = 'bilinear', scale = 2),
                              conv1x1(in_channels, out_channels)])

class DownConv(nn.Module):
    in_channels : int
    out_channels : int
    pooling : bool

    def setup(self):
        self.conv1 = conv3x3(self.in_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)
        self.conv3 = conv3x3(self.out_channels, self.out_channels)

    def __call__(self, x):
        
        xskip = self.conv1(x)
        x = nn.relu(self.conv2(xskip))
        x = nn.relu(self.conv3(x) + xskip)
        before_pool = x

        if self.pooling:
            x = nn.max_pool(x,
                            window_shape= (2, 2),
                            strides= (2, 2))
            
        return x, before_pool


class UpConv(nn.Module):
    in_channels : int
    out_channels : int
    merge_mode : str = 'concat'
    up_mode : str = 'transpose'

    def setup(self):
        self.upconv = upconv2x2(self.in_channels, self.out_channels,mode=self.up_mode)
        self.conv1 = conv3x3(self.out_channels, self.out_channels) ## refine for flax
        self.conv2 = conv3x3(self.out_channels, self.out_channels)
        self.conv3 = conv3x3(self.out_channels, self.out_channels)

    def __call__(self, from_down, from_up):
        
        from_up = self.upconv(from_up) 
        if self.merge_mode == 'concat':
            x = jnp.concatenate([from_up, from_down], axis = -1) # check axis channel is last for jax
        else:
            x = from_up + from_down

        xskip = self.conv1(x) 
        x = nn.relu(self.conv2(xskip))
        x = nn.relu(self.conv3(x) + xskip)

        return x


class UN(nn.Module):
    levels : int
    channels : int = 3
    depth : int = 5
    start_filts : int = 64
    up_mode : str = 'transpose'
    merge_mode : str = 'add'

    def setup(self):
        if self.up_mode not in ('transpose', 'upsample'):
            raise ValueError("\"{}\" is not a valid mode for "
                             "upsampling. Only \"transpose\" and "
                             "\"upsample\" are allowed.".format(self.up_mode))
        
        if self.merge_mode not in ('concat', 'add'):
            raise ValueError("\"{}\" is not a valid mode for"
                             "merging up and down paths. "
                             "Only \"concat\" and "
                             "\"add\" are allowed.".format(self.up_mode))

        if self.up_mode == 'upsample' and self.merge_mode == 'add':
            raise ValueError("up_mode \"upsample\" is incompatible "
                             "with merge_mode \"add\" at the moment "
                             "because it doesn't make sense to use "
                             "nearest neighbour to reduce "
                             "depth channels (by half).")
        
        down_convs = []
        up_convs = []

        for i in range(self.depth):
            ins = self.channels * self.levels if i == 0 else outs
            outs = self.start_filts*(2**i)
            pooling = True if i < self.depth-1 else False

            module = DownConv(ins, outs, pooling=pooling)
            down_convs.append(module)
        self.down_convs = down_convs

        for i in range(self.depth-1):
            ins = outs
            outs = ins // 2
            module = UpConv(ins, outs, up_mode=self.up_mode,merge_mode=self.merge_mode)
            up_convs.append(module)
        self.up_convs = up_convs

        self.conv_final = conv1x1(outs, self.channels)



    def __call__(self, x):
        
        stack = None
        factor = 10.0
        for i in range (self.levels):
            scale = x.copy()*(factor**(-i))
            scale = jnp.sin(scale)
            if stack is None:
                stack = scale
            else:
                stack = jnp.concatenate([stack,scale],axis = -1)
        
        x = stack
        
        encoder_outs = []
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i+2)]
            x = module(before_pool, x)
        
        x = self.conv_final(x)

        return x