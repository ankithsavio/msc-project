'''
Modified GAP Architecture for JAX
'''

import jax
import jax.numpy as jnp
import flax.linen as nn

class Upsample(nn.Module):
    method : str = 'bilinear'
    scale : int = 2
    dtype : jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape
        shape = (B, H*self.scale, W*self.scale, C)
        return jax.image.resize(image = x, shape = shape, method = self.method).astype(self.dtype)


class DownConv(nn.Module):
    out_channels : int
    pooling : bool = True
    dtype : jnp.dtype = jnp.float32

    def setup(self):
        self.conv1 = nn.Conv(features= self.out_channels,
                             kernel_size= (3, 3), 
                             strides= (1, 1),
                             padding= (1, 1),
                             use_bias= True,
                             name= 'DownConv-1',
                             kernel_init= nn.initializers.xavier_normal(dtype = self.dtype),
                             bias_init= nn.initializers.constant(value= 0, dtype = self.dtype))
        
        self.conv2 = nn.Conv(features= self.out_channels,
                             kernel_size= (3, 3), 
                             strides= (1, 1),
                             padding= (1, 1),
                             use_bias= True,
                             name= 'DownConv-2',
                             kernel_init= nn.initializers.xavier_normal(dtype = self.dtype),
                             bias_init= nn.initializers.constant(value= 0, dtype = self.dtype))
        
        self.conv3 = nn.Conv(features= self.out_channels,
                             kernel_size= (3, 3), 
                             strides= (1, 1),
                             padding= (1, 1),
                             use_bias= True,
                             name= 'DownConv-3',
                             kernel_init= nn.initializers.xavier_normal(dtype = self.dtype),
                             bias_init= nn.initializers.constant(value= 0, dtype = self.dtype))

    def __call__(self, x):
        
        xskip = self.conv1(x)
        x = self.conv2(xskip)
        x = nn.relu(x)
        x = self.conv3(x) + xskip
        x = nn.relu(x)
        before_pool = x

        if self.pooling:
            x = nn.max_pool(x,
                            window_shape= (2, 2),
                            strides= (2, 2)).astype(self.dtype)
            
        return x, before_pool


class UpConv(nn.Module):
    out_channels : int
    merge_mode : str = 'concat'
    up_mode : str = 'transpose'
    dtype : jnp.dtype = jnp.float32


    def setup(self):
        if self.up_mode == 'transpose':
            self.upconv = nn.ConvTranspose(features= self.out_channels,
                                           kernel_size= (2, 2),
                                           strides= (2, 2),
                                           name = 'ConvTranspose',
                                           kernel_init= nn.initializers.xavier_normal(dtype = self.dtype),
                                           bias_init= nn.initializers.constant(value= 0, dtype = self.dtype))
        else:
            self.upconv = nn.Sequential([Upsample(method = 'bilinear', scale = 2),
                                         nn.Conv(features= self.out_channels,  
                                                 kernel_size= (1, 1),
                                                 strides= (1, 1),
                                                 name = 'Upsample -> Conv1x1',
                                                 kernel_init= nn.initializers.xavier_normal(dtype = self.dtype),
                                                 bias_init= nn.initializers.constant(value=0, dtype = self.dtype))])


        self.conv1 = nn.Conv(features= self.out_channels,
                             kernel_size= (3, 3), 
                             strides= (1, 1),
                             padding= (1, 1),
                             use_bias= True,
                             name= 'UpConv-1',
                             kernel_init= nn.initializers.xavier_normal(dtype = self.dtype),
                             bias_init= nn.initializers.constant(value= 0, dtype = self.dtype)) 
        
        self.conv2 = nn.Conv(features= self.out_channels,
                             kernel_size= (3, 3), 
                             strides= (1, 1),
                             padding= (1, 1),
                             use_bias= True,
                             name= 'UpConv-2',
                             kernel_init= nn.initializers.xavier_normal(dtype = self.dtype),
                             bias_init= nn.initializers.constant(value= 0, dtype = self.dtype))
        
        self.conv3 = nn.Conv(features= self.out_channels,
                             kernel_size= (3, 3), 
                             strides= (1, 1),
                             padding= (1, 1),
                             use_bias= True,
                             name= 'UpConv-3',
                             kernel_init= nn.initializers.xavier_normal(dtype = self.dtype),
                             bias_init= nn.initializers.constant(value= 0, dtype = self.dtype))

    def __call__(self, from_down, from_up):
        
        from_up = self.upconv(from_up) 
        if self.merge_mode == 'concat':
            x = jnp.concatenate([from_up, from_down], axis = -1) 
        else:
            x = from_up + from_down

        xskip = self.conv1(x) 
        x = self.conv2(xskip)
        x = nn.relu(x)
        x = self.conv3(x) + xskip
        x = nn.relu(x)

        return x


class UN(nn.Module):
    levels : int
    channels : int = 3
    depth : int = 5
    start_filts : int = 64
    up_mode : str = 'transpose'
    merge_mode : str = 'add'
    dtype : jnp.dtype = jnp.float32

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
            # ins = self.channels * self.levels if i == 0 else outs
            outs = self.start_filts*(2**i)
            pooling = True if i < self.depth-1 else False

            module = DownConv(outs, pooling=pooling, dtype = self.dtype)
            down_convs.append(module)
        self.down_convs = down_convs

        for i in range(self.depth-1):
            # ins = outs
            outs = outs // 2
            module = UpConv(outs, up_mode=self.up_mode,merge_mode=self.merge_mode, dtype = self.dtype)
            up_convs.append(module)
        self.up_convs = up_convs

        self.conv_final = nn.Conv(features= self.channels,  
                                  kernel_size= (1, 1),
                                  strides= (1, 1),
                                  name = 'Final_Conv1x1',
                                  kernel_init= nn.initializers.xavier_normal(dtype = self.dtype),
                                  bias_init= nn.initializers.constant(value=0, dtype = self.dtype))



    def __call__(self, x):
        
        stack = None
        factor = 10.0
        for i in range(self.levels):
            scale = x.copy()*(factor**(-i))
            scale = jnp.sin(scale)
            if stack is None:
                stack = scale
            else:
                stack = jnp.concatenate((stack,scale),axis = -1)
        
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