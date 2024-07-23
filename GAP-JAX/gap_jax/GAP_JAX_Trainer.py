''' 
    Credits for Trainer Module : https://github.com/phlippe/jax_trainer
'''
import os
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
from flax.training.early_stopping import EarlyStopping
from flax.training import checkpoints
import optax
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
import orbax.checkpoint as ocp
from flax.training import orbax_utils



class TrainState(train_state.TrainState):
    value: jnp.array = None 

    def apply_gradients(self, *, grads, value, **kwargs):
        """
          value: The metric value to be passed to the learning rate scheduler.
        """
        updates, new_opt_state = self.tx.update(
            grads, self.opt_state, self.params, value=value
        )
        new_params = optax.apply_updates(self.params, updates)

        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            value=value,
            **kwargs,
        )


class Trainer:

    def __init__(self, root_dir, gradient_clip_val, epochs, dataloader, model: nn.Module, lr = 1e-4, seed = 42, channels = 1, early_stop = 1000):
        super().__init__()
        self.root_dir = root_dir
        self.gradient_clip_val = gradient_clip_val
        self.max_epochs = epochs
        self.steps_per_epoch = len(dataloader)
        self.dummy_img = jnp.ones(next(iter(dataloader))[-1:,:,:,channels:].shape)
        self.learning_rate= lr
        self.seed = seed
        self.channels = channels
        self.platform = jax.local_devices()[0].platform
        self.model = model
        self.logger = SummaryWriter(log_dir= self.root_dir)
        self.earlystop = EarlyStopping(min_delta= 1e-3, patience= early_stop)
        self.init_model()
        self.init_optimizer()
        self.init_training()
        orbax_checkpointer = ocp.PyTreeCheckpointer()
        options = ocp.CheckpointManagerOptions(max_to_keep=2, create=True)
        self.checkpoint_manager = ocp.CheckpointManager(f'{self.root_dir}orbax/', orbax_checkpointer, options)
        if self.platform == 'gpu': 
            print(f'GPU detected with {jax.process_count()} Device(s).')
        elif self.platform == 'tpu':
            print(f'TPU detected with {jax.process_count()} Device(s).')
        else: 
            print(f'CPU detected, Training will be slow!.')


    def init_model(self):
        ''' 
        Initializes the Unet model for GAP training
        '''
        init_rng = jax.random.key(self.seed)
        init_key, _ = jax.random.split(init_rng)
        print(f'\nUsing Dummy data of shape : {self.dummy_img.shape}')
        params = self.model.init(init_key, self.dummy_img)
        self.state = TrainState(step=0,
                                apply_fn=self.model.apply,
                                params=params,
                                tx=None,
                                opt_state=None)
    
    def init_optimizer(self):
        ''' 
        Initializes adam optimizer with gradient clipping and learning rate schedule to reduce on plateau
        '''
        tx = optax.chain(optax.clip_by_global_norm(self.gradient_clip_val),
                         optax.adam(learning_rate= self.learning_rate),
                         optax.contrib.reduce_on_plateau(factor = 0.5,
                                                         patience= 10,
                                                         accumulation_size= self.steps_per_epoch),
                        )
        
        self.state = TrainState.create(apply_fn= self.state.apply_fn,
                                       params= self.state.params,
                                       tx= tx)

    def photonLoss(self, result, target):
        ''' 
        GAP PhotonLoss
        '''
        expEnergy = jnp.exp(result) 
        perImage = -jnp.mean(result*target, axis = (-2, -3, -1), keepdims=True)
        perImage += jnp.log(jnp.mean(expEnergy, axis = (-2, -3, -1), keepdims= True))*jnp.mean(target, axis = (-2, -3, -1), keepdims= True)

        return jnp.mean(perImage)
    
    def init_training(self):
        def train_step(state, batch):
            '''
            Computes loss and applies a gradient step
            '''
            def compute_loss(params):
                ''' 
                Computes Photon Loss
                '''
                result = state.apply_fn(params, batch[:,:,:,self.channels:])
                loss = self.photonLoss(result, batch[:,:,:,:self.channels])
                return loss

            grad_fn = jax.value_and_grad(compute_loss)
            loss, grads = grad_fn(state.params)

            state = state.apply_gradients(grads = grads, value = loss)
            return state, loss
        
        def eval_step(state, batch):
            ''' 
            Computes Loss
            '''
            result = state.apply_fn(state.params, batch[:,:,:,self.channels:])
            loss = self.photonLoss(result, batch[:,:,:,:self.channels])
            return loss
        
        self.train_step = jax.jit(train_step)
        self.eval_step = jax.jit(eval_step)
    
    def train_epoch(self, train_loader, epoch):
        ''' 
        Trains the model for the given training dataloader
        '''
        avg_loss = 0
        for i, batch in enumerate(tqdm(train_loader, desc='Training', leave=False)):
            self.state, loss = self.train_step(self.state, batch)
            self.logger.add_scalar('Loss/train_per_step ', loss.item(), global_step=i)
            avg_loss += loss
        avg_loss /= len(train_loader)
        print(f'\nTrain Avg Loss: {avg_loss}\n')
        self.logger.add_scalar('Loss/train_per_epoch ', avg_loss.item(), global_step=epoch)

    def eval_model(self, data_loader):
        ''' 
        Evaluates the model for the given validation dataloader
        '''
        avg_ploss = 0
        for batch in data_loader:
            ploss = self.eval_step(self.state, batch)
            avg_ploss += ploss 
        eval_ploss = avg_ploss/len(data_loader)
        return eval_ploss.item()

    def save_model(self, step=0):
        ''' 
        Save the checkpoints of the model 
        '''
        ckpt = {'model': self.state}
        save_args = orbax_utils.save_args_from_target(ckpt)
        self.checkpoint_manager.save(step, ckpt, save_kwargs={'save_args': save_args})

        # if not os.path.exists(self.root_dir):
        #     os.makedirs(self.root_dir)
        # checkpoints.save_checkpoint(ckpt_dir=self.root_dir,
        #                             target=self.state.params,
        #                             step=step,
        #                             overwrite=True)

    def load_model(self):
        ''' 
        Loads the checkpoints of the model and creates its state
        '''
        step = self.checkpoint_manager.latest_step()  
        return self.checkpoint_manager.restore(step)

    def checkpoint_exists(self):
        return os.path.isfile(f'{self.root_dir}.ckpt')

    def train_model(self, train_loader, val_loader):
        best_eval = 1e6
        for epoch_idx in tqdm(range(1, self.max_epochs +1)):
            self.train_epoch(train_loader, epoch=epoch_idx)
            if epoch_idx % 1 == 0:
                eval_ploss = self.eval_model(val_loader)
                self.logger.add_scalar('Loss/val', eval_ploss, global_step=epoch_idx)
                print(f'\nVal Loss: {eval_ploss}\n')

                self.earlystop = self.earlystop.update(eval_ploss)
                if eval_ploss <= best_eval:
                    best_eval = eval_ploss
                    self.save_model(step=epoch_idx)
                if self.earlystop.should_stop:
                    print(f'Met early stopping criteria, breaking at epoch {epoch_idx}')
                    self.logger.flush()
                    break
            self.logger.flush()