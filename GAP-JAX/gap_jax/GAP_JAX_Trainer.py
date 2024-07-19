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

class TrainState(train_state.TrainState):
    value: jnp.array = None  # Add an additional attribute to store the metric value

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
    ''' 
    Credits for Trainer Module : https://github.com/phlippe/jax_trainer
    '''
    def __init__(self, root_dir, gradient_clip_val, epochs, dataloader, model: nn.Module, lr = 1e-4, seed = 42, channels = 1):
        super().__init__()
        self.root_dir = root_dir
        self.gradient_clip_val = gradient_clip_val
        self.max_epochs = epochs
        self.img = next(iter(dataloader))[:,:,:,channels:]
        self.learning_rate= lr
        self.seed = seed
        self.channels = channels
        self.platform = jax.local_devices()[0].platform
        self.model = model
        self.logger = SummaryWriter(log_dir= self.root_dir)
        self.earlystop = EarlyStopping(min_delta= 1e-3, patience= 10)
        self.init_training()
        self.init_model()
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
        params = self.model.init(init_rng, self.img)
        self.state = TrainState(step=0,
                                apply_fn=self.model.apply,
                                params=params,
                                tx=None,
                                opt_state=None)
    
    def init_optimizer(self):
        ''' 
        Initializes adam optimizer with gradient clipping and learning rate schedule to reduce on plateau
        '''
        tx = optax.chain(optax.clip(self.gradient_clip_val),
                         optax.adam(learning_rate= self.learning_rate),
                         optax.contrib.reduce_on_plateau(factor = 0.5))
        
        self.state = TrainState.create(apply_fn= self.state.apply_fn,
                                       params= self.state.params,
                                       tx= tx)

    def photonLoss(self, result, target):
        ''' 
        GAP PhotonLoss
        '''
        expEnergy = jnp.exp(result)
        perImage = -jnp.mean(result*target, axis = (-1, -2, -3), keepdims=True)
        perImage += jnp.log(jnp.mean(expEnergy, axis = (-1, -2, -3), keepdims= True))*jnp.mean(target, axis = (-1, -2, -3), keepdims= True)

        return jnp.mean(perImage)
    
    def init_training(self):
        def train_step(state, batch):
            '''
            Computes loss and applies a gradietn step
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
        for batch in tqdm(train_loader, desc='Training', leave=False):
            self.state, loss = self.train_step(self.state, batch)
            avg_loss += loss
        avg_loss /= len(train_loader)
        self.logger.add_scalar('Loss/train ', avg_loss.item(), global_step=epoch)

    def eval_model(self, data_loader):
        ''' 
        Evaluates the model for the given validation dataloader
        '''
        avg_ploss, count = 0, 0
        for batch in data_loader:
            ploss = self.eval_step(self.state, batch)
            avg_ploss += ploss * batch[0].shape[0]
            count += batch[0].shape[0]
        eval_ploss = (avg_ploss / count).item()
        return eval_ploss

    def save_model(self, step=0):
        ''' 
        Save the checkpoints of the model 
        '''
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)
        checkpoints.save_checkpoint(ckpt_dir=self.root_dir,
                                    target=self.state.params,
                                    step=step,
                                    overwrite=True)

    def load_model(self):
        ''' 
        Loads the checkpoints of the model and creates its state
        '''
        state_dict = checkpoints.restore_checkpoint(ckpt_dir=f'{self.root_dir}.ckpt',
                                                    target=self.state.params)
        self.state = TrainState.create(apply_fn=self.state.apply_fn,
                                       params=state_dict,
                                       tx=self.state.tx)

    def checkpoint_exists(self):
        return os.path.isfile(f'{self.root_dir}.ckpt')

    def train_model(self, train_loader, val_loader):
        self.init_optimizer()
        best_eval = 1e6
        for epoch_idx in tqdm(range(1, self.max_epochs +1)):
            self.train_epoch(train_loader, epoch=epoch_idx)
            if epoch_idx % 1 == 0:
                eval_ploss = self.eval_model(val_loader)
                self.logger.add_scalar('Loss/val', eval_ploss, global_step=epoch_idx)
                self.earlystop = self.earlystop.update(eval_ploss)
                if eval_ploss <= best_eval:
                    best_eval = eval_ploss
                    self.save_model(step=epoch_idx)
                if self.earlystop.should_stop:
                    print(f'Met early stopping criteria, breaking at epoch {epoch_idx}')
                    break
            self.logger.flush()