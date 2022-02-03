import os
import jax
import jax.numpy as jnp              

from flax import optim
from flax import linen as nn           
from flax.training import train_state
from flax.training import checkpoints
import jax_cfd.base as cfd
from jax_cfd.base.grids import AlignedArray
from jax_cfd.ml import tiling


import numpy as np                     
import optax                           
import torch.utils.data as Data
import torchvision
from tqdm import tqdm
from glob import glob
from functools import partial
from typing import Any, Optional, Callable, Tuple
import wandb
import functools


#------------------------------------------------------------------#    
# DATA LOADING
#------------------------------------------------------------------#    

std = 2.2

def numpy_collate(batch):
    if isinstance(batch[0], jnp.ndarray):
        return jnp.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        return type(batch[0])(numpy_collate(samples) for samples in zip(*batch))
    else:
        return jnp.asarray(batch)


class CFDDataset(Data.Dataset):
    """Dataset for solver-in-the-loop model. Only loads DP snapshots"""
    
    def __init__(self, config, train=True, sim=None, frac=1):
        self.data_dir = config['data_dir']
        self.timespan = config['timespan']
        self.dtype = np.float32
        self.inputs = []
        self.dp_data = []
        
        
        # if specific sim number is provided, use it, else use train/test split
        if sim:    
            dp_glob = sorted(glob(os.path.join(self.data_dir, 'DP', sim, '*.npy')))
            dp_arrays_temp = [np.load(f).astype(self.dtype) for f in dp_glob]
            
            self.inputs += dp_arrays_temp[:len(dp_glob) - self.timespan]
            self.dp_data += [
                np.stack(dp_arrays_temp[i+1:i+self.timespan+1]) 
                for i in range(len(dp_glob) - self.timespan)
            ]

        
        else:
            dp_sims = sorted(glob(os.path.join(self.data_dir, 'DP', 'train' if train else 'test', 'sim_*')))
            
            for i, sim in enumerate(dp_sims):
                dp_glob = sorted(glob(os.path.join(sim, '*.npy')))
                dp_arrays_temp = [np.load(f).astype(self.dtype) for f in dp_glob]
                
                self.inputs += dp_arrays_temp[:len(dp_glob) - self.timespan]
                self.dp_data += [
                    np.stack(dp_arrays_temp[i+1:i+self.timespan+1]) 
                    for i in range(len(dp_glob) - self.timespan)
                ]
                

        if frac!=1:
            self.inputs = [self.inputs[i] for i in range(int(frac * len(self.inputs)))]
            self.dp_data = [self.dp_data[i] for i in range(int(frac * len(self.dp_data)))]

        
    def __len__(self):
        return len(self.inputs)
        
    def __getitem__(self, i):
        return jnp.asarray(self.inputs[i]), jnp.asarray(self.dp_data[i])


#------------------------------------------------------------------#    
# UTILS
#------------------------------------------------------------------#

class TrainState(train_state.TrainState):
    dynamic_scale: optim.DynamicScale
    examples_seen: int
    best_val_loss: float
        
    def update_value(self, name, val):
        object.__setattr__(self, name, val)
    
def restore_checkpoint(state, workdir):
    return checkpoints.restore_checkpoint(workdir, state)

def save_checkpoint(state, workdir):
    step = int(state.step)
    checkpoints.save_checkpoint(workdir, state, step, keep=3)
    
def create_train_state(rng, config, examples_seen=0, best_val_loss=float("inf")):
    dynamic_scale = optim.DynamicScale()
    cnn = HybridSolver(dtype=config['dtype'], solver_dtype=config['solver_dtype'], solver_step_fn=step_fn)
    init_cnn = cnn.init(rng, jnp.ones([1, 256, 256, 2], dtype=config['dtype']))
    params=init_cnn['params']
    tx = optax.adamw(config['learning_rate'], config['weight_decay'])
    
    return TrainState.create(
        apply_fn=cnn.apply, 
        params=params, 
        tx=tx,
        examples_seen=examples_seen,
        best_val_loss=best_val_loss,
        dynamic_scale=dynamic_scale
    )


#------------------------------------------------------------------#    
# SOLVER
#------------------------------------------------------------------#

# Make sure these settings are the same as that of the simulation
re = 4000
size = 256
density = 1
viscosity = 1/re
inner_steps = 6
max_velocity = 10
peak_wavenumber = 4
cfl_safety_factor = 0.5
constant_magnitude = 4
constant_wavenumber = 4
linear_coefficient = -0.1

grid = cfd.grids.Grid((size, size), domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)))
dt = cfd.equations.stable_time_step(max_velocity, cfl_safety_factor, viscosity, grid)

forcing_fn = cfd.forcings.simple_turbulence_forcing(
        grid,
        constant_magnitude,
        constant_wavenumber,
        linear_coefficient,
        forcing_type='kolmogorov'
    )

step_fn = cfd.funcutils.repeated(cfd.equations.semi_implicit_navier_stokes(density=density, viscosity=viscosity, dt=dt, grid=grid, forcing=forcing_fn), steps=inner_steps)
step_fn = jax.vmap(jax.jit(step_fn))


def ndarray_to_field(arr):
    """ 
    Takes an array of shape (batch, spatial, spatial, field)
    Returns a tuple of length (field) of AlignedArrays with data of shape (batch, spatial, spatial)
    
    field = 2
    """
    return (AlignedArray(data=arr[:, :, :, 0], offset=(1.0, 0.5)), 
            AlignedArray(data=arr[:, :, :, 1], offset=(0.5, 1.0)))

def field_to_ndarray(field):
    """ 
    Takes a tuple of length (field) of AlignedArrays with data of shape (batch, spatial, spatial)
    Returns an array of shape (batch, spatial, spatial, field)
    """
    return jnp.stack([f.data for f in field], axis=-1)    


#------------------------------------------------------------------#
# MODEL
#------------------------------------------------------------------#

class PeriodicConv(nn.Module):
    features: int
    kernel_size: Tuple[int, ...]
    dtype: Any = jnp.float32
        
        
    def setup(self):
        rate = 1
        self._padding = [(1, 1), (1, 1)]
        self._conv_module = nn.Conv(features=self.features, kernel_size=self.kernel_size, padding='VALID', dtype=self.dtype)

    def __call__(self, inputs):
        layout_map = {
            1 : (1, 1),
            2 : (2, 1),
            4 : (2, 2),
            8 : (4, 2),
            16 : (4, 4),
            32 : (8, 4),
            64 : (8, 8),
        }
        layout = layout_map[inputs.shape[0]]
        padded = tiling.halo_exchange_pad(inputs, layout, self._padding)
        
        output = self._conv_module(padded)
        return output
        
    

class ResBlock(nn.Module):
    conv: Any
    n_features: int
        
    @nn.compact
    def __call__(self, x,):
        residual = x
        y = self.conv(features=self.n_features)(x)
        y = nn.relu(y)
        y = self.conv(features=self.n_features)(y)
        
        if residual.shape != y.shape:
            residual = self.conv(features=self.n_features, kernel_size=(1, 1))(residual)
        
        return nn.relu(residual + y)
    

class CNN(nn.Module):
    solver_step_fn: Callable
    dtype: Any = jnp.float32
    solver_dtype: Any = jnp.bfloat16
    features = jnp.array([32, 32, 32, 32, 32, 32])
    block_cls = ResBlock
 
    
    def neural_step(self, x):
            conv = partial(PeriodicConv, 
               features=32,
               kernel_size=(3, 3),
               dtype=self.dtype)

            x_in = x
            y = conv()(x)
            y = nn.relu(y)

            for i in self.features:
                y = self.block_cls(conv=conv, n_features=i)(y)

            y = conv(features=2)(y)
            return y
        
    def solver_step(self, x):
        x = x.astype(self.solver_dtype)
        x = field_to_ndarray(self.solver_step_fn(ndarray_to_field(x)))
        x = x.astype(self.dtype)
        return x
    
    @nn.compact
    def __call__(self, x, *args):
        x = self.solver_step(x)
        correction = self.neural_step(x / std) * std
        x = x + correction

#         x = self.neural_step(x / std) * std

        return x, x
        

class HybridSolver(nn.Module):
    solver_step_fn: Callable
    dtype: Any = jnp.float32
    solver_dtype: Any = jnp.bfloat16
    
    @nn.compact
    def __call__(self, x, timespan=2):
        hybrid_step_rollout = nn.scan(
            CNN,
            variable_broadcast='params',
            split_rngs={'params': False},
            length=timespan,
            out_axes=1
        )
        
        _, out = hybrid_step_rollout(
            solver_step_fn=self.solver_step_fn,
            dtype=self.dtype,
            solver_dtype=self.solver_dtype
        )(x.astype(self.dtype), None)
        
        return out
        
    
def MSE(pred, truth):
    return ((truth-pred)**2).mean()

def MAE(pred, truth):
    return jnp.abs(truth-pred).mean()


@partial(jax.jit, static_argnums=3)
def train_step(state, SP, DP, timespan):
    def loss_fn(params):
        pred = state.apply_fn({'params': params}, SP, timespan=timespan)
        loss = MSE(pred, DP)
        
        return loss, pred
    
    dynamic_scale = state.dynamic_scale
    
    if dynamic_scale:
        grad_fn = dynamic_scale.value_and_grad(loss_fn, has_aux=True)
        dynamic_scale, is_fin, aux, grads = grad_fn(state.params)
        state.update_value('dynamic_scale', dynamic_scale)
    else:
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        aux, grads = grad_fn(state.params)
        
    loss = aux[0]
    pred = aux[1]
        
    metrics = {'Train Loss': loss}
    new_state = state.apply_gradients(grads=grads)
    
    
    if dynamic_scale:
        new_state = new_state.replace(
            opt_state=jax.tree_multimap(
                partial(jnp.where, is_fin),
                new_state.opt_state,
                state.opt_state),
            params=jax.tree_multimap(
                partial(jnp.where, is_fin),
                new_state.params,
                state.params))
        metrics['Scale'] = dynamic_scale.scale
        
    return new_state, metrics, pred
    
    
@partial(jax.jit, static_argnums=3)
def eval_step(state, SP, DP, timespan):
    pred = state.apply_fn({'params': state.params}, SP, timespan=timespan)
    loss = MSE(pred, DP)
  
    metrics = {'Val Loss': loss}
    return metrics
    
    
def train_epoch(state, train_loader, epoch, config):

    for i, (SP, DP) in enumerate(tqdm(train_loader, desc=f"Training Epoch: {epoch}")):
        state, metrics, pred = train_step(state, SP, DP, timespan=config['timespan'])
        
        # LOG BATCH METRICS IN WANDB
        state.update_value('examples_seen', state.examples_seen + DP.shape[0])
        metrics = jax.device_get(metrics)
        
        wandb.log({
            "Examples Seen": state.examples_seen.item(), 
            "Train Loss": metrics['Train Loss'].item(),
            "Scale": metrics['Scale'].item(),
        })

    return state, metrics, pred

    
def eval_model(state, test_loader, config):
    batch_metrics = []

    for i, (SP, DP) in enumerate(tqdm(test_loader, desc="Validation")):
        metrics = eval_step(state, SP, DP, timespan=config['timespan'])
        batch_metrics.append(metrics)

    # compute mean of metrics across each batch in epoch.
    batch_metrics_np = jax.device_get(batch_metrics)
    test_metrics_np = {
      k: np.mean([metrics[k] for metrics in batch_metrics_np])
      for k in batch_metrics_np[0]}
    
    # LOG EPOCH TEST METRICS IN WANDB
    wandb.log({"Examples Seen": state.examples_seen.item(), 
               "Val Loss": test_metrics_np['Val Loss'].item()
    })
    
    if test_metrics_np['Val Loss'] < state.best_val_loss.item():
        print("New best val loss!")
        state.update_value('best_val_loss', test_metrics_np['Val Loss'])
        save_checkpoint(state, config['work_dir'])

    return test_metrics_np