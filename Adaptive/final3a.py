#FINAL: This one does not include the belief loss 

import os
import csv
import numpy as np
import random
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

import gymnasium as gym
import gymnasium.spaces as gspaces
from gymnasium.spaces import Box
from gymnasium.spaces.utils import flatdim, flatten, unflatten

# MiniGrid wrappers (v2 API)
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
# for optional video
from gymnasium.wrappers import RecordVideo
#PopGYm
import popgym

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Bernoulli, Normal, Independent


#using hidden_size, not in_features
#surprise here: trying to predict plain GRU 

#redid it to match POPGym 7/9

#7/27 evening, this is like the paper, we actually forgot to do the minibatch part so now we did 



# ---------------- Configuration as dataclass -----------------

@dataclass
class Config:
    seed: int = 0
    num_trials: int = 2
    model_name: str = "PPOSurpriseScalarHSSGRU"
    env_id: str = 'popgym-CountRecallEasy-v0'
    steps_per_epoch: int = 2048
    update_epochs: int = 1          # Number of epochs over collected data
    num_minibatches: int = 1        # Number of mini-batches per epoch
    learning_rate: float = 2.5e-4
    clip_epsilon: float = 0.25
    lambda_gae: float = 1.0
    value_loss_coef: float = 1.0
    entropy_coef: float = 0.05
    gamma: float = 0.99
    total_env_steps: int = 15_000_000
    tbptt_len: int = 1024          # <= truncation window (L). 
    USE_CUDA: bool = True          # set to False to force CPU
    debug: bool = False            # Enable debug prints
    
    # Surprise-specific parameters
    gate_width: int = 32           # Width of the surprise gate MLP
    belief_loss_coef: float = 0.25  # Coefficient for predictor loss
    use_stronger_predictor: bool = False #MLP is True and Linear is False for predictor surprise
    use_belief_warmup: bool = True #warmup to fix cold start problem 
    belief_warmup_epochs: int = 5 #number of warmup epochs 
    belief_ramp_epochs: int = 40 #when ramp up stops 
    pretrain_predictor_steps: int = 2000 #pretrain predictor so not spoardic 
    scalar_surprise: bool = False #True: scalar, False: vector gating


    grad_clip_norm: float = 0.5          # Gradient clipping, myabe try 1 or 3 here
    scratch_clip_value: float = 5.0      # Scratch memory clipping
    theta_max: float = 5.0  # Maximum theta value

    HSS: bool = True #True means HSS, False means 
    predict_GRU_or_step: bool = True #True means GRU, False means Step 


    emb_dim: int = 256             # Changed from 32 to match plain GRU
    hidden_dim: int = 256          # Keep as is


config = Config()

# ------------------ Utility Functions ------------------------

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    gym.utils.seeding.np_random(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@lru_cache(maxsize=None)
def episode_horizon(env: gym.Env) -> int:
    """
    Retrieves episode length cap for any environment,
    searching wrappers and unwrapped env.
    """
    # Check various possible attribute names
    for attr in ("max_steps", "_max_episode_steps", "max_episode_steps", "episode_length"):
        horizon = getattr(env, attr, None)
        if horizon is not None:
            return horizon
    
    # Check the unwrapped environment
    unwrapped = getattr(env, "unwrapped", env)
    for attr in ("max_steps", "_max_episode_steps", "max_episode_steps", "episode_length"):
        horizon = getattr(unwrapped, attr, None)
        if horizon is not None:
            return horizon
    
    # Check the spec if available
    if hasattr(env, "spec") and env.spec is not None:
        horizon = getattr(env.spec, "max_episode_steps", None)
        if horizon is not None:
            return horizon
    
    # For POPGym environments, many have a default of 1000 steps
    #if hasattr(env, "spec") and env.spec is not None and "popgym" in str(env.spec.id).lower():
    #    return 1000
    
    print(f"[episode_horizon] WARNING: no horizon found for {getattr(env, 'spec', env).id}; using default=1")
    return 1


def to_tensor(o, space: gym.Space, device: torch.device) -> torch.Tensor:
    """
    Convert an observation to a PyTorch tensor suitable for the network.
    Handles discrete, image, and vector observations.
    """
    if isinstance(space, gspaces.Discrete):
        # For discrete spaces, just return the scalar as a tensor
        return torch.tensor(o, dtype=torch.int64, device=device).unsqueeze(0)  # (1,)
    if isinstance(space, Box) and len(space.shape) == 3 and space.shape[-1] == 3:
        return torch.tensor(o, dtype=torch.uint8, device=device).unsqueeze(0)  # (1, H, W, C)
    flat = torch.tensor(flatten(space, o), dtype=torch.float32, device=device)
    return flat.unsqueeze(0)  # (1, F)


def make_env(env_id: str, view_size: int = 3, render: bool = False) -> gym.Env:
    """
    Creates a Gymnasium environment with wrappers for MiniGrid or POPGym.
    """
    if env_id.startswith("MiniGrid"):
        env = gym.make(env_id, agent_view_size=view_size,
                       render_mode="rgb_array" if render else None)
        env = ImgObsWrapper(RGBImgPartialObsWrapper(env))
    else:
        env = gym.make(env_id)
    return env

#Weight initializaion 
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight)
    elif isinstance(m, nn.GRU):
        for name, p in m.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "weight_ih" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.constant_(p, 0.)

# ------------------ Encoders (Keep existing) -----------------------------

class DenseEncoder(nn.Module):
    """Encodes discrete or vector observations into fixed-width embeddings."""
    def __init__(self, obs_space: gym.Space, emb_dim: int = 128):
        super().__init__()
        if isinstance(obs_space, gspaces.Discrete):
            self.mode = "discrete"
            self.embed = nn.Embedding(obs_space.n, emb_dim)
            self.preprocessor = nn.Sequential(
                nn.LayerNorm(emb_dim, elementwise_affine=False),
                nn.Linear(emb_dim, emb_dim),
                nn.LeakyReLU(inplace=True)
            )
            self.out_dim = emb_dim
        else:
            self.mode = "vector"
            in_dim = flatdim(obs_space)
            self.embed = nn.Linear(in_dim, emb_dim)
            self.preprocessor = nn.Sequential(
                nn.LayerNorm(emb_dim, elementwise_affine=False),
                nn.Linear(emb_dim, emb_dim),
                nn.LeakyReLU(inplace=True)
            )
            self.out_dim = emb_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that handles various input shapes.
        
        For discrete:
            - (B,) or (T,) -> (B, emb_dim) or (T, emb_dim)
            - (B, T) -> (B, T, emb_dim)
            
        For continuous:
            - (B, F) or (T, F) -> (B, emb_dim) or (T, emb_dim)
            - (B, T, F) -> (B, T, emb_dim)
        """
        if self.mode == "discrete":
            # Discrete observations
            if x.dim() == 1:
                x = self.embed(x.long())  # (B, emb_dim)
                x = self.preprocessor(x)   # Apply preprocessor
                return x
            elif x.dim() == 2:
                x = self.embed(x.long())  # (B, T, emb_dim)
                # Reshape for preprocessor
                B, T, F = x.shape
                x = x.reshape(B * T, F)
                x = self.preprocessor(x)
                x = x.reshape(B, T, self.out_dim)
                return x
            else:
                raise ValueError(f"Unexpected shape for discrete obs: {x.shape}")
        else:
            # Continuous observations
            if x.dim() == 2:
                x = self.embed(x.float())  # (B, emb_dim)
                x = self.preprocessor(x)   # Apply preprocessor
                return x
            elif x.dim() == 3:
                B, T, F = x.shape
                x_flat = x.reshape(B * T, F)
                out_flat = self.embed(x_flat.float())
                out_flat = self.preprocessor(out_flat)  # Apply preprocessor
                return out_flat.reshape(B, T, self.out_dim)
            else:
                raise ValueError(f"Unexpected shape for continuous obs: {x.shape}")


class ConvEncoder(nn.Module):
    """Simple 2-layer CNN encoder for image inputs."""
    def __init__(self, V: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), 
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, 3, padding=1), 
            nn.LeakyReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 3, V, V)
            self.out_dim = self.net(dummy).shape[-1]

    def forward(self, img_uint8: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for image observations.
        
        Input shapes:
            - (B, H, W, C) -> (B, out_dim)
            - (B, T, H, W, C) -> (B, T, out_dim)
        """
        if img_uint8.dim() == 4:
            # Single image per batch: (B, H, W, C)
            x = img_uint8.permute(0, 3, 1, 2).float() / 255.0  # (B, C, H, W)
            return self.net(x)  # (B, out_dim)
        elif img_uint8.dim() == 5:
            # Sequence of images: (B, T, H, W, C)
            B, T, H, W, C = img_uint8.shape
            img_flat = img_uint8.reshape(B * T, H, W, C)  # (B*T, H, W, C)
            x = img_flat.permute(0, 3, 1, 2).float() / 255.0  # (B*T, C, H, W)
            out_flat = self.net(x)  # (B*T, out_dim)
            return out_flat.reshape(B, T, self.out_dim)  # (B, T, out_dim)
        else:
            raise ValueError(f"Unexpected image shape: {img_uint8.shape}")


# ------------------ Surprise-GRU Cell (NEW) --------------------------

class SurpriseGRUCell(nn.Module):
    """GRU with a surprise-gated scratch path.
    
    Args:
        in_features: dimension of x_t (encoded obs-embedding)
        hidden_size: dimension of h_t (policy state)
        gate_width: width of the tiny MLP that outputs (η, θ, α)
    """
    def __init__(self, in_features: int, hidden_size: int, gate_width: int = 64, 
                 warmup_epochs: int = 10, ramp_epochs: int = 40, scratch_clip: float = 5.0,
                 theta_max: float = 10.0):
        super().__init__()
        self.theta_max = theta_max
        self.scratch_clip = scratch_clip
        self.hidden_size = hidden_size
        self.warmup_epochs = warmup_epochs
        self.ramp_epochs = ramp_epochs
        
        # Vanilla GRU parameters
        self.W_rz = nn.Linear(in_features + hidden_size, 2 * hidden_size)
        self.W_n = nn.Linear(in_features + hidden_size, hidden_size)
        
        # Surprise-gate MLP
        #self.gate_net = nn.Sequential(
        #    nn.Linear(in_features, gate_width), 
        #    nn.ReLU(),
        #    nn.Linear(gate_width, 3 * hidden_size)
        #)
        
        # Map encoder-space surprise to hidden-space
        #self.delta_proj = nn.Linear(in_features, hidden_size, bias=False) #*********

        self.gate_net = nn.Sequential(
            nn.Linear(hidden_size, gate_width),
            nn.LayerNorm(gate_width),  # Add normalization
            nn.ReLU(),
            nn.Dropout(0.1),  # Add regularization
            nn.Linear(gate_width, gate_width),
            nn.ReLU(),
            nn.Linear(gate_width, 3 * hidden_size)
        )

        # Keep delta_proj as-is since it's already hidden_size → hidden_size
        self.delta_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    @staticmethod
    def _match_batch(a, b):
        """Expand the 1st-dim of either tensor so both share the same batch length."""
        if a.size(0) == b.size(0):
            return a, b
        if a.size(0) == 1:
            a = a.expand(b.size(0), *a.shape[1:])
            return a, b
        if b.size(0) == 1:
            b = b.expand(a.size(0), *b.shape[1:])
            return a, b
        raise ValueError(f"Cannot broadcast batch: {a.size(0)} vs {b.size(0)}")

    def candidate_and_z(self, x_t, h_old):
        """Helper: vanilla GRU candidate & update gate (no surprise path)"""
        x_t = x_t.view(-1, x_t.size(-1))
        h_old = h_old.view(-1, h_old.size(-1))
        x_t, h_old = self._match_batch(x_t, h_old)

        concat = torch.cat([x_t, h_old], dim=-1)
        r_t, z_t = self.W_rz(concat).chunk(2, dim=-1)
        r_t, z_t = torch.sigmoid(r_t), torch.sigmoid(z_t)
        n_t = torch.tanh(self.W_n(torch.cat([x_t, r_t * h_old], dim=-1)))
        return n_t, z_t  # B×H, B×H

    def forward(self, x_t, h_old, delta, epoch=0):
        """
        Args:
            x_t: B×F (encoded observation)
            h_old: B×H (previous hidden state)
            delta: B×F (surprise signal)
        """
        # Vanilla GRU candidate & update
        x_t = x_t.view(-1, x_t.size(-1))
        h_old = h_old.view(-1, h_old.size(-1))
        x_t, h_old = self._match_batch(x_t, h_old)

        concat = torch.cat([x_t, h_old], dim=-1)
        r_t, z_t = self.W_rz(concat).chunk(2, dim=-1)
        r_t, z_t = torch.sigmoid(r_t), torch.sigmoid(z_t)
        n_t = torch.tanh(self.W_n(torch.cat([x_t, r_t * h_old], dim=-1)))


        # Surprise-driven scratch write WITH EPOCH SCALING
        # Calculate surprise scale based on current epoch
        if epoch < self.warmup_epochs:
            surprise_scale = 0.0
        else:
            progress = (epoch - self.warmup_epochs) / self.ramp_epochs
            surprise_scale = min(1.0, progress)


        # Surprise-driven scratch write
        g_raw = self.gate_net(delta)
        eta = torch.sigmoid(g_raw[:, :self.hidden_size])
        theta = F.softplus(g_raw[:, self.hidden_size:2*self.hidden_size])
        theta = torch.clamp(theta, max=self.theta_max)
        alpha = torch.sigmoid(g_raw[:, 2*self.hidden_size:]) * surprise_scale

        delta_h = self.delta_proj(delta)  # B × H
        s_t = torch.clamp(eta * h_old - theta * delta_h, -self.scratch_clip, self.scratch_clip)
        h_tilde = (1 - alpha) * h_old + alpha * s_t  # inject scratch

        # Final mix with GRU update
        h_new = z_t * h_tilde + (1 - z_t) * n_t
        return h_new, (eta, alpha, theta, s_t)


# ------------------ PPO-Surprise-GRU Policy --------------------------

class PPOSurpriseGRUPolicy(nn.Module):
    """
    PPO policy with Surprise-GRU that works for any Gymnasium action space.
    """
    def __init__(self, obs_space: gym.Space, action_space: gspaces.Space):
        super().__init__()
        self.hidden_size = config.hidden_dim
        self.encoder_dim = config.emb_dim
        self.action_space = action_space
        self.current_epoch = 0

        # ---------- observation encoder ----------
        img_like = (
            isinstance(obs_space, gspaces.Box)
            and obs_space.shape is not None
            and len(obs_space.shape) == 3
            and obs_space.shape[-1] == 3
        )
        if img_like:
            V = obs_space.shape[0]
            self.encoder = ConvEncoder(V)
        else:
            self.encoder = DenseEncoder(obs_space)

        # ---------- Predictor head for surprise ----------
        if config.use_stronger_predictor:
            # Use 2-layer MLP for stronger predictor
            self.predict = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LeakyReLU(),
                nn.Linear(self.hidden_size, self.hidden_size)
            )
        else:
            # Original: single linear layer
            self.predict = nn.Linear(self.hidden_size, self.hidden_size)


        # ADD THIS: Extra preprocessor layer (PopGym style)
        self.preprocessor = nn.Sequential(
            nn.Linear(self.encoder.out_dim, self.encoder.out_dim),
            nn.LeakyReLU(inplace=True)
        )
        
        # ---------- Surprise-GRU instead of vanilla GRU ----------
        self.sgru = SurpriseGRUCell(
        self.encoder.out_dim, 
        self.hidden_size, 
        gate_width=config.gate_width,
        warmup_epochs=config.belief_warmup_epochs,
        ramp_epochs=config.belief_ramp_epochs,
        scratch_clip=config.scratch_clip_value,  
        theta_max=config.theta_max 
        )
        
        # ---------- Value head ----------
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.encoder.out_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.encoder.out_dim, self.encoder.out_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.encoder.out_dim, 1)
        )

        # ---------- action heads ----------
        if isinstance(action_space, gspaces.Discrete):
            self.pi_head = nn.Sequential(
                nn.Linear(self.hidden_size, self.encoder.out_dim),
                nn.LeakyReLU(inplace=True),
                nn.Linear(self.encoder.out_dim, self.encoder.out_dim),
                nn.LeakyReLU(inplace=True),
                nn.Linear(self.encoder.out_dim, action_space.n)
            )

        elif isinstance(action_space, gspaces.MultiDiscrete):
            self.pi_head = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.hidden_size, self.encoder.out_dim),
                    nn.LeakyReLU(inplace=True),
                    nn.Linear(self.encoder.out_dim, self.encoder.out_dim),
                    nn.LeakyReLU(inplace=True),
                    nn.Linear(self.encoder.out_dim, int(n))
                ) for n in action_space.nvec
            ])

        elif isinstance(action_space, gspaces.MultiBinary):
            self.pi_head = nn.Sequential(
                nn.Linear(self.hidden_size, self.encoder.out_dim),
                nn.LeakyReLU(inplace=True),
                nn.Linear(self.encoder.out_dim, self.encoder.out_dim),
                nn.LeakyReLU(inplace=True),
                nn.Linear(self.encoder.out_dim, action_space.n)
            )

        elif isinstance(action_space, gspaces.Box):
            self.action_dim = int(np.prod(action_space.shape))
            self.mu_head = nn.Sequential(
                nn.Linear(self.hidden_size, self.encoder.out_dim),
                nn.LeakyReLU(inplace=True),
                nn.Linear(self.encoder.out_dim, self.encoder.out_dim),
                nn.LeakyReLU(inplace=True),
                nn.Linear(self.encoder.out_dim, self.action_dim)
            )
            # one log-std parameter per dimension
            self.log_std = nn.Parameter(torch.zeros(self.action_dim))

        else:
            raise NotImplementedError(f"Unsupported action space {action_space}")
        

        # Apply orthogonal initialization to all layers
        self.apply(init_weights)

        # Special initialization for final output layers (smaller std)
        if isinstance(action_space, gspaces.Discrete):
            nn.init.normal_(self.pi_head[-1].weight, std=0.01)
        elif isinstance(action_space, gspaces.MultiDiscrete):
            for head in self.pi_head:
                nn.init.normal_(head[-1].weight, std=0.01)
        elif isinstance(action_space, gspaces.MultiBinary):
            nn.init.normal_(self.pi_head[-1].weight, std=0.01)
        elif isinstance(action_space, gspaces.Box):
            nn.init.normal_(self.mu_head[-1].weight, std=0.01)

        nn.init.normal_(self.value_head[-1].weight, std=0.01)
        
    def set_epoch(self, epoch):
        """Update current epoch for surprise scaling."""
        self.current_epoch = epoch

    def step(self, x_enc, h_prev):
        """
        Single step with surprise computation.
        
        Returns:
            logits: Action logits
            value: Value estimate  
            h_new: New hidden state
            S_t: Scalar surprise (B, 1)
            eta: Eta gate values (B, H)
            alpha: Alpha gate values (B, H)
        """
        # Ensure proper shapes
        if x_enc.dim() == 1: 
            x_enc = x_enc.unsqueeze(0)
        if h_prev.dim() == 1: 
            h_prev = h_prev.unsqueeze(0)
        
        # 1. Predictor makes a guess based ONLY on previous hidden
        h_pred = self.predict(h_prev)  # B × H
        
        # 2. # GRU computes actual candidate using BOTH h_prev AND current input
        h_cand, _ = self.sgru.candidate_and_z(x_enc, h_prev)
        
        # 3. Hidden-space surprise
        delta = h_cand.detach() - h_pred  # B × H (stop-grad on candidate)
        S_t = delta.norm(p=2, dim=-1, keepdim=True)  # B × 1
        

        # 4. Prepare gate input based on scalar vs vector mode
        if config.scalar_surprise:
            gate_in = S_t.expand(-1, self.hidden_size)  # (B, 1) -> (B, H)
        else:
            # Normalize delta for vector surprise
            delta_norm = (delta - delta.mean(dim=-1, keepdim=True)) / (delta.std(dim=-1, keepdim=True) + 1e-6)
            gate_in = delta_norm  # (B, H)


        # 4. Surprise-gated update (expand scalar S_t -> B×enc_dim)
        #if config.scalar_surprise:                      # scalar   (B,1)
        #    gate_in = S_t.expand(-1, x_enc.size(-1))    # expand to (B, enc_dim)
        #else:
        #    delta_norm = (delta - delta.mean(dim=-1, keepdim=True)) \
        #                 / (delta.std(dim=-1, keepdim=True) + 1e-6)
        #    gate_in = delta_norm                  # full vector
        
        h_new, (eta, alpha, theta, s_t) = self.sgru(
            x_enc, h_prev, gate_in, epoch=self.current_epoch
        )
        
        
        # 5. Policy & value outputs
        value = self.value_head(h_new).squeeze(-1)  # B
        
        # Action logits depend on action space
        if isinstance(self.action_space, gspaces.Discrete):
            logits = self.pi_head(h_new)  # B × A
        elif isinstance(self.action_space, gspaces.MultiDiscrete):
            logits = [head(h_new) for head in self.pi_head]  # list of B × ni
        elif isinstance(self.action_space, gspaces.MultiBinary):
            logits = self.pi_head(h_new)  # B × bits
        elif isinstance(self.action_space, gspaces.Box):
            logits = self.mu_head(h_new)  # B × action_dim
            
        return logits, value, h_new, S_t, eta, alpha, theta, s_t

    def forward(self, obs: torch.Tensor, h: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass compatible with existing training loop.
        
        Returns 8 values now: logits, value, h1, S_t, eta, alpha, theta, s_t
        """
        # Encode observation
        x = self.encoder(obs)  # (B, emb_dim) or (B, T, emb_dim)
        
        if config.debug:
            print(f"[DEBUG] After encoder: x.shape = {x.shape}")
        
        # Handle sequences for TBPTT
        if x.dim() == 3:  # (B, T, emb_dim)
            B, T, F = x.shape
            if h is None:
                h = self.init_hidden(batch_size=B, device=x.device)
            
            # Process sequence step by step
            outputs = []
            S_ts = []
            etas = []
            alphas = []
            thetas = []  
            s_ts = []    
            
            for t in range(T):
                logits_t, value_t, h, S_t, eta, alpha, theta, s_t = self.step(x[:, t], h.squeeze(0) if h.dim() == 3 else h)
                outputs.append((logits_t, value_t))
                S_ts.append(S_t)
                etas.append(eta)
                alphas.append(alpha)
                thetas.append(theta)  
                s_ts.append(s_t)      
            
            # Stack outputs
            if isinstance(self.action_space, gspaces.MultiDiscrete):
                logits = []
                for i in range(len(self.pi_head)):
                    logits_i = torch.stack([o[0][i] for o in outputs], dim=1)  # B × T × ni
                    logits.append(logits_i)
            else:
                logits = torch.stack([o[0] for o in outputs], dim=1)  # B × T × A
            
            values = torch.stack([o[1] for o in outputs], dim=1)  # B × T
            S_t = torch.stack(S_ts, dim=1)  # B × T × 1
            eta = torch.stack(etas, dim=1)  # B × T × H
            alpha = torch.stack(alphas, dim=1)  # B × T × H
            theta = torch.stack(thetas, dim=1)  # B × T × H  
            s_t = torch.stack(s_ts, dim=1)      # B × T × H  
            
            return logits, values, h.unsqueeze(0) if h.dim() == 2 else h, S_t, eta, alpha, theta, s_t
        
        else:  # Single timestep
            if h is None:
                h = self.init_hidden(batch_size=x.size(0), device=x.device)
            
            # Remove extra dimension if present
            if h.dim() == 3 and h.size(0) == 1:
                h = h.squeeze(0)
            
            logits, value, h_new, S_t, eta, alpha, theta, s_t = self.step(x, h)
            
            # Add time dimension for compatibility
            if isinstance(self.action_space, gspaces.MultiDiscrete):
                logits = [l.unsqueeze(1) for l in logits]  # list of (B, 1, ni)
            else:
                logits = logits.unsqueeze(1)  # (B, 1, A)
            
            value = value.unsqueeze(1)  # (B, 1)
            eta = eta.unsqueeze(1)  # (B, 1, H)
            alpha = alpha.unsqueeze(1)  # (B, 1, H)
            theta = theta.unsqueeze(1)  # (B, 1, H)  
            s_t = s_t.unsqueeze(1)      # (B, 1, H)  
            
            return logits, value, h_new.unsqueeze(0), S_t, eta, alpha, theta, s_t

    def init_hidden(self, batch_size: int = 1, device: str = "cpu") -> torch.Tensor:
        return torch.zeros(1, batch_size, self.hidden_size, device=device)

    # Keep all the existing helper methods unchanged
    def _build_dist(self, logits):
        if isinstance(self.action_space, gspaces.Discrete):
            return Categorical(logits=logits)
        if isinstance(self.action_space, gspaces.MultiDiscrete):
            return [Categorical(logits=l) for l in logits]
        if isinstance(self.action_space, gspaces.MultiBinary):
            return Bernoulli(logits=logits)
        if isinstance(self.action_space, gspaces.Box):
            std = self.log_std.exp()
            base = Normal(logits, std)
            return Independent(base, 1)

    def _sample(self, dist, deterministic=False):
        if isinstance(self.action_space, gspaces.Discrete):
            action = dist.probs.argmax(-1) if deterministic else dist.sample()
            logp = dist.log_prob(action)
            return action, logp

        if isinstance(self.action_space, gspaces.MultiDiscrete):
            comps, logps = [], []
            for d in dist:
                a = d.probs.argmax(-1) if deterministic else d.sample()
                lp = d.log_prob(a)
                comps.append(a)
                logps.append(lp)
            action = torch.stack(comps, dim=-1)
            logp = torch.stack(logps, dim=-1).sum(-1)
            return action, logp

        if isinstance(self.action_space, gspaces.MultiBinary):
            probs = dist.probs
            action = ((probs > 0.5).float() if deterministic else dist.sample()).to(torch.uint8)
            logp = dist.log_prob(action).sum(-1)
            return action, logp

        if isinstance(self.action_space, gspaces.Box):
            action = dist.mean if deterministic else dist.rsample()
            logp = dist.log_prob(action)
            low = torch.as_tensor(self.action_space.low, device=action.device)
            high = torch.as_tensor(self.action_space.high, device=action.device)
            action = torch.max(torch.min(action, high), low)
            return action, logp

    def act(self, obs, h, deterministic=False):
        """Modified to handle surprise output"""
        logits, v, h1, S_t, eta, alpha, theta, s_t = self.forward(obs, h)

        if isinstance(self.action_space, gspaces.MultiDiscrete):
            logits_last = [l[:, -1] for l in logits]
        else:
            logits_last = logits[:, -1]

        v_last = v[:, -1]
        dist = self._build_dist(logits_last)
        act, logp = self._sample(dist, deterministic)
        return act, logp, v_last, h1

    def evaluate_actions(self, obs, h, actions):
        """Modified to return surprise for loss computation"""
        logits, v, _, S_t, eta, alpha, theta, s_t = self.forward(obs, h)  
        dist = self._build_dist(logits)

        if isinstance(self.action_space, gspaces.Discrete):
            logp = dist.log_prob(actions)
            entropy = dist.entropy()

        elif isinstance(self.action_space, gspaces.MultiDiscrete):
            logps, entrs = [], []
            for i, d in enumerate(dist):
                logps.append(d.log_prob(actions[..., i]))
                entrs.append(d.entropy())
            logp = torch.stack(logps, dim=-1).sum(-1)
            entropy = torch.stack(entrs, dim=-1).sum(-1)

        elif isinstance(self.action_space, gspaces.MultiBinary):
            logp = dist.log_prob(actions.float()).sum(-1)
            entropy = dist.entropy().sum(-1)

        elif isinstance(self.action_space, gspaces.Box):
            logp = dist.log_prob(actions)
            entropy = dist.base_dist.entropy().sum(-1)

        return v, logp, entropy, S_t, eta, alpha, theta, s_t


class TrajectoryBuffer:
    def __init__(
        self,
        size: int,
        obs_space: gym.Space,
        action_space: gspaces.Space,
        hidden_dim: int,
        device: torch.device,
    ):
        self.device = device
        self.obs_space = obs_space
        is_rgb = isinstance(obs_space, Box) and len(obs_space.shape) == 3 and obs_space.shape[-1] == 3

        if is_rgb:
            self.obs = torch.zeros((size, *obs_space.shape), dtype=torch.uint8, device=device)
        elif isinstance(obs_space, gspaces.Discrete):
            self.obs = torch.zeros(size, dtype=torch.int64, device=device)
        else:
            flat_dim = flatdim(obs_space)
            self.obs = torch.zeros((size, flat_dim), device=device)

        if isinstance(action_space, gspaces.MultiBinary):
            self.act = torch.zeros((size, action_space.n), dtype=torch.uint8, device=device)
        elif isinstance(action_space, gspaces.MultiDiscrete):
            self.act = torch.zeros((size, len(action_space.nvec)), dtype=torch.long, device=device)
        elif isinstance(action_space, gspaces.Box):
            self.act = torch.zeros((size, int(np.prod(action_space.shape))),
                                   dtype=torch.float32, device=device)
        else:
            self.act = torch.zeros(size, dtype=torch.long, device=device)
            
        self.rew = torch.zeros(size, device=device)
        self.val = torch.zeros(size, device=device)
        self.logp = torch.zeros(size, device=device)
        self.hid = torch.zeros(size + 1, 1, hidden_dim, device=device)
        self.done = torch.zeros(size, dtype=torch.bool, device=device)
        self.adv = torch.zeros(size, device=device)
        self.ret = torch.zeros(size, device=device)

        self.ptr = 0
        self.path_start = 0

    def store(self, obs: torch.Tensor, act, rew: float, val: float, logp: float, h: torch.Tensor, done):
        i = self.ptr
        if isinstance(self.obs_space, gspaces.Discrete):
            self.obs[i] = obs.squeeze()
        elif self.obs.dim() == 4:
            self.obs[i] = obs.squeeze(0)
        else:
            self.obs[i] = obs.squeeze(0).view(-1)
        
        if isinstance(act, torch.Tensor):
            act = act.detach()

        if self.act.dtype == torch.uint8 and isinstance(act, torch.Tensor) and act.dtype != torch.uint8:
            self.act[i] = act.to(torch.uint8)
        else:
            self.act[i] = act
            
        if isinstance(rew, np.ndarray):
            rew = float(rew.squeeze())
        self.rew[i] = rew
        self.val[i] = val
        self.logp[i] = logp
        self.hid[i] = h.detach()
        self.done[i] = done
        self.ptr += 1

    def finish_path(self, last_val: torch.Tensor, gamma: float = 0.99, lam: float = 0.95):
        sl = slice(self.path_start, self.ptr)
        rews = torch.cat((self.rew[sl], last_val[None]))
        vals = torch.cat((self.val[sl], last_val[None]))
        deltas, gae = rews[:-1] + gamma * vals[1:] - vals[:-1], 0
        for t in reversed(range(len(deltas))):
            gae = deltas[t] + gamma * lam * gae
            self.adv[sl][t] = gae
        self.ret[sl] = self.adv[sl] + self.val[sl]
        self.path_start = self.ptr

    def get(self):
        adv = (self.adv[:self.ptr] - self.adv[:self.ptr].mean()) / (self.adv[:self.ptr].std() + 1e-8)
        return (self.obs[:self.ptr], self.act[:self.ptr], self.logp[:self.ptr], 
                self.ret[:self.ptr], adv, self.hid[:self.ptr])



def pretrain_predictor(model, env, device, steps=1000):
    """Pretrain the predictor to match GRU dynamics."""
    if steps <= 0:
        return
        
    print(f"Pretraining predictor for {steps} steps...")
    optimizer = torch.optim.Adam(model.predict.parameters(), lr=1e-3)
    
    obs, _ = env.reset()
    h = model.init_hidden(device=device)
    
    for step in range(steps):
        obs_t = to_tensor(obs, env.observation_space, device)
        x_enc = model.encoder(obs_t)
        
        # Ensure proper shapes
        if x_enc.dim() == 1:
            x_enc = x_enc.unsqueeze(0)
        if h.dim() == 3 and h.size(0) == 1:
            h_use = h.squeeze(0)
        else:
            h_use = h
        
        # DETACH h_use to prevent graph accumulation
        h_use = h_use.detach()
        
        # Get GRU candidate as target
        with torch.no_grad():
            h_cand, _ = model.sgru.candidate_and_z(x_enc, h_use)
        
        # Train predictor to match
        h_pred = model.predict(h_use)
        loss = F.mse_loss(h_pred, h_cand)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Step environment - DETACH the action to prevent graph accumulation
        with torch.no_grad():
            act, _, _, h = model.act(obs_t, h)
        
        A = env.action_space
        if isinstance(A, gspaces.Discrete):
            step_act = act.item()
        elif isinstance(A, gspaces.MultiDiscrete):
            step_act = act.cpu().numpy().astype(np.int32)
        elif isinstance(A, gspaces.MultiBinary):
            step_act = act.cpu().numpy()
        elif isinstance(A, gspaces.Box):
            step_act = act.detach().cpu().numpy()
            if step_act.shape[0] == 1 and len(A.shape) == 1:
                step_act = step_act.squeeze(0)
        
        obs, rew, term, trunc, _ = env.step(step_act)
        done = term or trunc
        
        if done:
            obs, _ = env.reset()
            h = model.init_hidden(device=device)
            
        if step % 100 == 0:
            print(f"Pretrain step {step}, loss: {loss.item():.4f}")


# ------------------ Modified Training Loop with Surprise Loss -----------------------------

def train(
    env_id: str,
    train_env: gym.Env,
    eval_env: gym.Env,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 500,
    steps: int = 2048,
    log_path: Optional[Path] = None
) -> Tuple[float, int, Optional[Path]]:
    
    # Optional predictor pretraining
    if config.pretrain_predictor_steps > 0:
        pretrain_predictor(model, train_env, device, config.pretrain_predictor_steps)


    best_mmer = -float("inf")
    best_epoch = -1

    for ep in range(epochs):
        model.set_epoch(ep)

        # Calculate warm-up schedule ONCE per epoch
        if config.use_belief_warmup:
            if ep < config.belief_warmup_epochs:
                effective_belief_coef = 0.0
            else:
                # Linearly increase from 0 to config.belief_loss_coef
                progress = min((ep - config.belief_warmup_epochs) / config.belief_ramp_epochs, 1.0)
                effective_belief_coef = config.belief_loss_coef * progress
        else:
            # No warm-up, use constant coefficient
            effective_belief_coef = config.belief_loss_coef

        buf = TrajectoryBuffer(
            steps,
            train_env.observation_space,
            train_env.action_space,
            model.hidden_size,
            device,
        )
        obs, _ = train_env.reset()
        obs = to_tensor(obs, train_env.observation_space, device)
        h = model.init_hidden(device=device)

        rets, lens, norm_lens, succ = [], [], [], []
        kls, clips, gns, etas, alphas = [], [], [], [], []
        thetas, theta_sats, scratch_sats = [], [], []
        surprise_means, surprise_highs, alpha_means, scratch_norms = [], [], [], []
        ep_ret = ep_len = 0

        for t in range(steps):
            act, logp, val, h_next = model.act(obs, h)
            
            # ------------------------------------------------------------------
            # Convert the PyTorch action tensor into something env.step() accepts
            # ------------------------------------------------------------------
            A = train_env.action_space
            if isinstance(A, gspaces.Discrete):
                step_act = act.item()
                
            elif isinstance(A, gspaces.MultiDiscrete):
                # For MultiDiscrete, ensure we get a proper numpy array
                if act.dim() > 1:
                    step_act = act.squeeze().cpu().numpy()
                else:
                    step_act = act.cpu().numpy()
                # Ensure it's the right shape - some envs expect int type
                step_act = step_act.astype(np.int32)
                
            elif isinstance(A, gspaces.MultiBinary):
                step_act = act.cpu().numpy()
                
            elif isinstance(A, gspaces.Box):
                step_act = act.detach().cpu().numpy()
                # Handle potential shape issues for Box actions
                if step_act.shape[0] == 1 and len(A.shape) == 1:
                    step_act = step_act.squeeze(0)
                    
            else:
                raise NotImplementedError(f"Unsupported action space {A}")
                
            obs2, rew, term, trunc, _ = train_env.step(step_act)
            done = term or trunc

            obs2 = to_tensor(obs2, train_env.observation_space, device)
            buf.store(obs, act, rew, val.detach(), logp.detach(), h, done)

            obs, h = obs2, h_next

            if (t + 1) % config.tbptt_len == 0:
                h = h.detach()

            ep_ret += rew
            ep_len += 1

            if done or t == steps - 1:
                h = h.detach()
                _, last_val, _, _, _, _, _, _ = model(obs, h)
                buf.finish_path(torch.tensor(last_val.item(), device=device), 
                               gamma=config.gamma, lam=config.lambda_gae)
                rets.append(ep_ret)
                lens.append(ep_len)
                succ.append(int(float(rew) > 0))
                norm_lens.append(ep_len / episode_horizon(train_env))

                obs, _ = train_env.reset()
                obs = to_tensor(obs, train_env.observation_space, device)
                h = model.init_hidden(device=device)
                ep_ret = ep_len = 0

        obs_b, act_b, logp_b, ret_b, adv_b, hid_b = buf.get()

        pi_losses, v_losses, entropies, belief_losses = [], [], [], []
        
        # Find all episode boundaries
        episode_boundaries = []
        start = 0
        for i in range(buf.ptr):
            if buf.done[i] or i == buf.ptr - 1:
                episode_boundaries.append((start, i + 1))
                if i < buf.ptr - 1:
                    start = i + 1

        #print(f"[Epoch {ep}] Collected {len(episode_boundaries)} episodes in {buf.ptr} steps")

        # Store initial hidden state for each episode
        episode_hidden_states = {}
        for ep_start, _ in episode_boundaries:
            episode_hidden_states[ep_start] = hid_b[ep_start].detach()

        # V2-style: Direct episode grouping
        episodes_per_minibatch = max(1, len(episode_boundaries) // config.num_minibatches)

        # Multiple epochs over the collected data
        for update_epoch in range(config.update_epochs):
            # Shuffle episode order
            shuffled_episodes = list(range(len(episode_boundaries)))
            np.random.shuffle(shuffled_episodes)
            
            # Process mini-batches of episodes
            for mb_start in range(0, len(shuffled_episodes), episodes_per_minibatch):
                mb_end = min(mb_start + episodes_per_minibatch, len(shuffled_episodes))
                mb_episode_indices = shuffled_episodes[mb_start:mb_end]
                
                # Process each episode in this mini-batch
                for ep_idx in mb_episode_indices:
                    ep_start, ep_end = episode_boundaries[ep_idx]
                    
                    # Process this episode with TBPTT if needed
                    L = config.tbptt_len
                    i = ep_start
                    while i < ep_end:
                        j = min(i + L, ep_end)
                        
                        sl = slice(i, j)
                        T = j - i
                        
                        # Prepare tensors
                        obs_slice = obs_b[sl]
                        
                        if isinstance(buf.obs_space, gspaces.Discrete):
                            obs_seq = obs_slice.unsqueeze(0)
                        elif obs_slice.dim() == 3:
                            obs_seq = obs_slice.unsqueeze(0)
                        elif obs_slice.dim() == 2:
                            obs_seq = obs_slice.unsqueeze(0)
                        elif obs_slice.dim() == 1:
                            obs_seq = obs_slice.unsqueeze(0)
                        else:
                            raise ValueError(f"Unexpected obs_slice shape: {obs_slice.shape}")
                        
                        act_seq = act_b[sl].unsqueeze(0)
                        ret_seq = ret_b[sl].unsqueeze(0)
                        adv_seq = adv_b[sl].unsqueeze(0)
                        logp_old = logp_b[sl].unsqueeze(0)
                        
                        # Use stored initial hidden state for this episode
                        h0 = episode_hidden_states[ep_start].unsqueeze(0)
                        
                        # Modified to handle surprise output
                        val, logp, entropy, S_pred, eta, alpha, theta, s_t = \
                            model.evaluate_actions(obs_seq, h0, act_seq)
                        
                        val = val.view(-1)
                        logp = logp.view(-1)
                        entropy = entropy.view(-1)
                        ret_seq = ret_seq.view(-1)
                        adv_seq = adv_seq.view(-1)
                        logp_old = logp_old.view(-1)
                        
                        ratio = torch.exp(logp - logp_old)
                        clip_adv = torch.clamp(ratio, 1 - config.clip_epsilon,
                                              1 + config.clip_epsilon) * adv_seq
                        loss_pi = -(torch.min(ratio * adv_seq, clip_adv)).mean()
                        loss_v = F.mse_loss(val, ret_seq)
                        ent = entropy.mean()
                        
                        # Surprise/belief loss
                        loss_belief = S_pred.pow(2).mean()
                        
                        # Combined loss with belief component
                        loss = (loss_pi + 
                               config.value_loss_coef * loss_v + 
                               config.entropy_coef * ent)
                        
                        optimizer.zero_grad()
                        loss.backward()
                        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip_norm)
                        optimizer.step()
                        
                        pi_losses.append(loss_pi.item())
                        v_losses.append(loss_v.item())
                        entropies.append(ent.item())
                        belief_losses.append(loss_belief.item())
                        
                        with torch.no_grad():
                            kl = (logp_old - logp).mean().item()
                            clip_frac = ((ratio - 1.0).abs() > config.clip_epsilon).float().mean().item()
                            eta_mean = eta.mean().item()
                            alpha_sat = (alpha > 0.9).float().mean().item()
                            theta_mean = theta.mean().item()
                            theta_sat = (theta > config.theta_max * 0.9).float().mean().item()
                            scratch_sat = (s_t.abs() > config.scratch_clip_value * 0.95).float().mean().item()
                            surprise_mean = S_pred.mean().item()
                            surprise_high = (S_pred > S_pred.max() * 0.9).float().mean().item()
                            alpha_mean = alpha.mean().item()
                            scratch_norm = s_t.norm(p=2, dim=-1).mean().item()
                        
                        # store for epoch summary
                        kls.append(kl)
                        clips.append(clip_frac)
                        gns.append(grad_norm.item())
                        etas.append(eta_mean)
                        alphas.append(alpha_sat)
                        thetas.append(theta_mean)
                        theta_sats.append(theta_sat)
                        scratch_sats.append(scratch_sat)
                        surprise_means.append(surprise_mean)
                        surprise_highs.append(surprise_high)
                        alpha_means.append(alpha_mean)
                        scratch_norms.append(scratch_norm)
                        
                        i = j

        m_r = np.mean(rets) if rets else 0
        m_l = np.mean(lens) if lens else 0
        s_r = np.mean(succ) if succ else 0
        m_nl = np.mean(norm_lens) if norm_lens else 0

        if m_r > best_mmer:
            best_mmer = m_r
            best_epoch = ep

        if log_path is not None:
            with log_path.open("a", newline="") as f:
                csv.writer(f).writerow([
                    ep, m_r, m_l, m_nl, s_r,
                    np.mean(pi_losses), np.mean(v_losses), np.mean(entropies),
                    np.mean(belief_losses),
                    np.mean(kls), np.mean(clips), np.mean(gns),
                    np.mean(etas), np.mean(alphas),
                    np.mean(thetas), np.mean(theta_sats), np.mean(scratch_sats),
                    np.mean(surprise_means), np.mean(surprise_highs),  
                    np.mean(alpha_means), np.mean(scratch_norms),      
                    best_mmer, best_epoch
                ])

        if ep % 10 == 0:
            print(f"E{ep:3d} | R {m_r:6.1f} | L {m_l:4.1f} | NL {m_nl:4.2f} | S {s_r:4.1%} | "
                f"π {np.mean(pi_losses):+.3f} | V {np.mean(v_losses):.3f} | "
                f"H {np.mean(entropies):.3f} | B {np.mean(belief_losses):.3f} | "
                f"KL {np.mean(kls):.4f} | Clip {np.mean(clips):.2%} | "
                f"S {np.mean(surprise_means):.2f} (>0.9: {np.mean(surprise_highs):.1%}) | "  
                f"η {np.mean(etas):.2f} | "
                f"α {np.mean(alpha_means):.2f} (>0.9: {np.mean(alphas):.1%}) | "  
                f"θ {np.mean(thetas):.1f} (sat {np.mean(theta_sats):.1%}) | "
                f"Scratch: sat {np.mean(scratch_sats):.1%}, norm {np.mean(scratch_norms):.2f}")  

        if (ep + 1) % 10 == 0:
            rollout_one_episode(eval_env, model.eval(), device, record_dir=Path("videos"))
            model.train()

    return best_mmer, best_epoch, log_path


def setup_logger(env_id: str, config: Config, model_tag: str, trial_num: Optional[int] = None) -> Path:
    import json
    
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    clean_env_id = env_id.replace('/', '_').replace(':', '_')
    
    if trial_num is not None:
        filename = f"{model_tag}_{clean_env_id}_trial{trial_num}_{timestamp}.csv"
    else:
        filename = f"{model_tag}_{clean_env_id}_{timestamp}.csv"
    path = Path("results") / filename

    # Save config as JSON with same base name
    config_path = path.with_suffix('.config.json')
    with open(config_path, 'w') as f:
        json.dump(config.__dict__, f, indent=2)
    
    # Save CSV without config row (cleaner)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        # No config row - it's in the JSON file
        writer.writerow([
            "epoch", "mean_return", "mean_length", "norm_length", "success",
            "pi_loss", "v_loss", "entropy", "belief_loss",
            "kl", "clip_frac", "grad_norm", "eta_mean", "alpha_sat",
            "theta_mean", "theta_sat", "scratch_sat",
            "surprise_mean", "surprise_high", "alpha_mean", "scratch_norm",
            "best_mmer", "best_epoch"
        ])
    
    print(f"Config saved to: {config_path}")
    print(f"Results will be saved to: {path}")
    
    return path


def rollout_one_episode(env: gym.Env, model: nn.Module, device: torch.device,
                        record_dir: Optional[Path] = None, render: bool = False) -> float:
    pixel_env = env.spec and env.spec.id.startswith("MiniGrid")
    if record_dir and pixel_env:
        env = RecordVideo(env, video_folder=str(record_dir),
                          episode_trigger=lambda _: True, name_prefix="eval")
    obs, _ = env.reset()
    h = model.init_hidden(device=device, batch_size=1)
    ep_ret = 0.0
    done = False

    while not done:
        if render and pixel_env:
            env.render()
        obs_t = to_tensor(obs, env.observation_space, device)
        act, _, _, h = model.act(obs_t, h, deterministic=True)
        
        # Convert action appropriately
        A = env.action_space
        if isinstance(A, gspaces.Discrete):
            step_act = act.item()
            
        elif isinstance(A, gspaces.MultiDiscrete):
            # For MultiDiscrete, ensure we get a proper numpy array
            if act.dim() > 1:
                step_act = act.squeeze().cpu().numpy()
            else:
                step_act = act.cpu().numpy()
            step_act = step_act.astype(np.int32)
            
        elif isinstance(A, gspaces.MultiBinary):
            step_act = act.cpu().numpy()
            
        elif isinstance(A, gspaces.Box):
            step_act = act.detach().cpu().numpy()
            # Handle potential shape issues for Box actions
            if step_act.shape[0] == 1 and len(A.shape) == 1:
                step_act = step_act.squeeze(0)
                
        else:
            raise NotImplementedError(f"Unsupported action space {A}")
            
        obs, rew, term, trunc, _ = env.step(step_act)
        ep_ret += rew
        done = term or trunc

    if record_dir and pixel_env:
        env.close()
    return ep_ret


def run_multiple_trials(env_id: str, env_fn, model_class, model_kwargs,
                        optimizer_class, optimizer_kwargs, device: torch.device,
                        base_seed: int, num_trials: int = 5):

    all_mmers = []
    all_log_paths = []

    for trial in range(num_trials):
        trial_seed = base_seed + trial
        print(f"\nTrial {trial + 1}/{num_trials} with seed {trial_seed}")
        set_global_seed(trial_seed)

        train_env = env_fn(env_id)
        train_env.reset(seed=trial_seed)
        eval_env = env_fn(env_id)
        eval_env.reset(seed=trial_seed)

        model = model_class(*model_kwargs).to(device)
        model.apply(init_weights) #this does the better weight initialization
        optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)

        log_path = setup_logger(env_id, config, model_class.__name__, trial_num=trial + 1)

        best_mmer, best_epoch, log_path = train(
            env_id,
            train_env,
            eval_env,
            model,
            optimizer,
            device,
            epochs=config.total_env_steps // config.steps_per_epoch,
            steps=config.steps_per_epoch,
            log_path=log_path
        )

        all_mmers.append(best_mmer)
        all_log_paths.append(log_path)

        train_env.close()
        eval_env.close()

    mean_mmer = np.mean(all_mmers)
    std_mmer = np.std(all_mmers)
    print(f"\nTrial results for {env_id} with {model_class.__name__}:")
    print(f"MMER mean ± std over {num_trials} trials: {mean_mmer:.4f} ± {std_mmer:.4f}\n")

    return all_mmers, all_log_paths


def plot_mean_return_curves(log_paths, model_name: str, env_id: str):
    dfs = []
    for path in log_paths:
        df = pd.read_csv(path, skiprows=1)
        dfs.append(df[["epoch", "mean_return"]].set_index("epoch"))

    combined = pd.concat(dfs, axis=1)
    combined.columns = [f"trial_{i + 1}" for i in range(len(dfs))]

    mean_return = combined.mean(axis=1)
    std_return = combined.std(axis=1)
    epochs = mean_return.index

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, mean_return, label=f"{model_name} Mean Return")
    plt.fill_between(
        epochs,
        mean_return - 1.96 * std_return / np.sqrt(len(dfs)),
        mean_return + 1.96 * std_return / np.sqrt(len(dfs)),
        alpha=0.3
    )
    plt.xlabel("Epoch")
    plt.ylabel("Mean Return")
    plt.title(f"Learning Curve on {env_id}")
    plt.legend()
    plt.grid(True)
    
    os.makedirs("results", exist_ok=True)
    clean_env_id = env_id.replace('/', '_').replace(':', '_')
    plt.savefig(f"results/{clean_env_id}_{model_name}_learning_curve.png")
    plt.show()



# ------------------ Main Execution -----------------------------

if __name__ == "__main__":
    set_global_seed(config.seed)

    if config.USE_CUDA and torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU (CUDA)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    dummy_env = make_env(config.env_id)

    print(f"Environment: {config.env_id}")
    print(f"Observation space: {dummy_env.observation_space}")
    print(f"Action space: {dummy_env.action_space}")
    print(f"Action space type: {type(dummy_env.action_space)}")

    if isinstance(dummy_env.action_space, gym.spaces.MultiDiscrete):
        print(f"MultiDiscrete with nvec: {dummy_env.action_space.nvec}")
    elif isinstance(dummy_env.action_space, gym.spaces.Discrete):
        print(f"Discrete with n: {dummy_env.action_space.n}")

    obs_space = dummy_env.observation_space
    action_space = dummy_env.action_space
    dummy_env.close()

    # Use the Surprise-GRU model
    model_class = PPOSurpriseGRUPolicy
    model_kwargs = (obs_space, action_space)
    optimizer_class = torch.optim.Adam
    optimizer_kwargs = {"lr": config.learning_rate}

    mmers, log_paths = run_multiple_trials(
        config.env_id,
        make_env,
        model_class,
        model_kwargs,
        optimizer_class,
        optimizer_kwargs,
        device,
        base_seed=config.seed,
        num_trials=config.num_trials,
    )

    plot_mean_return_curves(log_paths, model_class.__name__, config.env_id)