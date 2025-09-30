import os
import csv
import json
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


# ---------------- Configuration as dataclass -----------------

@dataclass
class Config:
    seed: int = 30
    num_trials: int = 5
    model_name: str = "PPOGRU"
    env_id: str = 'MiniGrid-MemoryS13-v0'
    steps_per_epoch: int = 2048
    mini_batch_size: int = 256
    sgd_passes: int = 4
    learning_rate: float = 3e-4
    clip_epsilon: float = 0.25
    lambda_gae: float = 1.0
    value_loss_coef: float = 1.0
    entropy_coef: float = 0.05
    gamma: float = 0.99
    total_env_steps: int = 15_000_000
    tbptt_len: int = 1024          # <= truncation window (L). 32–256 is typical
    USE_CUDA: bool = True          # set to False to force CPU
    debug: bool = False            # Enable debug prints
    grad_clip_norm: float = 0.5    # Gradient clipping


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


# ------------------ Encoders -----------------------------

class DenseEncoder(nn.Module):
    """Encodes discrete or vector observations into fixed-width embeddings."""
    def __init__(self, obs_space: gym.Space, emb_dim: int = 32):
        super().__init__()
        if isinstance(obs_space, gspaces.Discrete):
            self.mode = "discrete"
            self.embed = nn.Embedding(obs_space.n, emb_dim)
            self.out_dim = emb_dim
        else:
            self.mode = "vector"
            in_dim = flatdim(obs_space)
            self.embed = nn.Linear(in_dim, emb_dim)
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
                # Could be (B,) for batch or (T,) for sequence
                # We'll treat it as (B,) and let the policy add time dim if needed
                return self.embed(x.long())  # (B, emb_dim) or (T, emb_dim)
            elif x.dim() == 2:
                # (B, T) -> embed each element
                return self.embed(x.long())  # (B, T, emb_dim)
            else:
                raise ValueError(f"Unexpected shape for discrete obs: {x.shape}")
        else:
            # Continuous observations
            if x.dim() == 2:
                # (B, F) or (T, F) -> (B, emb_dim) or (T, emb_dim)
                return self.embed(x.float())
            elif x.dim() == 3:
                # (B, T, F) -> need to reshape
                B, T, F = x.shape
                x_flat = x.reshape(B * T, F)  # (B*T, F)
                out_flat = self.embed(x_flat.float())  # (B*T, emb_dim)
                return out_flat.reshape(B, T, self.out_dim)  # (B, T, emb_dim)
            else:
                raise ValueError(f"Unexpected shape for continuous obs: {x.shape}")


class ConvEncoder(nn.Module):
    """Simple 2-layer CNN encoder for image inputs."""
    def __init__(self, V: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
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


# ------------------ PPO-GRU Policy --------------------------

class PPOGRUPolicy(nn.Module):
    """
    One policy that works for any Gymnasium action space:
      • Discrete          → single Categorical head
      • MultiDiscrete     → list of Categorical heads
      • MultiBinary       → single Bernoulli head
      • Box               → Normal distribution with learnable std
    """
    def __init__(self, obs_space: gym.Space, action_space: gspaces.Space, hidden_size: int = 256):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_space = action_space

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

        self.gru = nn.GRU(self.encoder.out_dim, hidden_size, batch_first=True)
        self.value_head = nn.Linear(hidden_size, 1)

        # ---------- action heads ----------
        if isinstance(action_space, gspaces.Discrete):
            self.pi_head = nn.Linear(hidden_size, action_space.n)

        elif isinstance(action_space, gspaces.MultiDiscrete):
            self.pi_head = nn.ModuleList(
                [nn.Linear(hidden_size, int(n)) for n in action_space.nvec]
            )

        elif isinstance(action_space, gspaces.MultiBinary):
            self.pi_head = nn.Linear(hidden_size, action_space.n)

        elif isinstance(action_space, gspaces.Box):
            self.action_dim = int(np.prod(action_space.shape))
            self.mu_head = nn.Linear(hidden_size, self.action_dim)
            # one log-std parameter per dimension
            self.log_std = nn.Parameter(torch.zeros(self.action_dim))

        else:
            raise NotImplementedError(f"Unsupported action space {action_space}")

    def forward(
        self,
        obs: torch.Tensor,
        h: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            obs: Observation tensor
            h: Hidden state tensor (num_layers, batch_size, hidden_size) or None

        Returns:
            logits: Action logits
            value: Value estimate
            h1: Next hidden state
        """
        # ----- encode -----
        x = self.encoder(obs)  # (B, emb_dim) or (B, T, emb_dim)
        
        if config.debug:
            print(f"[DEBUG] After encoder: x.shape = {x.shape}")
        
        # Ensure x has batch and time dimensions for GRU
        if x.dim() == 1:
            # Single observation without batch: (emb_dim,) -> (1, 1, emb_dim)
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 2:
            # Either (B, emb_dim) or (T, emb_dim)
            # We'll assume it's (B, emb_dim) and add time dimension
            x = x.unsqueeze(1)  # (B, 1, emb_dim)
        elif x.dim() == 3:
            # Already has batch and time: (B, T, emb_dim)
            pass
        else:
            raise ValueError(f"Expected encoder output to be 1D, 2D or 3D, got shape: {x.shape}")

        if config.debug and h is not None:
            print(f"[DEBUG] Before GRU: x.shape = {x.shape}, h.shape = {h.shape}")

        # ----- GRU trunk -----
        x, h1 = self.gru(x, h)  # (B, T, H), (num_layers, B, H)

        # ----- policy and value heads -----
        if isinstance(self.action_space, gspaces.Discrete):
            logits = self.pi_head(x)  # (B, T, A)

        elif isinstance(self.action_space, gspaces.MultiDiscrete):
            logits = [head(x) for head in self.pi_head]  # each is (B, T, ni)

        elif isinstance(self.action_space, gspaces.MultiBinary):
            logits = self.pi_head(x)  # (B, T, bits)

        elif isinstance(self.action_space, gspaces.Box):
            logits = self.mu_head(x)  # (B, T, action_dim)

        value = self.value_head(x).squeeze(-1)  # (B, T)
        return logits, value, h1

    def init_hidden(self, batch_size: int = 1, device: str = "cpu") -> torch.Tensor:
        return torch.zeros(1, batch_size, self.hidden_size, device=device)

    # ------------------------------------------------------------
    #   Helpers for sampling / evaluating actions
    # ------------------------------------------------------------
    def _build_dist(self, logits):
        if isinstance(self.action_space, gspaces.Discrete):
            return Categorical(logits=logits)
        if isinstance(self.action_space, gspaces.MultiDiscrete):
            return [Categorical(logits=l) for l in logits]
        if isinstance(self.action_space, gspaces.MultiBinary):
            return Bernoulli(logits=logits)
        if isinstance(self.action_space, gspaces.Box):
            std = self.log_std.exp()
            base = Normal(logits, std)  # logits ≡ μ
            return Independent(base, 1)  # treat dims jointly

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
            action = torch.stack(comps, dim=-1)  # (B, k)
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
            # clamp into Box bounds
            low = torch.as_tensor(self.action_space.low, device=action.device)
            high = torch.as_tensor(self.action_space.high, device=action.device)
            action = torch.max(torch.min(action, high), low)
            return action, logp

    # -------- public helpers used in rollout / PPO update --------
    def act(self, obs, h, deterministic=False):
        logits, v, h1 = self.forward(obs, h)

        if isinstance(self.action_space, gspaces.MultiDiscrete):
            logits_last = [l[:, -1] for l in logits]  # list of (B, ni)
        else:
            logits_last = logits[:, -1]  # (B, A) / (B, bits) / (B, µ)

        v_last = v[:, -1]  # (B,)
        dist = self._build_dist(logits_last)
        act, logp = self._sample(dist, deterministic)
        return act, logp, v_last, h1

    def evaluate_actions(self, obs, h, actions):
        """
        Evaluate actions for PPO loss computation.
        
        Args:
            obs: (B, T, ...) observations
            h: (1, B, H) initial hidden state
            actions: (B, T, ...) actions
        """
        logits, v, _ = self.forward(obs, h)
        dist = self._build_dist(logits)

        if isinstance(self.action_space, gspaces.Discrete):
            logp = dist.log_prob(actions)  # (B, T)
            entropy = dist.entropy()  # (B, T)

        elif isinstance(self.action_space, gspaces.MultiDiscrete):
            logps, entrs = [], []
            for i, d in enumerate(dist):
                logps.append(d.log_prob(actions[..., i]))
                entrs.append(d.entropy())
            logp = torch.stack(logps, dim=-1).sum(-1)  # (B, T)
            entropy = torch.stack(entrs, dim=-1).sum(-1)  # (B, T)

        elif isinstance(self.action_space, gspaces.MultiBinary):
            logp = dist.log_prob(actions.float()).sum(-1)
            entropy = dist.entropy().sum(-1)

        elif isinstance(self.action_space, gspaces.Box):
            logp = dist.log_prob(actions)
            entropy = dist.base_dist.entropy().sum(-1)

        return v, logp, entropy


# ------------------ Trajectory Buffer ------------------------

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
            # For discrete observations, store as 1D tensor of integers
            self.obs = torch.zeros(size, dtype=torch.int64, device=device)
        else:
            flat_dim = flatdim(obs_space)
            self.obs = torch.zeros((size, flat_dim), device=device)

        # Action buffer size depends on action space
        if isinstance(action_space, gspaces.MultiBinary):
            self.act = torch.zeros((size, action_space.n), dtype=torch.uint8, device=device)
        elif isinstance(action_space, gspaces.MultiDiscrete):
            self.act = torch.zeros((size, len(action_space.nvec)), dtype=torch.long, device=device)
        elif isinstance(action_space, gspaces.Box):
            self.act = torch.zeros((size, int(np.prod(action_space.shape))),
                                   dtype=torch.float32, device=device)
        else:  # Discrete
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
            # For discrete observations, store the scalar value
            self.obs[i] = obs.squeeze()
        elif self.obs.dim() == 4:  # RGB (B,H,W,C)
            self.obs[i] = obs.squeeze(0)
        else:  # vector / continuous
            self.obs[i] = obs.squeeze(0).view(-1)
        
        # --------  action  (drop grad graph + dtype)  --------
        if isinstance(act, torch.Tensor):
            act = act.detach()

        if self.act.dtype == torch.uint8 and isinstance(act, torch.Tensor) and act.dtype != torch.uint8:
            self.act[i] = act.to(torch.uint8)  # MultiBinary
        else:
            self.act[i] = act  # Discrete / MD / Box
            
        # reward might be np.ndarray((1,)) on some Box-action POPGym envs
        if isinstance(rew, np.ndarray):
            rew = float(rew.squeeze())  # → scalar
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


# ------------------ Training Loop -----------------------------

def train(
    env_id: str,
    train_env: gym.Env,
    eval_env: gym.Env,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 500,
    steps: int = 2048,
    mini_batch_size: int = 256,
    log_path: Optional[Path] = None
) -> Tuple[float, int, Optional[Path]]:
    best_mmer = -float("inf")
    best_epoch = -1

    for ep in range(epochs):
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
        ep_ret = ep_len = 0

        for t in range(steps):
            act, logp, val, h_next = model.act(obs, h)
            # ------------------------------------------------------------------
            # Convert the PyTorch action tensor into something env.step() accepts
            # ------------------------------------------------------------------
            A = train_env.action_space
            if isinstance(A, gspaces.Discrete):
                step_act = act.item()

            elif isinstance(A, (gspaces.MultiDiscrete, gspaces.MultiBinary)):
                step_act = act.cpu().numpy()

            elif isinstance(A, gspaces.Box):
                step_act = act.detach().cpu().numpy()

            else:
                raise NotImplementedError(f"Unsupported action space {A}")
                
            obs2, rew, term, trunc, _ = train_env.step(step_act)
            done = term or trunc

            obs2 = to_tensor(obs2, train_env.observation_space, device)
            buf.store(obs, act, rew, val.detach(), logp.detach(), h, done)

            obs, h = obs2, h_next

            # Truncate the gradient graph every L steps
            if (t + 1) % config.tbptt_len == 0:
                h = h.detach()

            ep_ret += rew
            ep_len += 1

            if done or t == steps - 1:
                h = h.detach()
                _, last_val, _ = model(obs, h)
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

        pi_losses, v_losses, entropies = [], [], []
        kls, clips, gns = [], [], []  # New metrics
        
        # ------------------------------------------------------------------
        #  TBPTT optimisation: iterate over contiguous (≤ L) slices
        # ------------------------------------------------------------------
        L = config.tbptt_len
        i = 0
        while i < buf.ptr:
            # 1. Choose slice end without crossing an episode boundary
            j = min(i + L, buf.ptr)
            done_slice = buf.done[i : j - 1]
            if done_slice.any():  # truncate at first 'done'
                j = i + done_slice.nonzero(as_tuple=False)[0, 0] + 1

            sl = slice(i, j)
            T = j - i  # actual sequence length (≤ L)

            # 2. Prepare tensors: add batch-dim (B=1) and time-dim (T) if needed
            obs_slice = obs_b[sl]
            
            if config.debug:
                print(f"[DEBUG] obs_slice.shape = {obs_slice.shape}, obs_space = {type(buf.obs_space)}")
            
            # Handle different observation types
            if isinstance(buf.obs_space, gspaces.Discrete):
                # Discrete observations are stored as 1D tensor of integers
                obs_seq = obs_slice.unsqueeze(0)  # (1, T)
            elif obs_slice.dim() == 4:  # RGB observations (T, H, W, C) - ADD THIS CASE
                obs_seq = obs_slice.unsqueeze(0)  # (1, T, H, W, C)
            elif obs_slice.dim() == 3:  # Could be RGB (H, W, C) for single timestep
                obs_seq = obs_slice.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W, C)
            elif obs_slice.dim() == 2:  # Vector observations (T, F)
                obs_seq = obs_slice.unsqueeze(0)  # (1, T, F)
            elif obs_slice.dim() == 1:  # Could be discrete stored as (T,)
                obs_seq = obs_slice.unsqueeze(0)  # (1, T)
            else:
                raise ValueError(f"Unexpected obs_slice shape: {obs_slice.shape}")
            
            act_seq = act_b[sl].unsqueeze(0)  # (1, T, ...)
            ret_seq = ret_b[sl].unsqueeze(0)
            adv_seq = adv_b[sl].unsqueeze(0)
            logp_old = logp_b[sl].unsqueeze(0)

            # 3. Initial hidden h0 = *detached* state recorded at time i
            # hid_b[sl.start] has shape (1, hidden_dim), we need (1, 1, hidden_dim) for GRU
            h0 = hid_b[sl.start].unsqueeze(0).detach()  # (1, 1, H)
            
            if config.debug:
                print(f"[DEBUG] Before evaluate_actions: obs_seq.shape = {obs_seq.shape}, h0.shape = {h0.shape}, act_seq.shape = {act_seq.shape}")

            # 4. Forward & losses
            val, logp, entropy = model.evaluate_actions(obs_seq, h0, act_seq)
            val = val.view(-1)  # flatten (1,T) -> (T,)
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
            loss = loss_pi + config.value_loss_coef * loss_v - config.entropy_coef * ent

            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip_norm)  # Gradient clipping
            optimizer.step()

            # Track new metrics
            with torch.no_grad():
                kl = (logp_old - logp).mean().item()
                clip_frac = ((ratio - 1.0).abs() > config.clip_epsilon).float().mean().item()

            # 5. Logging
            pi_losses.append(loss_pi.item())
            v_losses.append(loss_v.item())
            entropies.append(ent.item())
            kls.append(kl)
            clips.append(clip_frac)
            gns.append(grad_norm.item())

            # 6. Move to next slice
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
                    np.mean(kls), np.mean(clips), np.mean(gns),
                    best_mmer, best_epoch
                ])

        if ep % 10 == 0:
            print(f"E{ep:3d} | R {m_r:6.1f} | L {m_l:4.1f} | NL {m_nl:4.2f} | S {s_r:4.1%} | "
                  f"π {np.mean(pi_losses):+.3f} | V {np.mean(v_losses):.3f} | H {np.mean(entropies):.3f} | "
                  f"KL {np.mean(kls):.4f} | Clip {np.mean(clips):.2%} | GN {np.mean(gns):.2f}")

        if (ep + 1) % 10 == 0:
            rollout_one_episode(eval_env, model.eval(), device, record_dir=Path("videos"))
            model.train()

    return best_mmer, best_epoch, log_path


# ------------------ Logging and Rollouts ------------------------

def setup_logger(env_id: str, config: Config, model_tag: str, trial_num: Optional[int] = None) -> Path:
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Clean up env_id for filename
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
            "pi_loss", "v_loss", "entropy", "kl", "clip_frac", "grad_norm",
            "best_mmer", "best_epoch"
        ])
    
    print(f"Config saved to: {config_path}")
    print(f"Results will be saved to: {path}")
    
    return path


def rollout_one_episode(env: gym.Env, model: nn.Module, device: torch.device,
                        record_dir: Optional[Path] = None, render: bool = False) -> float:
    pixel_env = env.spec and env.spec.id.startswith("MiniGrid")
    if record_dir and pixel_env:
        # Create the video directory if it doesn't exist
        os.makedirs(record_dir, exist_ok=True)
        
        # Recreate environment with rgb_array render mode for video recording
        env_id = env.spec.id
        env = make_env(env_id, render=True)  # This creates env with render_mode="rgb_array"
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
        elif isinstance(A, (gspaces.MultiDiscrete, gspaces.MultiBinary)):
            step_act = act.cpu().numpy()
        elif isinstance(A, gspaces.Box):
            step_act = act.detach().cpu().numpy()
        else:
            raise NotImplementedError(f"Unsupported action space {A}")
            
        obs, rew, term, trunc, _ = env.step(step_act)
        ep_ret += rew
        done = term or trunc

    if record_dir and pixel_env:
        env.close()
    return ep_ret


# ------------------ Running Multiple Trials ---------------------

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
            mini_batch_size=config.mini_batch_size,
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


# ------------------ Plot Learning Curves ------------------------

def plot_mean_return_curves(log_paths, model_name: str, env_id: str):
    dfs = []
    for path in log_paths:
        df = pd.read_csv(path)  # No skiprows needed since we removed config row
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
    # Clean up env_id for filename
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

    model_class = PPOGRUPolicy
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