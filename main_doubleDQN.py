import argparse
from collections import deque, namedtuple
from typing import Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time
import os

# -------------------------- Hyperparameters -------------------------- #
# DEFAULTS = dict(
#     gamma=0.99,
#     lr=1e-3,
#     buffer_size=100_000,
#     batch_size=128,
#     epsilon_start=1.0,
#     epsilon_end=0.05,
#     epsilon_decay_steps=250_000,  # frames, not episodes
#     target_update_tau=0.005,
#     train_freq=4,  # gradient step every n env steps
#     warmup_steps=10_000,
#     episodes=1000,
#     max_steps=1000,
#     eval_interval=50,
#     seed=42,
# )

DEFAULTS = dict(
    gamma=0.99,
    lr=3e-4,                    # ↓ lower LR for stabler updates
    buffer_size=100_000,
    batch_size=256,             # ↑ larger mini-batches
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay_steps=75_000, # ↓ faster ε-decay (≈300 eps → ε≈0.1)
    target_update_tau=0.001,    # ↓ slightly faster target tracking
    train_freq=1,               # update every env step
    warmup_steps=10_000,
    episodes=1000,
    max_steps=1000,
    eval_interval=50,
    seed=42,
)

Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)


# ----------------------------- Network ----------------------------- #
class QNetwork(nn.Module):
    def __init__(self, obs_size: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 256), #first hidden
            nn.ReLU(), #activation
            nn.Linear(256, 256), # second hidden
            nn.ReLU(),  #activation
            nn.Linear(256, n_actions),  # output layer
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------- ReplayBuffer ---------------------------- #
class ReplayBuffer:
    def __init__(self, capacity: int, obs_shape: Tuple[int]):
        self.capacity = capacity
        self.obs_buf = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.next_obs_buf = np.zeros_like(self.obs_buf)
        self.acts_buf = np.zeros((capacity, 1), dtype=np.int64)
        self.rews_buf = np.zeros((capacity, 1), dtype=np.float32)
        self.done_buf = np.zeros((capacity, 1), dtype=np.float32)
        self.idx = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done):
        self.obs_buf[self.idx] = state
        self.acts_buf[self.idx] = action
        self.rews_buf[self.idx] = reward
        self.next_obs_buf[self.idx] = next_state
        self.done_buf[self.idx] = done
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device):
        idxs = np.random.randint(0, self.size, size=batch_size)
        obses = torch.tensor(self.obs_buf[idxs], device=device)
        actions = torch.tensor(self.acts_buf[idxs], device=device)
        rewards = torch.tensor(self.rews_buf[idxs], device=device)
        next_obses = torch.tensor(self.next_obs_buf[idxs], device=device)
        dones = torch.tensor(self.done_buf[idxs], device=device)
        return obses, actions, rewards, next_obses, dones


# -------------------------- DQN Agent -------------------------- #
class DQNAgent:
    def __init__(self, env, cfg):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() and cfg.cuda else "cpu")

        # obs_size = env.single_observation_space.shape[0]
        # n_actions = env.single_action_space.n
        
        if hasattr(env, "single_observation_space"):      # vector-env path
            obs_size  = env.single_observation_space.shape[0]
            n_actions = env.single_action_space.n
        else:                                             # classic env
            obs_size  = env.observation_space.shape[0]
            n_actions = env.action_space.n

        self.q_net = QNetwork(obs_size, n_actions).to(self.device)
        self.target_net = QNetwork(obs_size, n_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=cfg.lr)
        self.buffer = ReplayBuffer(cfg.buffer_size, (obs_size,))
        self.cfg = cfg
        self.steps_done = 0
        self.epsilon = cfg.epsilon_start

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            if hasattr(self.env, "single_action_space"):
                return self.env.single_action_space.sample()   # vector env
            else:
                return self.env.action_space.sample()          # classic env
        
        
        with torch.no_grad():
            state = torch.tensor(state, device=self.device).unsqueeze(0)
            q_values = self.q_net(state)
            return int(q_values.argmax(dim=1).item())

    def update_epsilon(self):
        self.epsilon = max(
            self.cfg.epsilon_end,
            self.cfg.epsilon_start
            - (self.cfg.epsilon_start - self.cfg.epsilon_end)
            * (self.steps_done / self.cfg.epsilon_decay_steps),
        )

    def learn(self):
        # wait until we have enough data / past warm-up
        if self.buffer.size < self.cfg.batch_size or self.steps_done < self.cfg.warmup_steps:
            return

        (states, actions, rewards, next_states, dones) = \
            self.buffer.sample(self.cfg.batch_size, self.device)

        # 1. current Q-values
        q_vals = self.q_net(states).gather(1, actions)

        # 2. --------- Double DQN target ---------
        #   a) action selection with *online* network
        with torch.no_grad():
            next_actions = self.q_net(next_states).argmax(1, keepdim=True)

            #   b) action evaluation with *target* network
            next_q_vals = self.target_net(next_states).gather(1, next_actions)

            target = rewards + self.cfg.gamma * (1 - dones) * next_q_vals
        # ----------------------------------------

        loss = nn.functional.smooth_l1_loss(q_vals, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        # soft-update target network (unchanged)
        with torch.no_grad():
            for p, tp in zip(self.q_net.parameters(), self.target_net.parameters()):
                tp.data.lerp_(p.data, self.cfg.target_update_tau)

    def train(self, writer: SummaryWriter):
        global_step = 0
        for ep in range(self.cfg.episodes):
            state, _ = self.env.reset(seed=self.cfg.seed)
            ep_reward = 0.0
            for t in range(self.cfg.max_steps):
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                self.buffer.add(state, action, reward, next_state, float(done))

                state = next_state
                ep_reward += reward
                self.learn()

                self.steps_done += 1
                global_step += 1
                self.update_epsilon()

                if done:
                    break

            writer.add_scalar("train/episode_reward", ep_reward, ep)
            writer.add_scalar("train/epsilon", self.epsilon, ep)

            if ep % self.cfg.eval_interval == 0:
                avg_r = self.evaluate(5)
                writer.add_scalar("eval/avg_reward", avg_r, ep)
                print(f"Episode {ep:4d} | Reward {ep_reward:7.2f} | Eval {avg_r:7.2f} | ε {self.epsilon:.3f}")

            if ep % 100 == 0:
                # If you made watch a *method* of the agent:
                self.watch(episodes=1)

            if (ep + 1) % 100 == 0:
                self.save()
                # self.save(f"checkpoint_ep{ep+1}.pth")
                

    def evaluate(self, episodes=5):
        total_r = 0.0
        for _ in range(episodes):
            state, _ = self.env.reset()
            done = False
            while not done:
                with torch.no_grad():
                    state_t = torch.tensor(state, device=self.device).unsqueeze(0)
                    action = int(self.q_net(state_t).argmax(dim=1).item())
                state, reward, terminated, truncated, _ = self.env.step(action)
                total_r += reward
                done = terminated or truncated
        return total_r / episodes
    
    def watch(agent, episodes=3):
    # A fresh env that opens a Pygame window
        vis_env = gym.make("LunarLander-v3",
                       continuous=False,
                       render_mode="human")   # <- key line
        for ep in range(episodes):
            state, _ = vis_env.reset(seed=agent.cfg.seed)
            done = False
            ep_ret = 0
            while not done:
                with torch.no_grad():
                    a = int(agent.q_net(torch.tensor(state,
                                                 device=agent.device)
                                        .unsqueeze(0)).argmax())
                state, r, term, trunc, _ = vis_env.step(a)
                done = term or trunc
                ep_ret += r
                time.sleep(0.03)  # ~30 FPS; adjust for your monitor
            print(f"Episode {ep}: reward {ep_ret:.1f}")
        vis_env.close()

    def save(self, path=None):
        if path is None:
            path = os.path.join(self.cfg.checkpoint_dir, f"checkpoint_ep{self.steps_done}.pth")
        torch.save(
            {
                "q_net": self.q_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(checkpoint["q_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])


# ---------------------------- Main script ---------------------------- #

def make_env(seed: int, vectorized: bool = False):
    if vectorized:
        env = gym.vector.SyncVectorEnv(
            [lambda: gym.make("LunarLander-v3", continuous=False) for _ in range(4)]
        )
    else:
        env = gym.make("LunarLander-v3", continuous=False)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def parse_args():
    parser = argparse.ArgumentParser(description="DQN for LunarLander‑v3")
    parser.add_argument("--episodes", type=int, default=DEFAULTS["episodes"])
    parser.add_argument("--cuda", action="store_true", help="Use GPU if available")
    parser.add_argument("--logdir", type=str, default="runs/dqn_lunarlander")
    return parser.parse_args()

def run_at_checkpoint():
    args = parse_args()
    # cfg = argparse.Namespace(**DEFAULTS, **vars(args))
    cfg_dict = {**DEFAULTS, **vars(args)}   # later values overwrite earlier ones
    cfg = argparse.Namespace(**cfg_dict) 
    env = make_env(cfg.seed)
    agent = DQNAgent(env, cfg)
    agent.load("checkpoint_ep300.pth")
    agent.watch(episodes=3)


def main():
    import os
    from datetime import datetime
    from torch.utils.tensorboard import SummaryWriter

    args = parse_args()
    
    # Create unique run name using timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"dqn_lunarlander_{timestamp}"

    # Create directories for checkpoints and logs
    checkpoint_dir = os.path.join("checkpoints", run_name)
    log_dir = os.path.join("runs", run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Update config with logging and checkpoint paths
    cfg_dict = {**DEFAULTS, **vars(args)}
    cfg_dict["checkpoint_dir"] = checkpoint_dir
    cfg_dict["logdir"] = log_dir
    cfg = argparse.Namespace(**cfg_dict)

    # Create environment and writer
    env = make_env(cfg.seed)
    writer = SummaryWriter(cfg.logdir)

    agent = DQNAgent(env, cfg)
    try:
        agent.train(writer)
    finally:
        env.close()
        writer.close()



if __name__ == "__main__":
    main()
