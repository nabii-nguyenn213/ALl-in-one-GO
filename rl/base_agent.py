import torch
import random 
import numpy as np 
import gymnasium as gym
from abc import ABC, abstractmethod
from .utils import seed_everything

class BaseTabularAgent: 
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1, seed=42): 
        self.env = env 
        self.alpha = alpha 
        self.gamma = gamma 
        self.epsilon = epsilon
        self.seed = seed 

        seed_everything(seed)

        assert hasattr(env.observation_space, "n"), "State space must be discrete."
        assert hasattr(env.action_space, "n"), "Action space must be discrete."

        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n

        self.Q = torch.zeros(
            self.n_states,
            self.n_actions,
            dtype=torch.float32,
        )

    def action_probs(self, state): 
        probs = torch.ones(self.n_actions) * (self.epsilon/self.n_actions)
        q_values = self.Q[state]
        max_q = torch.max(q_values)
        best_actions = torch.where(q_values==max_q)[0]
        for action in best_actions: 
            probs[action] += (1.0-self.epsilon) / len(best_actions)
        return probs

    def select_action(self, state): 
        probs = self.action_probs(state)
        action = torch.multinomial(probs, num_samples=1).item()
        return action

    def greedy_action(self, state): 
        return torch.argmax(self.Q[state]).item()

    def train(self, episodes=5000): 
        episode_rewards = []
        for episode in range(episodes): 
            total_reward = self.train_one_episode(seed=self.seed + episode)
            episode_rewards.append(total_reward)
        return episode_rewards

    def train_one_episode(self, seed = None): 
        raise NotImplementedError 

    def evaluate(self, episodes=10): 
        rewards = []
        for episode in range(episodes): 
            state, info = self.env.reset(seed=self.seed+10000+episode)
            done = False 
            total_reward = 0 
            while not done: 
                action = self.greedy_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                total_reward += reward 
                state = next_state 
            rewards.append(total_reward)
        return sum(rewards)/len(rewards)

class BaseDeepAgent(ABC): 
    def __init__(self, env, gamma=0.99, device="auto", seed=42): 
        self.env = env 
        self.gamma = gamma
        self.seed = seed 

        seed_everything(seed=seed)

        if device=="auto": 
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else: 
            self.device = torch.device(device)
        self.obs_dim = env.observation_space.shape[0]
        if hasattr(env.action_space, "n"): 
            self.is_discrete = True 
            self.action_dim = env.action_space.n 
        else: 
            self.is_discrete = False 
            self.action_dim = env.action_space.shape[0]

    @abstractmethod
    def select_action(self, state, evaluate=False): 
        pass

    @abstractmethod
    def train(self): 
        pass 

    def evaluate(self, episodes = 10): 
        rewards = []
        for ep in range(episodes): 
            state, info = self.env.reset(seed=self.seed+10000+ep)
            done = False 
            total_reward=0.0

            while not done: 
                action = self.select_action(state, evaluate=True)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                total_reward += reward
                state = next_state
            rewards.append(total_reward)
        return sum(rewards) / len(rewards)

class BaseOnPolicyAgent(BaseDeepAgent): 
    def __init__(self, env, gamma=0.99, device="auto", seed=42):
        super().__init__(env=env, gamma=gamma, device=device, seed=seed)

    def to_tensor(self, x):
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    @abstractmethod
    def update(self, rollout):
        pass

class BaseOffPolicyAgent(BaseDeepAgent): 
    def __init__(self, env, gamma=0.99, device="auto", seed=42, batch_size=256, learning_start=1000):
        super().__init__(env=env, gamma=gamma, device=device, seed=seed)
        self.batch_size = batch_size
        self.learning_start = learning_start

    def to_tensor(self, x):
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    @abstractmethod
    def update(self, batch): 
        pass
