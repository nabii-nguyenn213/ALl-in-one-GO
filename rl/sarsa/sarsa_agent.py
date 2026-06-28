import torch
from ..base_agent import BaseTabularAgent

class SARSAAgent(BaseTabularAgent): 
    def train_one_episode(self, seed=None):
        state, info = self.env.reset(seed=seed)
        state = int(state)
        action = self.select_action(state)
        action = int(action)

        done = False 
        total_reward = 0.0

        while not done: 
            next_state, reward, terminated, truncated, info = self.env.step(action)
            next_state = int(next_state)
            done = terminated or truncated

            total_reward += reward 

            if done: 
                td_target = torch.tensor(reward, dtype=torch.float32)
                next_state = None 
            else: 
                next_action = self.select_action(next_state)
                next_action = int(next_action)
                td_target = reward + self.gamma * self.Q[next_state, next_action] 

            td_error = td_target - self.Q[state, action]
            self.Q[state, action] += self.alpha * td_error

            state = next_state 
            if next_action is not None: 
                action = next_action
        return total_reward
