import torch 
from ..base_agent import BaseTabularAgent

class ExpectedSARSAAgent(BaseTabularAgent): 
    def train_one_episode(self, seed=None):
        state, info = self.env.reset(seed=seed)
        state = int(state)

        done = False
        total_reward = 0.0
        while not done:
            action = self.select_action(state)
            action = int(action)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            next_state = int(next_state)
            reward = float(reward)
            done = terminated or truncated
            total_reward += reward
            if done:
                td_target = torch.tensor(reward, dtype=torch.float32)
            else:
                next_action_probs = self.action_probs(next_state)
                expected_next_q = torch.sum(next_action_probs * self.Q[next_state])
                td_target = reward + self.gamma * expected_next_q
            td_error = td_target - self.Q[state, action]
            self.Q[state, action] += self.alpha * td_error
            state = next_state
        return total_reward
