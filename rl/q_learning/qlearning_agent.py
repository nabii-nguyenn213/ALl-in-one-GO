import torch 
from ..base_agent import BaseTabularAgent

class QLearningAgent(BaseTabularAgent): 
    def train_one_episode(self, seed=None):
        state, info = self.env.reset(seed=seed)
        state = int(state)
        done = False
        total_reward = 0.0

        while not done:
            # Behavior policy: epsilon-greedy
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
                # Q-learning target uses max_a Q(S', a)
                best_next_q = torch.max(self.Q[next_state])
                td_target = reward + self.gamma * best_next_q
            td_error = td_target - self.Q[state, action]
            self.Q[state, action] += self.alpha * td_error
            state = next_state
        return total_reward
