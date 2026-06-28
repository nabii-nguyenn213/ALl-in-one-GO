import gymnasium as gym
from .qlearning_agent import QLearningAgent

if __name__ == "__main__":
    env_name = "FrozenLake-v1"
    env = gym.make(env_name, is_slippery=False)
    q_learning_agent = QLearningAgent(
        env=env,
        alpha=0.1,
        gamma=0.99,
        epsilon=0.1,
        seed=42,
    )
    q_learning_rewards = q_learning_agent.train(episodes=5000)
    print("Q-learning Q-table:")
    print(q_learning_agent.Q)
    print("Q-learning evaluation:")
    print(q_learning_agent.evaluate(episodes=20))
    env.close()
