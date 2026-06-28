import gymnasium as gym
from .sarsa_agent import SARSAAgent
from .expected_sarsa_agent import ExpectedSARSAAgent

if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", is_slippery=False)

    sarsa_agent = SARSAAgent(
        env=env,
        alpha=0.1,
        gamma=0.99,
        epsilon=0.1,
        seed=42,
    )

    sarsa_rewards = sarsa_agent.train(episodes=5000)

    print("SARSA Q-table:")
    print(sarsa_agent.Q)

    print("SARSA evaluation:")
    print(sarsa_agent.evaluate(episodes=20))

    env.close()

    env = gym.make("FrozenLake-v1", is_slippery=False)

    expected_sarsa_agent = ExpectedSARSAAgent(
        env=env,
        alpha=0.1,
        gamma=0.99,
        epsilon=0.1,
        seed=42,
    )

    expected_sarsa_rewards = expected_sarsa_agent.train(episodes=5000)

    print("\nExpected SARSA Q-table:")
    print(expected_sarsa_agent.Q)

    print("Expected SARSA evaluation:")
    print(expected_sarsa_agent.evaluate(episodes=20))

    env.close()
