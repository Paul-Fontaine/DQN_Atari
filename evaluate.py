from agent import AgentDoubleDQN
from env import env, human_env


def evaluate(episodes: int = 10):
    agent = AgentDoubleDQN(env, human_env)
    agent.load()

    total_rewards = []
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            state = next_state
            total_reward += reward

        total_rewards.append(total_reward)
        print(f"Episode {episode}, Total Reward: {total_reward}")

    average_reward = sum(total_rewards) / episodes
    print(f"Average Reward over 10 episodes: {average_reward}")
    env.close()


if __name__ == "__main__":
    evaluate()
