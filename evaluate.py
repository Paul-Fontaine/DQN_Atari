import gymnasium as gym
import torch
import numpy as np
from train import ENV_NAME
from DQN import DQN, preprocess_for_dqn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")


def evaluate():
    env = gym.make(ENV_NAME, render_mode=None)
    env = gym.wrappers.AtariPreprocessing(env, grayscale_obs=True, frame_skip=1, scale_obs=False)
    env = gym.wrappers.FrameStack(env, 4)

    dqn = DQN((4, 84, 84), env.action_space.n).to(device)
    dqn.load_state_dict(torch.load("dqn_checkpoint.pth", map_location=device))
    dqn.eval()

    total_rewards = []
    for episode in range(10):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            with torch.no_grad():
                action = dqn(preprocess_for_dqn(state)).argmax().item()
            next_state, reward, done, _, _ = env.step(action)
            state = next_state
            total_reward += reward

        total_rewards.append(total_reward)
        print(f"Episode {episode}, Total Reward: {total_reward}")

    print(f"Average Reward over 10 episodes: {np.mean(total_rewards)}")
    env.close()


if __name__ == "__main__":
    evaluate()
