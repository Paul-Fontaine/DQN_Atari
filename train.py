import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from DQN import DQN, preprocess_for_dqn, device

print(f"Using {device}")

# Hyperparameters
EPISODES = 10000
GAMMA = 0.99
LEARNING_RATE = 1e-4
BUFFER_SIZE = 100000
BATCH_SIZE = 32
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 1000000
TARGET_UPDATE = 1000
ENV_NAME = "ALE/Breakout-v5"


def train():
    env = gym.make(ENV_NAME, render_mode=None)
    env = gym.wrappers.AtariPreprocessing(env, grayscale_obs=True, frame_skip=1, scale_obs=False)
    env = gym.wrappers.FrameStack(env, 4)

    dqn = DQN((4, 84, 84), env.action_space.n).to(device)
    target_dqn = DQN((4, 84, 84), env.action_space.n).to(device)
    target_dqn.load_state_dict(dqn.state_dict())
    target_dqn.eval()

    optimizer = optim.Adam(dqn.parameters(), lr=LEARNING_RATE)
    memory = deque(maxlen=BUFFER_SIZE)
    epsilon = EPSILON_START
    steps_done = 0

    for episode in range(EPISODES):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            steps_done += 1
            epsilon = max(EPSILON_END, EPSILON_START - (steps_done / EPSILON_DECAY))
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = dqn(preprocess_for_dqn(state)).argmax().item()
            next_state, reward, done, _, _ = env.step(action)
            memory.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            if len(memory) >= BATCH_SIZE:
                batch = random.sample(memory, BATCH_SIZE)
                state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
                state_batch = torch.tensor(np.array(state_batch), dtype=torch.float32, device=device)
                action_batch = torch.tensor(action_batch, dtype=torch.int64, device=device)
                reward_batch = torch.tensor(reward_batch, dtype=torch.float32, device=device)
                next_state_batch = torch.tensor(np.array(next_state_batch), dtype=torch.float32, device=device)
                done_batch = torch.tensor(done_batch, dtype=torch.float32, device=device)

                q_values = dqn(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
                next_q_values = target_dqn(next_state_batch).max(1)[0].detach()
                expected_q_values = reward_batch + GAMMA * next_q_values * (1 - done_batch)

                loss = nn.MSELoss()(q_values, expected_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if steps_done % TARGET_UPDATE == 0:
                target_dqn.load_state_dict(dqn.state_dict())

        print(f"Episode {episode}, Total Reward: {total_reward}")

        if episode % 100 == 0:
            torch.save(dqn.state_dict(), "dqn_checkpoint.pth")

    env.close()


if __name__ == "__main__":
    train()
