from typing import Tuple
from collections import deque
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.optim import Adam
from torch.nn import MSELoss
from random import random, sample
from DQN import DQN, preprocess_for_dqn, device


class AgentDoubleDQN:
    def __init__(self, env, human_env, device=device, buffer_size: int = 100000):
        self.env = env
        self.human_env = human_env
        input_shape = env.observation_space.shape
        self.dqn = DQN(input_shape, env.action_space.n).to(device)
        self.target_dqn = DQN(input_shape, env.action_space.n).to(device)
        self.target_dqn.load_state_dict(self.dqn.state_dict())
        self.target_dqn.eval()
        self.memory = deque(maxlen=buffer_size)

    def act(self, state: np.ndarray) -> int:
        with torch.no_grad():
            return self.dqn(preprocess_for_dqn(state)).argmax().item()

    def train(self, episodes: int = 10000, gamma: float = 0.99, learning_rate: float = 1e-4, batch_size: int = 32, epsilon_start: float = 1.0, epsilon_end: float = 0.1, epsilon_decay: int = 100000, target_update: int = 1000, chekpoint_frequency: int = 100, rendering_frequency: int = 0):
        start_time = time.time()
        optimizer = Adam(self.dqn.parameters(), lr=learning_rate)
        epsilon = epsilon_start
        steps_done = 0
        rewards = []

        for episode in range(episodes):
            state, info = self.env.reset()
            previous_nb_lives = info["lives"]
            self.env.step(1)  # fire the ball
            total_reward = 0
            done = False

            while not done:
                steps_done += 1
                epsilon = max(epsilon_end, epsilon_start - (steps_done / epsilon_decay))

                # if the ball is not in play, fire it
                if info["lives"] < previous_nb_lives:
                    action = 1
                    previous_nb_lives = info["lives"]

                if random() < epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = self.act(state)
                next_state, reward, done, _, info = self.env.step(action)
                self.memory.append((state, action, reward, next_state, done))
                state = next_state
                total_reward += reward

                if len(self.memory) == batch_size:
                    print("Memory filled, start training")
                if len(self.memory) >= batch_size:
                    batch = sample(self.memory, batch_size)
                    state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
                    state_batch = torch.tensor(np.array(state_batch), dtype=torch.float32, device=device)
                    action_batch = torch.tensor(action_batch, dtype=torch.int64, device=device)
                    reward_batch = torch.tensor(reward_batch, dtype=torch.float32, device=device)
                    next_state_batch = torch.tensor(np.array(next_state_batch), dtype=torch.float32, device=device)
                    done_batch = torch.tensor(done_batch, dtype=torch.float32, device=device)

                    q_values = self.dqn(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
                    next_q_values = self.target_dqn(next_state_batch).max(1)[0].detach()
                    expected_q_values = reward_batch + gamma * next_q_values * (1 - done_batch)

                    loss = MSELoss()(q_values, expected_q_values)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if steps_done % target_update == 0:
                    self.target_dqn.load_state_dict(self.dqn.state_dict())

            print(f"Episode {episode}, Total Reward: {total_reward}")

            if episode % chekpoint_frequency == 0:
                self.save()
                rewards.append(total_reward)

                avg_speed = (time.time() - start_time) / (episode + 1)
                time_left = (episodes - episode - 1) * avg_speed
                # convert time left from sec to hours minutes and seconds
                h, m, s = time_left // 3600, (time_left % 3600) // 60, time_left % 60
                print(f"Average Speed: {avg_speed:.2f} s/episode, Time Left: {int(h)}h {int(m)}m {int(s)}s")

            if rendering_frequency and episode % rendering_frequency == 0:
                self.watch()

        self.env.close()

        # plot rewards
        plt.figure(figsize=(10, 5))
        plt.plot(rewards)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        # set x ticks to display 10 values
        plt.xticks(np.arange(0, episodes, episodes // 10))
        plt.show()

    def evaluate(self, episodes: int = 10):
        total_rewards = []
        for episode in range(episodes):
            state, info = self.env.reset()
            previous_nb_lives = info["lives"]
            self.env.step(1)  # fire the ball
            total_reward = 0
            done = False

            while not done:
                if info["lives"] < previous_nb_lives:
                    action = 1
                    previous_nb_lives = info["lives"]
                else:
                    action = self.act(state)
                next_state, reward, done, _, info = self.env.step(action)
                state = next_state
                total_reward += reward

            total_rewards.append(total_reward)
            print(f"Episode {episode}, Total Reward: {total_reward}")

        print(f"Average Reward over 10 episodes: {np.mean(total_rewards)}")
        self.env.close()

    def watch(self):
        state, info = self.human_env.reset()
        previous_nb_lives = info["lives"]
        self.human_env.step(1)  # fire the ball
        done = False

        while not done:
            if info["lives"] < previous_nb_lives:
                action = 1
                previous_nb_lives = info["lives"]
            else:
                action = self.act(state)
            state, _, done, _, info = self.human_env.step(action)

        self.human_env.close()

    def save(self, path: str = "agent_checkpoint.pth"):
        torch.save(self.dqn.state_dict(), path)

    def load(self, path: str = "agent_checkpoint.pth"):
        self.dqn.load_state_dict(torch.load(path, map_location=device))
        self.target_dqn.load_state_dict(self.dqn.state_dict())
