import gymnasium as gym
import torch
from train import ENV_NAME
from DQN import DQN, preprocess_for_dqn


def watch():
    env = gym.make(ENV_NAME, render_mode="human")
    env = gym.wrappers.AtariPreprocessing(env, grayscale_obs=True, frame_skip=1, scale_obs=False)
    env = gym.wrappers.FrameStack(env, 4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dqn = DQN((4, 84, 84), env.action_space.n).to(device)
    dqn.load_state_dict(torch.load("dqn_checkpoint.pth", map_location=device))
    dqn.eval()

    state, _ = env.reset()
    done = False

    while not done:
        with torch.no_grad():
            action = dqn(preprocess_for_dqn(state)).argmax().item()
        state, _, done, _, _ = env.step(action)

    env.close()


if __name__ == "__main__":
    watch()
