import gymnasium as gym

ENV_NAME = "ALE/Breakout-v5"


def get_training_env_and_human_env(env_name=ENV_NAME):
    env = gym.make(env_name, render_mode=None)
    env = gym.wrappers.AtariPreprocessing(env, grayscale_obs=True, frame_skip=1, scale_obs=True)
    env = gym.wrappers.FrameStack(env, 4)

    human_env = gym.make(env_name, render_mode="human")
    human_env = gym.wrappers.AtariPreprocessing(human_env, grayscale_obs=True, frame_skip=1, scale_obs=False)
    human_env = gym.wrappers.FrameStack(human_env, 4)

    return env, human_env


env, human_env = get_training_env_and_human_env()
