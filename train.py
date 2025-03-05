from agent import AgentDoubleDQN, device
from env import get_training_env_and_human_env

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
CHECKPOINT_FREQUENCY = 100
RENDERING_FREQUENCY = 2
ENV_NAME = "ALE/Breakout-v5"


def train_agent():
    env, human_env = get_training_env_and_human_env(ENV_NAME)
    agent = AgentDoubleDQN(env, human_env, device, BUFFER_SIZE)
    print(f"Using {device}")
    print(f"Number of trainable parameters: {sum(p.numel() for p in agent.dqn.parameters() if p.requires_grad)}")
    agent.train(EPISODES, GAMMA, LEARNING_RATE, BATCH_SIZE, EPSILON_START, EPSILON_END, EPSILON_DECAY, TARGET_UPDATE, CHECKPOINT_FREQUENCY, RENDERING_FREQUENCY)
    agent.save()


if __name__ == "__main__":
    train_agent()
