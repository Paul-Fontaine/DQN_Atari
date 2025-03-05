from agent import AgentDoubleDQN
from env import env, human_env


def evaluate():
    agent = AgentDoubleDQN(env, human_env)
    agent.load()
    agent.evaluate(episodes=10)


if __name__ == "__main__":
    evaluate()
