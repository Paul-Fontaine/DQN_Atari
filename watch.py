from agent import AgentDoubleDQN
from env import env, human_env


def watch():
    agent = AgentDoubleDQN(env, human_env)
    agent.load()

    agent.watch()


if __name__ == "__main__":
    watch()
