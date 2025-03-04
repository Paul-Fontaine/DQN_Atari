from agent import AgentDoubleDQN
from env import env, human_env


def watch():
    agent = AgentDoubleDQN(env, human_env)
    agent.load()

    state, _ = human_env.reset()
    done = False

    while not done:
        action = agent.act(state)
        state, _, done, _, _ = human_env.step(action)

    human_env.close()


if __name__ == "__main__":
    watch()
