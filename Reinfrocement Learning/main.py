from RL_model import DQN
import gym

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    env = gym.make('CartPole-v1')
    env = env.unwrapped

    ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape

    dqn = DQN(N_STATES = env.observation_space.shape[0],
              N_ACTIONS = env.action_space.n,
              MEMORY_CAPACITY = 2000,
              LR = 0.01,
              EPSILON = 0.9)

    for i_episode in range(400):
        s = env.reset()
        ep_r = 0
        while True:
            env.render()
            a = dqn.choose_action(s, ENV_A_SHAPE)

            # take action
            s_, r, done, info = env.step(a)

            # modify the reward
            x, x_dot, theta, theta_dot = s_
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            r = r1 + r2

            dqn.store_transition(s, a, r, s_)

            ep_r += r
            if dqn.memory_counter > dqn.memory_capacity:
                dqn.learn(TARGET_REPLACE_ITER = 100, GAMMA = 0.9, BATCH_SIZE = 32)
                if done:
                    print('Ep: ', i_episode,
                          '| Ep_r: ', round(ep_r, 2))

            if done:
                break
            s = s_

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
