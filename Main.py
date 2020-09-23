import gym
from DQNClasses import Agent
import numpy as np

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    agent = Agent(gamma = 0.99, eps = 0.05, lr = 0.0001, n_actions = env.action_space.n, max_mem = 100000, batch_size = 512, obs_dim = env.observation_space.sample().shape[0])
    scores, eps_history = [], []
    n_games = 0

    while True:
        score = 0
        done = False
        observation = env.reset()
        timer = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transitions(observation,action,reward,observation_,done)
            agent.learn()
            observation = observation_
            timer += 1
            # if i > 495:
            #     env.render()
        # print(reward)
        n_games += 1
        scores.append(score)
        avg_score = np.mean(scores[-10:])
        if avg_score > 400:
            break
        print("Episode: {}, Score: {}, Avg Score: {}, Timer: {}".format(n_games + 1, score, avg_score, timer))


    agent.eps = 0
    while True:
        done = False
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            observation = observation_
            env.render()
