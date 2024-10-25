import gym
import numpy as np
import time
from td3_tf2 import Agent
from utils import plot_learning_curve

def play(name_game, n_games, isLearn: bool=False):
    hasRender = False
    env = env = gym.make(name_game)
    filename = 'plots/' + 'walker_' + str(n_games) + '_games.png'
    agent = Agent(alpha=0.001, beta=0.001,
            input_dims=env.observation_space.shape[0], tau=0.005,
            env=env, batch_size=100, layer1_size=400, layer2_size=300,
            n_actions=env.action_space.shape[0])
    
    best_score = env.reward_range[0]
    score_history = []
    if not isLearn:
      agent.load_models()
    dem = 0
    for i in range(n_games):
        observation = env.reset()[0]
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if isLearn:
                agent.remember(observation, action, reward, observation_, done)
                agent.learn()
            else:
                env.render()
            score += reward
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if isLearn and avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode ', i, 'score %.1f' % score,
                'average score %.1f' % avg_score)
        
        if hasRender:
            break

        if (not isLearn) and score > 200:
            env = gym.make(name_game, render_mode="human")
            hasRender = True

        dem = i
        if avg_score == 500:
            break
    if isLearn:
        x = [i+1 for i in range(dem+1)]
        plot_learning_curve(x, score_history, filename)

def train(name_game, n_games):
    play(name_game, n_games, isLearn=True)

def demo(name_game, n_games):
    play(name_game, n_games, isLearn=False)

if __name__ == '__main__':
    # name_game = 'LunarLanderContinuous-v2'
    # name_game = 'Pendulum-v0'
    name_game = 'BipedalWalker-v3'
    n_games = 1000
    demo(name_game, n_games)
