import argparse
import json
import numpy as np
import os
import shutil
from copy import deepcopy
from tqdm import tqdm
import time
import numpy as np
import torch
from plot import *
from Auction import Auction

from agents.Random.Random import Random
from agents.Constant.Constant import Constant
from agents.DQN.DQN import DQN


def draw_features():
    run2item_features = np.zeros((num_runs,num_agents,feature_dim))

    for run in range(num_runs):
        for agent in range(num_agents):
            run2item_features[run,agent] = rng.normal(0.0, 1.0, size=feature_dim)

    return run2item_features

def instantiate_agents():
    agents = []
    for name,config in agent_list.items():
        constructor_call = config['constructor_call']
        agents.append(eval(constructor_call))
    return agents

if __name__ == '__main__':
    # Parse commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=str, default='0')
    args = parser.parse_args()

    with open('config/training_config.json') as f:
        training_config = json.load(f)

    # Set up Random Generator
    rng = np.random.default_rng(training_config['random_seed'])
    np.random.seed(training_config['random_seed'])

    # training loop config
    num_runs = training_config['num_runs']
    num_episodes  = training_config['num_episodes']
    window_size = training_config['averaging_window']
    horizon = training_config['horizon']
    budget = training_config['budget']

    # context, item feature config
    context_dim = training_config['context_dim']
    feature_dim = training_config['feature_dim']
    context_dist = training_config['context_distribution']

    os.environ["CUDA_VISIBLE_DEVICES"]= args.cuda
    print("running in {}".format('cuda' if torch.cuda.is_available() else 'cpu'))

    # output directory
    output_dir = 'results/' + time.strftime('%y%m%d-%H%M%S') + '/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    shutil.copy('config/training_config.json', os.path.join(output_dir, 'training_config.json'))

    #list of agents
    path_list = os.listdir('src/agents')
    agent_list = {}
    for name in path_list:
        if name=='.DS_Store':
            continue
        path = 'src/agents/'+name+'/'
        with open(os.path.join(path,'config.json')) as f:
            config = json.load(f)
        agent_list[name] = config
    num_agents = len(agent_list)

    run2item_features = draw_features()

    # memory recording rewards
    reward = np.zeros((num_runs, num_episodes, num_agents))

    for run in range(num_runs):

        CTR_param = rng.normal(0.0, 1.0, size=(context_dim, feature_dim))

        agents = instantiate_agents()
        auction = Auction(rng, agents, CTR_param, run2item_features[run], context_dim, context_dist, horizon, budget)

        t = 0
        for i in tqdm(range(num_episodes), desc=f'run {run}'):
            s, _ = auction.reset()
            done = truncated = False
            start = t
            episode_reward = np.zeros((num_agents))
            episode_win = np.zeros((num_agents))

            while not np.all(done):
                
                actions = []
                for j,agent in enumerate(agents):
                    bidding = agent.bid(s[j])
                    bidding = min(bidding,s[j][-2])
                    actions.append(bidding)

                s_, r, done, info = auction.step(actions)
                for agent in agents:
                    agent.newdata(s[j], actions[j], r[j], s_[j],
                                        done[j], info['win'][j], info['outcome'][j])
                    agent.update()

                t += 1
                episode_reward += r
                s = s_
            
            for agent in agents:
                agent.update()
            reward[run,i] = episode_reward
    
    reward = average(reward, window_size)
    cumulative_reward = np.cumsum(reward, axis=1)

    reward_df = numpy2df(reward, agents, 'Reward')
    reward_df.to_csv(output_dir + '/reward.csv', index=False)
    plot_measure(reward_df, 'Reward', window_size, output_dir)

    cumulative_reward_df = numpy2df(cumulative_reward, agents, 'Cumulative Reward')
    plot_measure(cumulative_reward_df, 'Cumulative Reward', window_size, output_dir)