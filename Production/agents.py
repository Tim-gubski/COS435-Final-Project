from env import FlowEnv

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import random
from collections import deque
import pickle

from utils import MLPNetwork, ReplayBuffer, evaluate_policy, plot_result

from typing import List
import argparse

# Multi-agent approach (each intersection has its own agent)

class Agent:
    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim
    
    def act(self):
        raise NotImplementedError

class RandomAgent(Agent):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__(state_dim, action_dim)
    
    def act(self, observation, eval=False):
        rand = np.random.randint(0, self.action_dim)
        # while rand == 3 or rand == 4:
        #     rand = np.random.randint(0, self.action_dim)
        return rand
    
class CycleAgent(Agent):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__(state_dim, action_dim)
        
        self.last_action = 0
        self.tick = 0
    
    def act(self, observation, eval=False):
        self.tick  += 1
        if self.tick == 20:
            action = (self.last_action + 1) % self.action_dim
            # if action == 3:
            #     action = 5
            self.last_action = action
            return action
        elif self.tick > 20 and self.tick < 25:
           return 0
        elif self.tick == 25:
           self.tick = 0
           return 0
        
        return self.last_action

class RulesAgent(Agent):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__(state_dim, action_dim)
        
        self.tick = 0
        self.last_waiting = 0
        self.last_vehicles = 0
        self.last_action = 0
        
    def act(self, observation, eval=False):
        if np.random.rand() < 0.5:
            rand = np.random.randint(0, self.action_dim)
            return rand

        self.tick  += 1
        if self.tick == 20:
            
            total_waiting = observation[-2]
            total_vehicles = observation[-1]
            
            action = 0
            
            if total_waiting > self.last_waiting:
                action = (self.last_action + 1) % self.action_dim
                # if action == 3:
                #     action = 5
                
            self.last_waiting = total_waiting
            self.last_vehicles = total_vehicles

            self.last_action = action
            return action
        elif self.tick > 20 and self.tick < 25:
           return 0
        elif self.tick == 25:
           self.tick = 0
           return 0
        
        return self.last_action

class DQNAgent(Agent):
    def __init__(self, state_dim: int, action_dim: int, MLP_DIM = 64, LR=0.001,
                 GAMMA=0.999, EPS=0.9, EPS_MIN=0.02, TAU=0.005, batch_size=32,
                 EXPLORE_FLAG=True):
        super().__init__(state_dim, action_dim)
        self.action_dim = action_dim
        self.GAMMA = GAMMA # discount factor
        self.EPS, self.EPS_MIN = EPS, EPS_MIN # epsilon decay
        self.TAU = TAU # soft update factor
        self.batch_size = batch_size # buffer sampling
        self.EXPLORE_FLAG = EXPLORE_FLAG # epsilon greedy exploration

        self.mlp = MLPNetwork(state_dim, action_dim, MLP_DIM)
        self.target_mlp = MLPNetwork(state_dim, action_dim, MLP_DIM) 
        self.update_target_network()
        
        self.optimizer = optim.Adam(self.mlp.parameters(), lr=LR)
        self.buffer = ReplayBuffer()
    
    def act(self, observation, eval=False):
        # random action
        if not eval and self.EXPLORE_FLAG and random.random() < self.EPS:
            return np.random.randint(0, self.action_dim)
        
        # MLP action
        observation = torch.FloatTensor(observation).unsqueeze(0)
        with torch.no_grad():
            q_vals = self.mlp(observation)
        return q_vals.argmax().item()
    
    def remember(self, obs, action, reward, next_obs, done):
        self.buffer.add(obs, action, reward, next_obs, done)

    def update(self):
        if len(self.buffer) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.buffer.sample_vectorized(self.batch_size)

        with torch.no_grad():
            next_q_vals = self.target_mlp(next_states).max(1, keepdim=True)[0]
            target_q = rewards + (1 - dones) * self.GAMMA * next_q_vals


        current_q = self.mlp(states).gather(1, actions)

        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.mlp.parameters(), max_norm=10.0)
        self.optimizer.step()
        

        for param, target_param in zip(self.mlp.parameters(), self.target_mlp.parameters()):
            target_param.data.copy_(self.TAU * param.data + (1.0 - self.TAU) * target_param.data)
    
    def update_target_network(self):
        self.target_mlp.load_state_dict(self.mlp.state_dict())

    def update_epsilon(self, episode, total_episodes):
        anneal_episode = total_episodes//2
        self.EPS = self.EPS_MIN if (episode > anneal_episode) else (self.EPS_MIN + (1-self.EPS_MIN)*(anneal_episode-episode)/anneal_episode)

if __name__ == '__main__':
    # Intersection Descriptions
    # Intersection1: 1 x 1
    # Intersection2: 1 x 2
    # Intersection3: 1 x 3
    # Intersection4: 2 x 2
    # IntersectionP: Princeton flow
    test_intersection = "P-mini"
    config_path = f"Intersection{test_intersection}/config.json"
    roadnet_path = f"Intersection{test_intersection}/roadnet.json"

    # ENVIRONMENT SETUP
    ENV_MAX_STEPS = 2000
    env = FlowEnv(config_path, roadnet_path, ENV_MAX_STEPS, save_replay=True)
    observations = env.reset()
    
    print("Intersections:", len(env.intersections.keys()))
    print("Observe dims:", env.observation_dims)
    print("Action dims:", env.action_dims)
    print("====================")

    # AGENT SETUP
    parser = argparse.ArgumentParser(description='Agent Selection')
    parser.add_argument('agent_type', type=str, choices=['random', 'cycle', 'rules', 'dqn'], default='random')
    args = parser.parse_args()

    agent_type = args.agent_type

    agents = {}
    for id in env.intersections.keys():
        obs_dim, action_dim = env.observation_dims[id], env.action_dims[id]
        if agent_type == 'random':
            agents[id] = RandomAgent(obs_dim, action_dim)
        elif agent_type == 'cycle':
            agents[id] = CycleAgent(obs_dim, action_dim)
        elif agent_type == 'rules':
            agents[id] = RulesAgent(obs_dim, action_dim)
        elif agent_type == 'dqn':
            agents[id] = DQNAgent(obs_dim, action_dim)

    if agent_type == 'dqn':
        NUM_EPISODES = 30
        ep_history = []
        for ep in range(NUM_EPISODES):
            observations = env.reset()
            total_waiting = 0
            total_waiting_arr = [0] * env.MAX_STEPS

            for step in range(env.MAX_STEPS):
                can_update = {agent_id: env.can_act(agent_id) for agent_id in agents.keys()}
                actions = {agent_id: agents[agent_id].act(observations[agent_id]) for agent_id in agents.keys() if can_update[agent_id]}
                next_observations, rewards, done = env.step(actions)

                for agent_id in actions.keys():
                    agents[agent_id].remember(observations[agent_id], actions[agent_id], rewards[agent_id], next_observations[agent_id], done)
                    agents[agent_id].update() 
                if done:
                    break
                observations = next_observations

                vehicles_waiting = sum(env.get_waiting_vehicle_count().values())
                total_waiting += vehicles_waiting
                # print(f"Step: {step+1} | Reward: {list(rewards.values())} | Avg Waiting Vehicles: {total_waiting/env.MAX_STEPS}")
                # for name, param in agents['0'].mlp.named_parameters():
                #     if param.requires_grad:
                #         print(f"{name}:\n{param.data}\n")
                total_waiting_arr[step] += vehicles_waiting


            if (ep+1)%10 == 0:
                with open(f"Intersection{test_intersection}/models.pkl", 'wb') as f:
                    pickle.dump(agents, f)
            total_waiting_arr = np.array(total_waiting_arr)
            np.save(f"Intersection{test_intersection}/results_{agent_type}.npy", total_waiting_arr)
            

            agents[agent_id].update_epsilon(ep, NUM_EPISODES) 
            print(f"Episode: {ep+1} | Reward: {list(rewards.values())} | Avg Waiting Vehicles: {total_waiting/env.MAX_STEPS}")
            ep_history.append(total_waiting/env.MAX_STEPS)
        
        np.save(f"Intersection{test_intersection}/results_{agent_type}_HISTORY.npy", ep_history)
    
    evaluate_policy(env, agents, test_intersection, agent_type)