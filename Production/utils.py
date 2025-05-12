import torch
import torch.nn as nn

import random
from collections import deque
import numpy as np
import matplotlib.pyplot as plt

class MLPNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, mlp_dim):
        super(MLPNetwork, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, output_dim),
        )
    
    def forward(self, x):
        return self.mlp(x)
    
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state)
        action = torch.LongTensor([action])
        reward = torch.FloatTensor([reward])
        next_state = torch.FloatTensor(next_state)
        done = torch.FloatTensor([1 if done else 0])
        
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def sample_vectorized(self, batch_size):
        batch = self.sample(batch_size)

        state_list, action_list, reward_list, next_state_list, done_list = zip(*batch)
        state_batch = torch.stack(state_list)
        action_batch = torch.stack(action_list)
        reward_batch = torch.stack(reward_list)
        next_state_batch = torch.stack(next_state_list)
        done_batch = torch.stack(done_list)


        return state_batch, action_batch, reward_batch, next_state_batch, done_batch 
    
    
    def __len__(self):
        return len(self.buffer)    

def evaluate_policy(env, agents, test_intersection, agent_type):
    N_ROLLOUTS = 10
    total_waiting_accum = 0
    total_waiting = [0] * env.MAX_STEPS

    print("Evaluating: ")
    for _ in range(N_ROLLOUTS):
        observations = env.reset()

        for step in range(env.MAX_STEPS):
            can_update = {agent_id: env.can_act(agent_id) for agent_id in agents.keys()}
            actions = {agent_id: agents[agent_id].act(observations[agent_id]) for agent_id in agents.keys()}
            next_observations, rewards, done = env.step(actions)
            if done:
                break
            observations = next_observations

            vehicles_waiting = sum(env.get_waiting_vehicle_count().values())
            
            total_waiting[step] += vehicles_waiting
    
    total_waiting = np.array(total_waiting) / N_ROLLOUTS
    print(f'Avg cars waiting: {total_waiting[-1]:.2f}')  
    np.save(f"Intersection{test_intersection}/results_{agent_type}.npy", total_waiting)
    return 0

def plot_result(test_intersection):
    random = np.load(f"Intersection{test_intersection}/results_random.npy")
    cycle = np.load(f"Intersection{test_intersection}/results_cycle.npy")
    rules = np.load(f"Intersection{test_intersection}/results_rules.npy")
    dqn = np.load(f"Intersection{test_intersection}/results_dqn.npy")
    
    plt.figure(figsize=(8, 4))
    plt.plot(random, label=f"Random")
    plt.plot(cycle, label=f"Cycle")
    plt.plot(rules, label=f"Rules")
    plt.plot(dqn, label=f"DQN")

    plt.title("Total Waiting Cars over Time (Princeton Roadnet, Safety)")
    plt.xlabel("Timestep (1 step ≈ 1s)")
    plt.ylabel("Total Waiting Cars")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()

if __name__ == '__main__':
    test_intersection = "P-mini"
    plot_result(test_intersection)