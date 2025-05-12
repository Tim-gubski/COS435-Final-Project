from agents import FlowEnv
from agents import RandomAgent, CycleAgent, RulesAgent, DQNAgent

# choose configuration
test_intersection = "1"
config_path = f"Intersection{test_intersection}/config.json"
roadnet_path = f"Intersection{test_intersection}/roadnet.json"

# ENVIRONMENT
MAX_STEPS = 1000
env = FlowEnv(config_path, roadnet_path, MAX_STEPS, save_replay=True)
action_dims = env.get_action_dims()
state = env.reset()

print("State dim: ", len(state), "| Action_dims", action_dims)
print("================================================")


# AGENT SETUP

# agent = RandomAgent(state_dim=len(state), action_dims=action_dims)
# agent = CycleAgent(state_dim=len(state), action_dims=action_dims)
# agent = RulesAgent(state_dim=len(state), action_dims=action_dims)
agent_list = []
for dim in action_dims: # For every intersection
    agent_list.append(DQNAgent(len(state), dim))

NUM_EPISODES = 100
for ep in range(NUM_EPISODES):
    state = env.reset()
    done = False

    for step in range(MAX_STEPS):
        actions = [agent_list[i].act(state) for i in range(len(agent_list))]

        next_state, reward, done = env.step(actions)

        if isinstance(agent_list[0], DQNAgent):
            for i, agent in enumerate(agent_list):
                agent.remember(state, actions[i], reward, next_state, done)
                agent.update()

        if done:
            break
        state = next_state
    for agent in agent_list:
        agent.update_epsilon(ep, NUM_EPISODES)

    print(f"Episode {ep+1}/{NUM_EPISODES} | Waiting: {sum(env.get_lane_waiting_vehicle_count().values())}")