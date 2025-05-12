import ray
from ray.rllib.algorithms.ppo import PPOConfig
from multi_signal_env import CityFlowMulti

ray.init()

env_cfg = {
    "cfg_file": "examples/customconfig.json",
    "roadnet":  "examples/customroadnet.json",
    "delta_t":  5,
    "max_steps": 3600
}

algo = PPOConfig().environment(...)
algo.restore("chkpt")

env = CityFlowMulti(**env_cfg)
obs, _ = env.reset()
done = False
while not done:
    act = {aid: algo.compute_single_action(o, policy_id=aid)
           for aid, o in obs.items()}
    obs, _, term, _, _ = env.step(act)
    done = term["__all__"]
