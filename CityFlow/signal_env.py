import gymnasium as gym, numpy as np, cityflow, json, pathlib

class SignalEnv(gym.Env):
    """
    Single-intersection wrapper – identical to previous example.
    """
    def __init__(self, cfg_file, roadnet, inter_id,
                 delta_t=5, max_steps=3600):
        super().__init__()
        self.cfg, self.inter_id = cfg_file, inter_id
        self.delta_t, self.max_steps = delta_t, max_steps

        net = json.load(open(roadnet))
        inter = next(i for i in net["intersections"]
                     if i["id"] == inter_id)
        self.phases = len(inter["trafficLight"]["lightphases"])

        in_lanes = []
        for rl in inter["roadLinks"]:
            start = next(r for r in net["roads"]
                         if r["id"] == rl["startRoad"])
            in_lanes += [f"{start['id']}_{k}"
                         for k in range(len(start["lanes"]))]
        self.in_lanes = sorted(set(in_lanes))

        hi = np.full(len(self.in_lanes), 1000, dtype=np.float32)
        self.observation_space = gym.spaces.Box(0, hi, shape=hi.shape,
                                                dtype=np.float32)
        self.action_space      = gym.spaces.Discrete(self.phases)
        self.eng = None
        self.time = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.eng: del self.eng
        self.eng = cityflow.Engine(self.cfg, thread_num=1)
        self.time = 0
        return self._obs(), {}

    def step(self, action):
        self.eng.set_tl_phase(self.inter_id, int(action))
        for _ in range(self.delta_t):
            self.eng.next_step()
            self.time += 1
        obs = self._obs()
        rew = -obs.sum() 
        term = self.time >= self.max_steps
        return obs, rew, term, False, {}

    def _obs(self):
        wait = self.eng.get_lane_waiting_vehicle_count()
        return np.array([wait.get(l, 0) for l in self.in_lanes],
                        dtype=np.float32)

    def close(self):
        if self.eng: del self.eng
