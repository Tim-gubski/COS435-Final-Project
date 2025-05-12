import cityflow
import numpy as np
import json
import os
from typing import Dict, Tuple, List, Any, Optional


class TrafficEnvironment:
    '''
    Old traffic environment. 
    Built upon CityFlow API: https://cityflow.readthedocs.io/en/latest/start.html#simulation
    '''
    def __init__(self, 
                 config_path: str,
                 roadnet_path: str,
                 yellow_time: int = 3,
                 max_steps: int = 1000,
                 normalize_state: bool = True,
                 reward_type: str = "waiting_time",
                 observation_type: str = "lane_vehicle_count"):

        self.config_path = config_path
        self.roadnet_path = roadnet_path
        self.yellow_time = yellow_time
        self.max_steps = max_steps
        self.normalize_state = normalize_state
        self.reward_type = reward_type
        self.observation_type = observation_type
        
        self.eng = cityflow.Engine(config_path, thread_num=1)
        
        self.roadnet = json.load(open(roadnet_path))
        
        self.intersections = [i for i in self.roadnet["intersections"] 
                              if "trafficLight" in i]
        self.intersection_ids = [i["id"] for i in self.intersections]
        
        self._init_intersections()
        self._init_lanes()
        
        self.current_step = 0
        self.current_phases = {i_id: 0 for i_id in self.intersection_ids}
        self.yellow_phase_count = {i_id: 0 for i_id in self.intersection_ids}
        self.is_yellow = {i_id: False for i_id in self.intersection_ids}
        
        self.previous_metrics = self._get_traffic_metrics()
    
    def _init_intersections(self):
        self.phases_per_intersection = {}
        self.action_spaces = {}
        self.phase_to_action = {}
        self.action_to_phase = {}
        
        for intersection in self.intersections:
            i_id = intersection["id"]
            phases = intersection["trafficLight"]["lightphases"]
            
            self.phases_per_intersection[i_id] = len(phases)
            
            self.action_spaces[i_id] = len(phases)
            
            self.phase_to_action[i_id] = {j: j for j in range(len(phases))}
            self.action_to_phase[i_id] = {j: j for j in range(len(phases))}
    
    def _init_lanes(self):
        self.incoming_lanes = {i_id: [] for i_id in self.intersection_ids}
        self.outgoing_lanes = {i_id: [] for i_id in self.intersection_ids}
        
        for road in self.roadnet["roads"]:
            start_intersection = road["startIntersection"]
            end_intersection = road["endIntersection"]
            
            if start_intersection in self.intersection_ids:
                for lane_idx in range(len(road["lanes"])):
                    lane_id = f"{road['id']}_{lane_idx}"
                    self.outgoing_lanes[start_intersection].append(lane_id)
            
            if end_intersection in self.intersection_ids:
                for lane_idx in range(len(road["lanes"])):
                    lane_id = f"{road['id']}_{lane_idx}"
                    self.incoming_lanes[end_intersection].append(lane_id)
        
        self.state_size = 0
        for i_id in self.intersection_ids:
            self.state_size += len(self.incoming_lanes[i_id]) + 1
    
    def reset(self) -> np.ndarray:
        self.eng = cityflow.Engine(self.config_path, thread_num=1)
        
        self.current_step = 0
        self.current_phases = {i_id: 0 for i_id in self.intersection_ids}
        self.yellow_phase_count = {i_id: 0 for i_id in self.intersection_ids}
        self.is_yellow = {i_id: False for i_id in self.intersection_ids}
        
        self.previous_metrics = self._get_traffic_metrics()
        
        return self._get_state()
    
    def step(self, actions: Dict[str, int]) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        for i_id, action in actions.items():
            if self.is_yellow[i_id]:
                self.yellow_phase_count[i_id] += 1
                if self.yellow_phase_count[i_id] >= self.yellow_time:
                    self.eng.set_tl_phase(i_id, self.action_to_phase[i_id][action])
                    self.current_phases[i_id] = action
                    self.is_yellow[i_id] = False
                    self.yellow_phase_count[i_id] = 0
            else:
                if action != self.current_phases[i_id]:
                    yellow_phase_id = 0 
                    self.eng.set_tl_phase(i_id, yellow_phase_id)
                    self.is_yellow[i_id] = True
                    self.yellow_phase_count[i_id] = 1
        
        self.eng.next_step()
        self.current_step += 1
        
        next_state = self._get_state()
        
        reward = self._compute_reward()
        
        done = self.current_step >= self.max_steps
        
        info = self._get_info()
        
        return next_state, reward, done, info
    
    def _get_state(self) -> np.ndarray:
        state = []
        
        if self.observation_type == "lane_vehicle_count":
            vehicle_counts = self.eng.get_lane_vehicle_count()
        else: 
            vehicle_counts = self.eng.get_lane_waiting_vehicle_count()
        
        for i_id in self.intersection_ids:
            for lane_id in self.incoming_lanes[i_id]:
                count = vehicle_counts.get(lane_id, 0)
                if self.normalize_state:
                    count = count / 30.0
                state.append(count)
            
            phase = self.current_phases[i_id]
            if self.normalize_state:
                phase = phase / float(self.phases_per_intersection[i_id])
            state.append(phase)
        
        return np.array(state, dtype=np.float32)
    
    def _get_traffic_metrics(self) -> Dict[str, Any]:
        metrics = {}
        
        lane_vehicle_count = self.eng.get_lane_vehicle_count()
        lane_waiting_count = self.eng.get_lane_waiting_vehicle_count()
        
        metrics["total_vehicles"] = sum(lane_vehicle_count.values())
        
        metrics["total_waiting"] = sum(lane_waiting_count.values())
        
        try:
            vehicle_delay = self.eng.get_vehicle_delay()
            metrics["average_delay"] = np.mean(list(vehicle_delay.values())) if vehicle_delay else 0
        except:
            metrics["average_delay"] = 0
        
        metrics["throughput"] = self.eng.get_vehicle_count()
        
        return metrics
    
    def _compute_reward(self) -> float:
        current_metrics = self._get_traffic_metrics()
        
        if self.reward_type == "waiting_time":
            reward = -current_metrics["total_waiting"]
        
        elif self.reward_type == "queue_length":
            prev_queue = self.previous_metrics["total_waiting"]
            curr_queue = current_metrics["total_waiting"]
            reward = prev_queue - curr_queue
        
        elif self.reward_type == "delay":
            reward = -current_metrics["average_delay"]
        
        elif self.reward_type == "throughput":
            prev_throughput = self.previous_metrics["throughput"]
            curr_throughput = current_metrics["throughput"]
            reward = curr_throughput - prev_throughput
        
        else:
            reward = -current_metrics["total_waiting"]
        
        self.previous_metrics = current_metrics
        
        return reward
    
    def _get_info(self) -> Dict[str, Any]:
        metrics = self._get_traffic_metrics()
        
        metrics["step"] = self.current_step
        
        return metrics
    
    def get_intersection_phases(self, intersection_id: str) -> int:
        return self.phases_per_intersection[intersection_id]
    
    def get_action_space(self) -> Dict[str, int]:
        return self.action_spaces
    
    def get_state_size(self) -> int:
        return self.state_size
    
    def render(self, mode: str = "text") -> None:
        if mode == "text":
            # Print current step and traffic metrics
            metrics = self._get_traffic_metrics()
            print(f"Step: {self.current_step}")
            print(f"Total vehicles: {metrics['total_vehicles']}")
            print(f"Waiting vehicles: {metrics['total_waiting']}")
            print(f"Phases: {self.current_phases}")
            print("-" * 50)
    
    def close(self) -> None:
        pass


if __name__ == "__main__":
    config_path = "./examples/config.json"
    roadnet_path = "./examples/roadnet.json"
    
    env = TrafficEnvironment(config_path, roadnet_path, max_steps=100)
    
    state = env.reset()
    print(f"Initial state shape: {state.shape}")
    
    for _ in range(10):
        actions = {i_id: np.random.randint(env.action_spaces[i_id]) 
                  for i_id in env.intersection_ids}
        
        next_state, reward, done, info = env.step(actions)
        
        env.render()
        
        print(f"Reward: {reward}")
        
        if done:
            break
    
    print("Environment test complete!")