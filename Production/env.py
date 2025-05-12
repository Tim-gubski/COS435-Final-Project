import cityflow
import json

import torch
import numpy as np

class FlowEnv:
    '''
    Citations.
    Built on CityFlow API: https://cityflow.readthedocs.io/en/latest/start.html#simulation
    Roadnet parse format from documentation: https://cityflow.readthedocs.io/en/latest/roadnet.html
    '''
    def __init__(self, config_path: str, roadnet_path: str, MAX_STEPS: int, save_replay: bool=False):
        self.roadnet = json.load(open(roadnet_path))
        self.eng = cityflow.Engine(config_path, thread_num=1)
        self.eng.set_save_replay(save_replay)

        self.MAX_STEPS = MAX_STEPS
        self.curr_step = 0
        
        self.intersections, self.action_dims, self.action_history = self.initialize_intersections()

    def initialize_intersections(self):
        intersections_list= [intersection for intersection in self.roadnet['intersections'] if 'trafficLight' in intersection and len(intersection['roadLinks']) > 0]
        intersections_list = [intersection for intersection in intersections_list if len(intersection['trafficLight']) > 0]

        intersections = {}
        for intersection in intersections_list:
            intersections[intersection['id']] = intersection
        
        action_dims = {}
        for id in intersections.keys():
            action_dims[id] = len(intersections[id]['trafficLight']['lightphases'])

        HIST_LENGTH = 5 
        action_history = {id: [0] * HIST_LENGTH for id in intersections.keys()}
        return intersections, action_dims, action_history

    def reset(self):
        self.eng.reset()
        self.curr_step = 0
        observations = self.get_observations()
        self.observation_dims = self.get_observation_dims(observations)
        return observations

    def get_observations(self, LANE_NORM: int = 20, GLOBAL_NORM: int = 100):
        waiting_vehicles = self.get_waiting_vehicle_count()
        all_vehicles = self.get_vehicle_count()

        observations = {}
        for id in self.intersections.keys():
            state = []

            lanes_incoming = []
            for road in self.roadnet['roads']:
                if road['endIntersection'] == id:
                    for i in range(len(road["lanes"])):
                        lane_id = f"{road['id']}_{i}"
                        lanes_incoming.append(lane_id)

            for lane in lanes_incoming:
                num_lane_waiting = waiting_vehicles.get(lane, 0) / LANE_NORM
                num_lane_vehicles = all_vehicles.get(lane, 0) / LANE_NORM
                state.append(num_lane_waiting)
                state.append(num_lane_vehicles)

            num_total_waiting = sum(waiting_vehicles.values()) / GLOBAL_NORM 
            num_total_vehicles = sum(all_vehicles.values()) / GLOBAL_NORM 
            state.append(num_total_waiting)
            state.append(num_total_vehicles)
                        
            observations[id] = state
        return observations

    def get_observation_dims(self, observations):
        dims = {}
        for id in self.intersections.keys():
            dims[id] = len(observations[id])
        return dims

    def step(self, actions):
        for id in self.intersections.keys():
            action = 0
            if id in actions.keys():
                action = actions[id]
            self.set_traffic_light_phase(id, action)

        self.eng.next_step()
        self.curr_step+=1

        next_observations = self.get_observations()
        rewards = self.get_rewards()
        return next_observations, rewards, (self.curr_step > self.MAX_STEPS)
    
    def get_rewards(self):
        waiting_vehicles = self.get_waiting_vehicle_count()
        all_vehicles = self.get_vehicle_count()

        rewards = {}
        for id in self.intersections.keys():
            lanes_incoming = []
            for road in self.roadnet['roads']:
                if road['endIntersection'] == id:
                    for i in range(len(road["lanes"])):
                        lane_id = f"{road['id']}_{i}"
                        lanes_incoming.append(lane_id)

            num_vehicles = 0
            num_waiting_vehicles = 0
            for lane in lanes_incoming:
                num_lane_vehicles = all_vehicles.get(lane, 0)
                num_lane_waiting = waiting_vehicles.get(lane, 0)
                num_vehicles += num_lane_vehicles
                num_waiting_vehicles += num_lane_waiting

            rewards[id] = -(num_waiting_vehicles + 0.25 * num_vehicles)
        return rewards
    
    def can_act(self, id):
        # return True
        return sum(self.action_history[id]) == 0
        
      
    def get_vehicle_count(self):
        return self.eng.get_lane_vehicle_count()

    def get_waiting_vehicle_count(self):
        return self.eng.get_lane_waiting_vehicle_count()

    def set_traffic_light_phase(self, id: int, new_phase: int, SAFETY = True):
        if SAFETY:
            action_history = self.action_history[id]
            if (action_history[-1] != new_phase) and sum(action_history) > 0:
                self.eng.set_tl_phase(id, 0)
                self.action_history[id].pop(0)
                self.action_history[id].append(0)
                return False 

        self.eng.set_tl_phase(id, new_phase)
        self.action_history[id].pop(0)
        self.action_history[id].append(new_phase)
        return True