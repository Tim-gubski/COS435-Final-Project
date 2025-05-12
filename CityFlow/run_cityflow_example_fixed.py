import cityflow
import json
import numpy as np
import matplotlib.pyplot as plt
import random

# Initialize simulation
eng = cityflow.Engine("./examples/customconfig.json", thread_num=1)
roadnet = json.load(open("./examples/customroadnet.json"))
steps = 6000
waiting = []

last_change = {}
current_phase = {}
min_phase_time = 10 
max_phase_time = 100

state = 0

for intersection in roadnet["intersections"]:
    if "trafficLight" in intersection:
        intersection_id = intersection["id"]
        current_phase[intersection_id] = 0
        last_change[intersection_id] = 0


previous_waiting = 0

for t in range(steps):
    eng.next_step()
    
    if t % 10 == 0:
        veh_ids = eng.get_vehicles()
        print(f"Step {t:3d}: {len(veh_ids)} vehicles")
        
        lane_waiting = eng.get_lane_waiting_vehicle_count()
        waiting_car_count = sum(lane_waiting.values())
        waiting.append(waiting_car_count)
        print(f"Step {t:3d}: {waiting_car_count} waiting cars")

        for intersection in roadnet["intersections"]:
            if "trafficLight" in intersection:
                intersection_id = intersection["id"]
                time_in_phase = t - last_change[intersection_id]
                
                if ((time_in_phase >= min_phase_time and waiting_car_count > previous_waiting) or 
                    time_in_phase >= max_phase_time):
                    
                    num_phases = len(intersection["trafficLight"]["lightphases"])
                    
                    if waiting_car_count > previous_waiting * 1.5 and num_phases > 2:
                        # Try a random phase that's not the current one
                        candidates = list(range(num_phases))
                        candidates.remove(current_phase[intersection_id])
                        next_phase = random.choice(candidates)
                    else:
                        next_phase = (current_phase[intersection_id] + 1) % num_phases
                    
                    eng.set_tl_phase(intersection_id, next_phase)
                    current_phase[intersection_id] = next_phase
                    last_change[intersection_id] = t
                    print(f"Changed intersection {intersection_id} to phase {next_phase}")
        
        previous_waiting = waiting_car_count

# Save replay and plot results
eng.set_save_replay(True)
print("Simulation finished!")

np.save("./data/waiting_cars_simple_cycle.npy", waiting)
plt.plot(waiting)
plt.xlabel('Time (s)')
plt.ylabel('Number of Waiting Cars')
plt.title('Waiting Cars Over Time (Simple Cycling Controller)')
plt.grid()
plt.show()
