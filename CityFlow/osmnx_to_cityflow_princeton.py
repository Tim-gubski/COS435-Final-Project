import osmnx as ox
import json
import uuid
from shapely.geometry import LineString
import math

def get_princeton_roadnet():
    place_name = "Princeton, New Jersey, USA"
    G = ox.graph_from_place(place_name, network_type='drive', simplify=True)
    G = ox.project_graph(G)
    
    nodes = []
    roads = []
    
    node_id_map = {node: str(uuid.uuid4()) for node in G.nodes}
    
    node_roads = {node_id: {"in": [], "out": []} for node_id in node_id_map.values()}
    
    road_data = {}
    
    for u, v, key, data in G.edges(keys=True, data=True):
        # Calculate road length
        length = data.get('length', 100.0)
        
        highway_type = data.get('highway', 'residential')
        lanes = 2 if highway_type in ['primary', 'secondary', 'motorway'] else 1
        
        if 'lanes' in data:
            lanes_data = data['lanes']
            if isinstance(lanes_data, list):
                for lane in lanes_data:
                    try:
                        lanes = int(lane)
                        break
                    except (ValueError, TypeError):
                        continue
            else:
                try:
                    lanes = int(lanes_data)
                except (ValueError, TypeError):
                    pass
        
        if 'geometry' in data:
            line = data['geometry']
        else:
            line = LineString([(G.nodes[u]['x'], G.nodes[u]['y']), 
                             (G.nodes[v]['x'], G.nodes[v]['y'])])
        
        points = [{"x": x, "y": y} for x, y in line.coords]
        
        road_id = str(uuid.uuid4())
        
        road_dict = {
            "id": road_id,
            "startIntersection": node_id_map[u],
            "endIntersection": node_id_map[v],
            "points": points,
            "lanes": [
                {
                    "width": 3.5,
                    "maxSpeed": 11.176
                } for _ in range(lanes)
            ]
        }
        roads.append(road_dict)
        
        node_roads[node_id_map[u]]["out"].append(road_id)
        node_roads[node_id_map[v]]["in"].append(road_id)
        
        road_data[road_id] = {
            "startIntersection": node_id_map[u],
            "endIntersection": node_id_map[v],
            "points": points,
            "lanes": lanes
        }
    
    for node, data in G.nodes(data=True):
        node_id = node_id_map[node]
        incoming_roads = node_roads[node_id]["in"]
        outgoing_roads = node_roads[node_id]["out"]
        
        road_links = []
        for in_road in incoming_roads:
            for out_road in outgoing_roads:
                if in_road == out_road: 
                    continue
                in_points = road_data[in_road]["points"]
                out_points = road_data[out_road]["points"]
                
                in_vector = (in_points[-1]["x"] - in_points[-2]["x"], in_points[-1]["y"] - in_points[-2]["y"])
                out_vector = (out_points[1]["x"] - out_points[0]["x"], out_points[1]["y"] - out_points[0]["y"])
                angle = math.degrees(math.atan2(out_vector[1], out_vector[0]) - math.atan2(in_vector[1], in_vector[0]))
                angle = (angle + 360) % 360 
                
                if 45 <= angle < 135:
                    movement = "right"
                elif 225 <= angle < 315:
                    movement = "left"
                else:
                    movement = "straight"
                
                in_lanes = road_data[in_road]["lanes"]
                out_lanes = road_data[out_road]["lanes"]
                lane_links = [
                    {"startLaneIndex": i, "endLaneIndex": i % out_lanes}
                    for i in range(min(in_lanes, out_lanes))
                ]
                
                road_link = {
                    "type": movement,
                    "startRoad": in_road,
                    "endRoad": out_road,
                    "direction": movement,
                    "laneLinks": lane_links
                }
                road_links.append(road_link)
        
        traffic_light = {
            "roadLinkIndices": list(range(len(road_links))),
            "lightphases": [
                {
                    "time": 30, 
                    "availableRoadLinks": list(range(len(road_links))) 
                }
            ]
        } if not (len(G[node]) < 3) else {}
        
        node_dict = {
            "id": node_id,
            "point": {
                "x": data['x'],
                "y": data['y']
            },
            "virtual": len(G[node]) < 3,
            "roads": incoming_roads + outgoing_roads,
            "width": 10.0,
            "roadLinks": road_links,
            "trafficLight": traffic_light
        }
        nodes.append(node_dict)
    
    roadnet = {
        "intersections": nodes,
        "roads": roads
    }
    
    return roadnet

def save_roadnet(roadnet, filename="examples/princeton_roadnet.json"):
    with open(filename, 'w') as f:
        json.dump(roadnet, f, indent=2)

def main():
    roadnet = get_princeton_roadnet()
    save_roadnet(roadnet)
    print(f"Road network saved to examples/princeton_roadnet.json")

if __name__ == "__main__":
    main()