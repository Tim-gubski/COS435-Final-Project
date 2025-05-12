import json
import random
import networkx as nx
from pathlib import Path

def load_roadnet(roadnet_file="examples/princeton_roadnet.json"):
    """Load the roadnet JSON file and return intersections and roads."""
    with open(roadnet_file, 'r') as f:
        roadnet = json.load(f)
    return roadnet['intersections'], roadnet['roads']

def build_graph(intersections, roads):
    """Build a directed graph from the roadnet for pathfinding."""
    G = nx.DiGraph()
    # Add intersections as nodes
    for intersection in intersections:
        G.add_node(intersection['id'])
    # Add roads as edges
    for road in roads:
        G.add_edge(road['startIntersection'], road['endIntersection'], road_id=road['id'])
    return G

def validate_route(roads, intersections, route):
    """Validate that a route is a connected sequence of road IDs and respects roadLinks."""
    road_dict = {road['id']: road for road in roads}
    intersection_dict = {i['id']: i for i in intersections}
    
    if not route or not all(road_id in road_dict for road_id in route):
        print(f"Validation failed: Route contains invalid road IDs: {route}")
        return False
    
    for i in range(len(route) - 1):
        current_road = road_dict[route[i]]
        next_road = road_dict[route[i + 1]]
        current_end = current_road['endIntersection']
        next_start = next_road['startIntersection']
        
        if current_end != next_start:
            print(f"Validation failed: Disconnected roads {current_road['id']} to {next_road['id']} "
                  f"(end: {current_end}, start: {next_start})")
            return False
        
        # Check if the transition is allowed in roadLinks
        intersection = intersection_dict.get(current_end)
        if not intersection:
            print(f"Validation failed: Intersection {current_end} not found")
            return False
        
        road_links = intersection.get('roadLinks', [])
        link_exists = any(
            link['startRoad'] == current_road['id'] and link['endRoad'] == next_road['id']
            for link in road_links
        )
        if not link_exists:
            print(f"Validation failed: No roadLink from {current_road['id']} to {next_road['id']} "
                  f"at intersection {current_end}")
            return False
    
    return True

def generate_random_route(G, start_intersection, end_intersection, roads, intersections, max_attempts=20):
    """Generate a random shortest path between start and end intersections."""
    for _ in range(max_attempts):
        try:
            # Find a shortest path
            path = nx.shortest_path(G, start_intersection, end_intersection)
            # Convert node path to road IDs
            route = []
            for i in range(len(path) - 1):
                edge_data = G.get_edge_data(path[i], path[i + 1])
                if not edge_data:
                    print(f"No edge between {path[i]} and {path[i+1]}")
                    return None
                route.append(edge_data['road_id'])
            # Validate the route
            if validate_route(roads, intersections, route):
                return route
            else:
                print(f"Generated invalid route: {route}")
                continue
        except nx.NetworkXNoPath:
            print(f"No path from {start_intersection} to {end_intersection}")
            return None
    print(f"Failed to find valid route after {max_attempts} attempts")
    return None

def generate_flow(intersections, roads, num_flows=50, max_interval=120):
    """Generate random vehicle flows."""
    G = build_graph(intersections, roads)
    non_virtual_intersections = [i['id'] for i in intersections if not i['virtual']]
    
    # Check graph connectivity
    components = list(nx.strongly_connected_components(G))
    print(f"Graph has {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Number of strongly connected components: {len(components)}")
    if len(components) > 1:
        largest_component = max(components, key=len)
        non_virtual_intersections = [i for i in non_virtual_intersections if i in largest_component]
        print(f"Restricted to largest component with {len(largest_component)} nodes")

    flows = []
    flow_count = 0
    while flow_count < num_flows:
        # Randomly select start and end intersections (non-virtual)
        start_intersection = random.choice(non_virtual_intersections)
        end_intersection = random.choice(non_virtual_intersections)
        while end_intersection == start_intersection:
            end_intersection = random.choice(non_virtual_intersections)
        
        # Generate a route
        route = generate_random_route(G, start_intersection, end_intersection, roads, intersections)
        if not route:
            print(f"Skipping flow {flow_count + 1}: No valid route from {start_intersection} to {end_intersection}")
            continue
        
        # Define vehicle types
        vehicle_types = [
            {"type": "car", "length": 5.0, "width": 2.0, "maxPosAcc": 2.0, "maxNegAcc": 4.5, "usualPosAcc": 1.0, "usualNegAcc": 2.0, "minGap": 2.5, "maxSpeed": 11.176, "headwayTime": 1.5},
            {"type": "truck", "length": 10.0, "width": 2.5, "maxPosAcc": 1.5, "maxNegAcc": 3.0, "usualPosAcc": 0.8, "usualNegAcc": 1.5, "minGap": 3.0, "maxSpeed": 8.94, "headwayTime": 2.0}
        ]
        
        # Generate flow
        flow = {
            "vehicle": random.choice(vehicle_types),
            "route": route,
            "interval": random.uniform(1, 5),  # 1-5 seconds between vehicles
            "startTime": 0,  # Start immediately
            "endTime": 120  # End at simulation duration
        }
        flows.append(flow)
        flow_count += 1
        print(f"Generated flow {flow_count}: route length {len(route)}, interval {flow['interval']:.2f}s")
    
    return flows

def save_flow(flows, filename="examples/princeton_flow.json"):
    """Save the flows to a JSON file."""
    with open(filename, 'w') as f:
        json.dump(flows, f, indent=2)

def main():
    # Check if roadnet file exists
    roadnet_file = "examples/princeton_roadnet.json"
    if not Path(roadnet_file).exists():
        print(f"Error: {roadnet_file} not found. Please generate the roadnet first.")
        return
    
    # Load roadnet
    intersections, roads = load_roadnet(roadnet_file)
    
    # Generate flows
    flows = generate_flow(intersections, roads, num_flows=50, max_interval=120)
    
    # Save to JSON file
    save_flow(flows)
    print(f"Flow file saved to examples/princeton_flow.json")

if __name__ == "__main__":
    main()