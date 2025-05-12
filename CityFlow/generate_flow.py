import json, random

# 1. load your network
with open("examples/customroadnet.json") as f:
    roads = [r["id"] for r in json.load(f)["roads"]]

flows = []
for _ in range(200):                    # e.g. 200 random routes
    start, end = random.sample(roads, 2)
    flows.append({
      "vehicle": {
        "length": 5.0, "width": 2.0,
        "maxPosAcc": 2.0, "maxNegAcc": 4.5,
        "usualPosAcc": 2.0, "usualNegAcc": 4.5,
        "minGap": 2.5, "maxSpeed": 30.0,
        "headwayTime": 1.5
      },
      "route": [ start, end ],          # only start & end
      "interval": 2.0,
      "startTime": 0,
      "endTime": 3600
    })

# 3. write it out
with open("customflow.json", "w") as f:
    json.dump(flows, f, indent=2)
