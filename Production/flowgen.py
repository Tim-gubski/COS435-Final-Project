import json
import random

'''Generates random flows'''

roadnet = json.load(open("./IntersectionP-mini/roadnet.json"))

roadnames = [roadnet['roads'][i]['id'] for i in range(len(roadnet['roads']))]
# print(roadnames)

flow = []

for i in range(4):
  flow.append(
    {
      "vehicle": {
        "length": 5.0,
        "width": 2.0,
        "maxPosAcc": 2.0,
        "maxNegAcc": 4.5,
        "usualPosAcc": 2.0,
        "usualNegAcc": 4.5,
        "minGap": 2.5,
        "maxSpeed": 30.0,
        "headwayTime": 1.5
      },
      "route": [random.choice(roadnames), random.choice(roadnames)],
      "interval": 20.0,
      "startTime": 0,
      "endTime": 3600
    }
  )
print(flow)
json.dump(flow, open("flow.json", "w"), indent=2)



