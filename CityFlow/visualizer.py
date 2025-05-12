import json, math, matplotlib.pyplot as plt

with open("examples/customroadnet.json") as f:
    net = json.load(f)

ax = plt.gca()
LABEL_OFFSET = 4 
LABEL_SIZE   = 6


groups = {}
for r in net["roads"]:
    key = tuple(sorted((r["startIntersection"], r["endIntersection"])))
    groups.setdefault(key, []).append(r)

def midpoint(pts):
    """point halfway along the polyline `pts`"""
    mid_idx = len(pts)//2
    return pts[mid_idx]["x"], pts[mid_idx]["y"]

def unit_normal(p1, p2):
    """length-1 vector perpendicular to p1→p2 (left-hand normal)"""
    dx, dy = p2["x"]-p1["x"], p2["y"]-p1["y"]
    length = math.hypot(dx, dy) or 1
    return -dy/length, dx/length

for roads in groups.values():
    xs = [p["x"] for p in roads[0]["points"]]
    ys = [p["y"] for p in roads[0]["points"]]
    ax.plot(xs, ys, linewidth=0.8, color="black")

    mx, my = midpoint(roads[0]["points"])
    nx, ny = unit_normal(roads[0]["points"][0], roads[0]["points"][-1])

    offsets = [-1, 1] if len(roads) == 2 else [1]
    for road, side in zip(roads, offsets):
        label = road.get("name", road["id"])
        ax.text(mx + side*nx*LABEL_OFFSET,
                my + side*ny*LABEL_OFFSET,
                label, fontsize=LABEL_SIZE,
                ha="center", va="center")

ax.set_aspect("equal")
plt.show()
