import json, math, sys, argparse, pygame
pygame.init()

ap = argparse.ArgumentParser()
ap.add_argument("bg", nargs="?", help="background image")
args = ap.parse_args()

W, H = 1000, 700
WHITE, RED, BLUE = (255,255,255), (255,64,64), (0,128,255)
NODE_R, ROAD_W, SNAP, DEL = 6, 6, 20, 20
scr  = pygame.display.set_mode((W, H))
pygame.display.set_caption("draw=Left Click  delete=Right Click  save=S")
font = pygame.font.SysFont(None, 20)

bg_surf = None; bg_scale, bg_off = 1.0, [0,0]
if args.bg:
    bg_surf = pygame.image.load(args.bg).convert_alpha()
    bw,bh = bg_surf.get_size()
    bg_scale=min(W/bw, H/bh); bg_off=[(W - bw*bg_scale)/2, (H - bh*bg_scale)/2]

w2s=lambda p:(p[0]*bg_scale + bg_off[0], H-(p[1]*bg_scale + bg_off[1]))
s2w=lambda p:((p[0] - bg_off[0])/bg_scale, (H-p[1] - bg_off[1])/bg_scale)

nodes=[]
roads=[] 
first=None

snap=lambda x,y: next((i for i,(px,py) in enumerate(nodes)
                       if math.hypot(px-x,py-y)<=SNAP), None)
def dist_pt_seg(px, py, ax, ay, bx, by):
    vx, vy, wx, wy = bx - ax, by - ay, px - ax, py - ay
    c1 = vx * wx + vy * wy
    if c1 <= 0: return math.hypot(px - ax, py - ay)
    c2 = vx * vx + vy * vy
    if c2 <= c1: return math.hypot(px - bx, py - by)
    t = c1/c2
    projx, projy = ax + t*vx, ay + t*vy
    return math.hypot(px - projx, py - projy)

road_exists = lambda a, b:any(s == a and e == b for s, e, _ in roads)

heading = lambda p, q: math.degrees(math.atan2(q[1] - p[1], q[0] - p[0])) % 360

def classify(a, b):
    d = (b - a) % 360
    if d < 30 or d > 330: return ("go_straight", 0)
    return ("turn_left", 2) if d > 180 else ("turn_right", 1)

lerp=lambda p, q, t:(p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1]))

def add_pair(s,e):
    if not road_exists(s,e): 
        roads.append((s, e, f"r_{s}_{e}_{len(roads)}"))
    if not road_exists(e, s): 
        roads.append((e, s, f"r_{e}_{s}_{len(roads)}_rev"))

def delete_pair(screen_pos):
    sx, sy = screen_pos
    for s, e, _ in roads[:]:
        ax, ay = w2s(nodes[s]); bx, by = w2s(nodes[e])
        if dist_pt_seg(sx, sy, ax, ay, bx, by) <= DEL:
            roads[:] = [r for r in roads if not ((r[0] == s and r[1] == e) or (r[0] == e and r[1] == s))]
            return

flipY = lambda y:y

def save():
    inter = []
    for idx, (x, y) in enumerate(nodes):
        con = [rid for s, e, rid in roads if s == idx or e == idx]
        inter.append({
            "id": str(idx),
            "point": {"x": x, "y": flipY(y)},
            "width": 10,
            "roads": con,
            "virtual": len(con) == 1,
            "roadLinks": []
        })

    rb   = {rid: (s, e) for s, e, rid in roads}
    pmap = {rid: [nodes[s], nodes[e]] for s, e, rid in roads}

    for I in inter:
        nid = int(I["id"])
        cx, cy = nodes[nid]
        inc = [rid for rid in I["roads"] if rb[rid][1] == nid]
        out = [rid for rid in I["roads"] if rb[rid][0] == nid]

        for rin in inc:
            pin = pmap[rin]
            hin = heading(*pin)
            p_in = lerp(*pin, 0.7)
            for rout in out:
                pout = pmap[rout]
                hout = heading(*pout)
                rtype, code = classify(hin, hout)

                if code == 1: code = 2
                elif code == 2: code = 1
                p_out = lerp(*pout, 0.3)

                I["roadLinks"].append({
                    "type": rtype,
                    "startRoad": rin,
                    "endRoad": rout,
                    "direction": code,
                    "laneLinks": [{
                        "startLaneIndex": 0,
                        "endLaneIndex": 0,
                        "points": [
                            {"x": p_in[0], "y": flipY(p_in[1])},
                            {"x": cx,       "y": flipY(cy)},
                            {"x": p_out[0], "y": flipY(p_out[1])}
                        ]
                    }]
                })

        if I["virtual"]:
            continue

        # quadrant 0=E,1=S,2=W,3=N
        quad_of = lambda rl: int(((heading(*pmap[rl['startRoad']]) + 45) % 360) // 90)

        EW_sr, EW_left, NS_sr, NS_left, rights = [], [], [], [], []
        for idx, rl in enumerate(I["roadLinks"]):
            q = quad_of(rl)
            if rl["type"] == "turn_right":
                rights.append(idx)
            if q in (0, 2): # East / West 
                if rl["type"] in ("go_straight", "turn_right"):
                    EW_sr.append(idx)
                elif rl["type"] == "turn_left":
                    EW_left.append(idx)
            else: # North / South 
                if rl["type"] in ("go_straight", "turn_right"):
                    NS_sr.append(idx)
                elif rl["type"] == "turn_left":
                    NS_left.append(idx)

        phases = []
        def add_phase(ids, t):
            if ids:
                phases.append({"time": t, "availableRoadLinks": ids})

        OFF_TIME = 5
        def add_off():
            if rights: 
                phases.append({"time": OFF_TIME, "availableRoadLinks": rights})

        add_phase(EW_sr, 30)
        add_off()
        add_phase(EW_left, 15)
        add_off()
        add_phase(NS_sr, 30)
        add_off()
        add_phase(NS_left, 15)
        add_off()

        I["trafficLight"] = {"lightphases": phases}

    roads_cf = [{
        "id": rid,
        "startIntersection": str(s),
        "endIntersection":   str(e),
        "lanes": [{"maxSpeed": 10.0, "width": 3.2}],
        "points": [
            {"x": nodes[s][0], "y": flipY(nodes[s][1])},
            {"x": nodes[e][0], "y": flipY(nodes[e][1])}
        ]
    } for s, e, rid in roads]

    json.dump({"intersections": inter, "roads": roads_cf},
              open("roadnet.json", "w"), indent = 2)
    print("roadnet.json saved")

clock = pygame.time.Clock()
while True:
    scr.fill(WHITE)
    if bg_surf:
        bw,bh = bg_surf.get_size()
        scr.blit(pygame.transform.smoothscale(bg_surf,(int(bw*bg_scale),int(bh*bg_scale))),bg_off)

    for s,e,_ in roads: 
        pygame.draw.line(scr, BLUE, w2s(nodes[s]), w2s(nodes[e]), ROAD_W)

    for x,y in nodes: 
        pygame.draw.circle(scr, RED, w2s((x,y)), NODE_R)

    scr.blit(font.render("LMB draw | RMB del | S save | +/- zoom | arrows pan", True, (0,0,0)), (10,10))
    pygame.display.flip(); clock.tick(60)

    for ev in pygame.event.get():
        if ev.type == pygame.QUIT: pygame.quit(); sys.exit()
        if ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_s: 
                save()
            if ev.key in (pygame.K_EQUALS,pygame.K_PLUS): 
                bg_scale*=1.1
            if ev.key == pygame.K_MINUS: 
                bg_scale/=1.1
            if ev.key == pygame.K_LEFT: 
                bg_off[0]+=20
            if ev.key == pygame.K_RIGHT: 
                bg_off[0]-=20
            if ev.key == pygame.K_UP:   
                bg_off[1]-=20
            if ev.key == pygame.K_DOWN: 
                bg_off[1]+=20

        if ev.type==pygame.MOUSEBUTTONDOWN:
            wx,wy = s2w(ev.pos)
            if ev.button == 1: # left click 
                idx = snap(wx,wy)
                if idx is None: 
                  idx = len(nodes)
                  nodes.append((wx,wy))
                if first is None: 
                    first=idx
                elif idx != first: 
                  add_pair(first,idx); first=None
            if ev.button == 3: # right click
                delete_pair(ev.pos)