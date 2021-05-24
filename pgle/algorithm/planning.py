import pygame
import random
import sys
import numpy as np

def circle2box(pos, radius):
    return (pos[0] - radius, pos[1] - radius, pos[0] + radius, pos[1] + radius)

def in_box(point, box):
    return point[0] <= box[2] and point[0] >= box[0] and point[1] <= box[3] and point[1] >= box[1] 

class PlanningCollect:
    def __init__(self, env):
        self.n_action = len(env.getActionSet())
        self.name = env.name
        self.env = env
        assert self.name[:5] == "Water" or self.name[:8] == "Billiard"
        assert self.name[-2:] != "1d" and self.name[-4:] != "Maze"
        self.actions_name = ["left", "right", "up", "down", "noop"]
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
        self.gap = 0.2

    def init_map(self):
        self.state = self.env.getGameState()
        assert self.state["local"][0]['type'] == 'player'
        delta = self.state["local"][0]["speed"]
        player_x, player_y = self.state["local"][0]["position"][0], self.state["local"][0]["position"][1]
        rightside = int((self.env.game.width - self.env.game.AGENT_RADIUS - player_x) // delta)
        leftside = int((player_x - self.env.game.AGENT_RADIUS) // delta)
        downside = int((self.env.game.height - self.env.game.AGENT_RADIUS - player_y) // delta)
        upside = int((player_y - self.env.game.AGENT_RADIUS) // delta)
        self.map_shape = [leftside + rightside + 1, upside + downside + 1]
        self.good_map = np.zeros(((leftside + rightside + 1), (upside + downside + 1)))
        self.bad_map = np.zeros(((leftside + rightside + 1), (upside + downside + 1)))
        self.x = leftside
        self.y = upside
        for i in range(leftside + rightside + 1):
            for j in range(upside + downside + 1):
                x_pos = (i - leftside) * delta + player_x
                y_pos = (j - upside) * delta + player_y
                point = [x_pos, y_pos]
                for k, info in enumerate(self.state["local"][1:]): 
                    pos = [info["position"][0] + info["velocity"][0], info["position"][1] + info["velocity"][1]]
                    creep_type = info["_type"]

                    if creep_type == "GOOD":
                        if in_box(point, circle2box(pos, (1 - self.gap)*self.env.game.CREEP_RADII[0]+self.env.game.AGENT_RADIUS)):
                            self.good_map[i, j] += 1
                    elif creep_type == "BAD":
                        if in_box(point, circle2box(pos, (1 + self.gap)*self.env.game.CREEP_RADII[1]+self.env.game.AGENT_RADIUS)):
                            self.bad_map[i, j] += 1
        self.delta = delta
        self.player_x, self.player_y = player_x, player_y

    def shortest_path(self):
        # bfs
        path = []
        actions = []
        dxys = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        random.shuffle(dxys)
        openlist = []
        openlist.append((self.x, self.y, None))
        search_map = np.zeros_like(self.bad_map)
        search_map[self.x, self.y] = 1
        find_out = False
        end_point = None
        while len(openlist) > 0:
            node = openlist[0]
            direction = []
            for dxy in dxys:
                x = node[0] + dxy[0]
                y = node[1] + dxy[1]
                if not in_box((x, y), (0, 0, self.map_shape[0]-1, self.map_shape[1]-1)) or search_map[x, y] != 0 or self.bad_map[x, y] > 0:
                    continue
                # First search the path not close to the edge of the bad node
                flag = False
                for ddxy in dxys:
                    if ddxy[0] == -dxy[0] and ddxy[1] == -dxy[1]:
                        continue
                    if in_box((x + ddxy[0], y + ddxy[1]), (0, 0, self.map_shape[0]-1, self.map_shape[1]-1)) and self.bad_map[x + ddxy[0], y + ddxy[1]] > 0:
                        flag = True
                        break
                if flag == True:
                    direction.append((x, y))
                else:
                    direction.insert(0, (x, y))

            for x, y in direction:
                openlist.append((x, y, node))
                search_map[x, y] = 1
                if self.good_map[x, y] > 0 and self.bad_map[x, y] == 0:
                    find_out = True
                    end_point = openlist[-1]
                    break
                # If the bad node covers the good node, approach first
                elif self.good_map[x, y] > 0 and node[0] != self.x and node[1] != self.y:
                    find_out = True
                    end_point = openlist[-1]
                    break
            if find_out is True:
                break
            openlist.pop(0)

        if find_out is False:
            if len(openlist) < 2:
                return random.randint(0, self.n_action - 1)
            else:
                dxy = (openlist[1][0] - openlist[0][0], openlist[1][1] - openlist[0][1])
                return self.env.getActionIndex(self.actions_name[self.directions.index(dxy)])

        father = end_point[2]
        while True:
            if father[2] == None:
                dxy = (end_point[0] - father[0], end_point[1] - father[1])
                break
            end_point = father
            father = end_point[2]

        return self.env.getActionIndex(self.actions_name[self.directions.index(dxy)])

    def exe(self): 
        self.init_map()
        return self.shortest_path()


class PlanningPac:
    def __init__(self, env):
        self.n_action = len(env.getActionSet())
        self.name = env.name
        self.env = env
        assert self.name[:3] == "Pac"
        assert self.name[-4:] != "Maze" or self.name[-2:] != "1d"
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
        self.actions_name = ["left", "right", "up", "down", "noop"]

    def cal_time(self, x, y, vx, vy, v0):
        choice = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        for p1, p2 in choice:
            if (v0 - (p1 * vx + p2 * vy)) == 0:
                continue
            t = (p1 * x + p2 * y) / (v0 - (p1 * vx + p2 * vy))
            if t > 0 and p1 * (x / t + vx) >= 0 and p2 * (y / t + vy) >= 0:
                return t 
        return None

    def init_map(self):
        env_state = self.env.getEnvState()
        times = env_state["state"]["global"]["times"]
        ticks = env_state["state"]["global"]["ticks"]
        rest_ticks = times - ticks + 1
        assert env_state["state"]["local"][0]["type"] == "player"
        player_pos = env_state["state"]["local"][0]["position"]
        delta = env_state["state"]["local"][0]["speed"]
        node_map = np.zeros((len(env_state["state"]["local"]), len(env_state["state"]["local"])))
        values = []
        pos_source = env_state["state"]["local"][0]["position"]
        min_dis = [self.env.game.width + 1, self.env.game.height + 1]
        node = 1
        for i in range(len(env_state["state"]["local"])):
            values.append(env_state["state"]["local"][i]["type_index"][1])
            info = env_state["state"]["local"][i]
            if info['type'] == 'creep':
                creep_pos = info["position"]
                dis = [creep_pos[0] - pos_source[0], creep_pos[1] - pos_source[1]]
                if dis[0] * dis[0] + dis[1] * dis[1] < min_dis[0] * min_dis[0] + min_dis[1] * min_dis[1]:
                    min_dis = dis
                    node = i
            for j in range(i + 1, len(env_state["state"]["local"])):
                pos_i = env_state["state"]["local"][i]["position"]
                pos_j = env_state["state"]["local"][j]["position"]
                vel_i = env_state["state"]["local"][i]["velocity"]
                vel_j = env_state["state"]["local"][j]["velocity"]
                

                t = self.cal_time((pos_j[0] - pos_i[0]), (pos_j[1] - pos_i[1]), vel_j[0], vel_j[1], delta)
                if t != None:
                    node_map[i, j] = int(t) + 1
                else:
                    node_map[i, j] = 1e5

                t = self.cal_time((pos_i[0] - pos_j[0]), (pos_i[1] - pos_j[1]), vel_i[0], vel_i[1], delta)
                if t != None:
                    node_map[j, i] = int(t) + 1
                else:
                    node_map[j, i] = 1e5

        values[0] = 0
        self.rest_ticks, self.node_map, self.values = rest_ticks, node_map, values
        self.num_nodes = len(env_state["state"]["local"])
        self.env_state = env_state
        self.node = node
    
    def tsp(self, path, rest, value):
        flag = False
        for value in sorted(self.values, reverse=True):
            i = self.values.index(value)
            if i in path:
                continue
            if rest > self.node_map[path[-1], i]:
                self.tsp(path + [i], rest - self.node_map[path[-1], i], value+self.values[i])
                flag = True 
        if flag == False:
            self.paths.append(path)
            self.most_values.append(value)

    def exe(self): 
        self.init_map()
        self.paths = []
        self.most_values = []
        self.tsp([0], self.rest_ticks, 0)
        max_index = self.most_values.index(max(self.most_values))
        path = self.paths[max_index]
        pos_source = self.env_state["state"]["local"][0]["position"]
        if len(path) > 1:
            node = path[1]
        else:
            node = self.node
        pos_target = self.env_state["state"]["local"][node]["position"]
        vel_target = self.env_state["state"]["local"][node]["velocity"]
        pos_target = [pos_target[0] + vel_target[0], pos_target[1] + vel_target[1]]
        min_dis = [pos_target[0] - pos_source[0], pos_target[1] - pos_source[1]]
        projection = [d[0] * min_dis[0] + d[1] * min_dis[1] for d in self.directions]
        names = [self.actions_name[i] for i in range(len(projection)) if projection[i] > 0]
        projection = [projection[i] for i in range(len(projection)) if projection[i] > 0]
        assert len(names) <= 2
        if projection[0] / sum(projection) >= 0.5:
            return self.env.getActionIndex(names[0])
        else:
            return self.env.getActionIndex(names[1])

class PlanningShoot1d:
    def __init__(self, env, srange=2):
        assert env.name[:5] == "Shoot" and env.name[-2:] == "1d"
        self.directions = [(-1, 0), (1, 0), None, (0, 0)]
        self.actions_name = ["left", "right", "fire", "noop"]
        self.fps = env.game.fps
        self.env = env
        self.game = env.game
        self.srange = srange
        self.init()

    def get_real_speed(self, speed):
        return speed / 1000.0 * (1000.0 / self.fps)
    
    def cal_time(self, creep):
        speed_x, speed_y = self.get_real_speed(creep.direction.x*creep.speed), self.get_real_speed(creep.direction.y*creep.speed)
        speed_a = self.get_real_speed(self.game.AGENT_SPEED)
        speed_b = self.get_real_speed(self.game.BULLET_SPEED)
        dx = creep.pos.x - self.game.player.pos.x
        dy = -(creep.pos.y - self.game.player.pos.y)
        t_1 = (speed_x * dy + (speed_b - speed_y) * dx) / (speed_a*speed_b - speed_x*speed_b - speed_y*speed_a)
        t_2 = (speed_y * dx + (speed_a - speed_x) * dy) / (speed_a*speed_b - speed_x*speed_b - speed_y*speed_a)
        if t_1 < 0:
            t_1 = (speed_x * dy + (speed_b - speed_y) * dx) / (-speed_a*speed_b - speed_x*speed_b + speed_y*speed_a)
            t_2 = (speed_y * dx + (-speed_a - speed_x) * dy) / (-speed_a*speed_b - speed_x*speed_b + speed_y*speed_a)
        if t_2 < 0:
            return (t_1, -t_2, t_1 - t_2)
        else:
            return (t_1, t_2, t_1 + t_2)

    def init(self):
        # print('begin to initialize...')
        self.ready2hit = {}
        self.target = None
        self.times = [(creep, self.cal_time(creep)) for creep in self.env.game.creeps.sprites()]
        self.times.sort(key=lambda x:x[1][2])

    def action(self):
        self.actions = []
        self.path = []
        pop_item = []

        for creep in self.ready2hit.keys():
            if self.ready2hit[creep] <= 0:
                pop_item.append(creep)
            else:
                self.ready2hit[creep] -= 1
        self.ready2hit = {key:val for key, val in self.ready2hit.items() if key not in pop_item}

        find_out = False
        for i, (creep, (t1, t2, _)) in enumerate(self.times):
            if self.target == None:
                if creep in self.ready2hit.keys():
                    continue
                else:
                    self.target = creep
                    find_out = True
                    break

            if creep == self.target:
                find_out = True
                break
        # print('target = ', self.target, t1, t2)
        # print(self.ready2hit)
        # print(self.times)
        if find_out:
            find_out = False
            t1s = list(range(int(t1) - self.srange // 2 + 1, int(t1) - self.srange // 2 + self.srange + 1))
            t2s = list(range(int(t2) - self.srange // 2 + 1, int(t2) - self.srange // 2 + self.srange + 1))
            for t_1 in t1s:
                for t_2 in t2s[::-1]:
                    if t_1 < 0 or t_2 < 0:
                        continue
                    creep_x = creep.pos.x + self.get_real_speed(creep.direction.x*creep.speed) * (t_1 + t_2)
                    creep_y = creep.pos.y + self.get_real_speed(creep.direction.y*creep.speed) * (t_1 + t_2)
                    dx = creep_x - self.game.player.pos.x
                    if dx > 0:
                        bullet_x = self.game.player.pos.x + self.get_real_speed(self.game.AGENT_SPEED) * t_1
                    else:
                        bullet_x = self.game.player.pos.x - self.get_real_speed(self.game.AGENT_SPEED) * t_1
                    bullet_y = self.game.player.pos.y - self.get_real_speed(self.game.BULLET_SPEED) * t_2
                    if in_box([bullet_x, bullet_y], circle2box([creep_x, creep_y], creep.radius+self.env.game.BULLET_RADIUS-1)):
                        if t_1 == 0:
                            if not in_box([creep_x, creep_y], [0, 0, self.game.width*1.1, self.game.height*1.1]):
                                continue
                            else:
                                self.ready2hit[creep] = t_2
                                self.target = None
                        find_out = True
                        
                        if dx > 0:
                            self.actions.extend(['right' for _ in range(t_1)] + ['fire'])
                        else:
                            self.actions.extend(['left' for _ in range(t_1)] + ['fire'])
                        break
                if find_out:
                    return True
            if not find_out:
                # print("not find out: ", creep, t1, t2, dx)
                if t1s[0] > 0:
                    if dx > 0:
                        self.actions.extend(['right'])
                    else:
                        self.actions.extend(['left'])
                else:
                    self.actions.append(['left','right'][random.randrange(0, 2)])
                return True
        else:
            # print("only one nodes")
            self.actions.append(['left','right'][random.randrange(0, 2)])
            return True

        return True

    def reset(self):
        self.times = [(creep, self.cal_time(creep)) for creep in self.env.game.creeps.sprites()]
        self.times.sort(key=lambda x:x[1][2])
  
    def exe(self):
        self.reset()
        self.action()
        action_name = self.actions[0]
        return self.env.getActionIndex(action_name)


class PlanningShoot1dV1:
    def __init__(self, env, srange=2):
        assert env.name[:5] == "Shoot" and env.name[-2:] == "1d"
        self.directions = [(-1, 0), (1, 0), None, (0, 0)]
        self.actions_name = ["left", "right", "fire", "noop"]
        self.fps = env.game.fps
        self.env = env
        self.game = env.game
        self.srange = srange
        self.init()

    def get_real_speed(self, speed):
        return speed / 1000.0 * (1000.0 / self.fps)
    
    def cal_time(self, creep):
        speed_x, speed_y = self.get_real_speed(creep.direction.x*creep.speed), self.get_real_speed(creep.direction.y*creep.speed)
        speed_a = self.get_real_speed(self.game.AGENT_SPEED)
        speed_b = self.get_real_speed(self.game.BULLET_SPEED)
        dx = creep.pos.x - self.game.player.pos.x
        dy = -(creep.pos.y - self.game.player.pos.y)
        t_1 = (speed_x * dy + (speed_b - speed_y) * dx) / (speed_a*speed_b - speed_x*speed_b - speed_y*speed_a)
        t_2 = (speed_y * dx + (speed_a - speed_x) * dy) / (speed_a*speed_b - speed_x*speed_b - speed_y*speed_a)
        if t_1 < 0:
            t_1 = (speed_x * dy + (speed_b - speed_y) * dx) / (-speed_a*speed_b - speed_x*speed_b + speed_y*speed_a)
            t_2 = (speed_y * dx + (-speed_a - speed_x) * dy) / (-speed_a*speed_b - speed_x*speed_b + speed_y*speed_a)
        if t_2 < 0:
            return (t_1, -t_2, t_1 - t_2)
        else:
            return (t_1, t_2, t_1 + t_2)

    def init(self):
        # print('begin to initialize...')
        self.ready2hit = {}
        self.target = None
        self.times = [(creep, self.cal_time(creep)) for creep in self.env.game.creeps.sprites()]
        self.times.sort(key=lambda x:x[1][2])

    def action(self):
        self.actions = []
        self.path = []
        pop_item = []

        for creep in self.ready2hit.keys():
            if self.ready2hit[creep] <= 0:
                pop_item.append(creep)
            else:
                self.ready2hit[creep] -= 1
        self.ready2hit = {key:val for key, val in self.ready2hit.items() if key not in pop_item}

        find_out = False
        for i, (creep, (t1, t2, _)) in enumerate(self.times):
            if self.target == None:
                if creep in self.ready2hit.keys():
                    continue
                else:
                    self.target = creep
                    find_out = True
                    break

            if creep == self.target:
                find_out = True
                break
        # print('target = ', self.target, t1, t2)
        # print(self.ready2hit)
        # print(self.times)
        if find_out:
            find_out = False
            t1s = list(range(int(t1) - self.srange // 2 + 1, int(t1) - self.srange // 2 + self.srange + 1))
            t2s = list(range(int(t2) - self.srange // 2 + 1, int(t2) - self.srange // 2 + self.srange + 1))
            for t_1 in t1s:
                for t_2 in t2s[::-1]:
                    if t_1 < 0 or t_2 < 0:
                        continue
                    creep_x = creep.pos.x + self.get_real_speed(creep.direction.x*creep.speed) * (t_1 + t_2)
                    creep_y = creep.pos.y + self.get_real_speed(creep.direction.y*creep.speed) * (t_1 + t_2)
                    dx = creep_x - self.game.player.pos.x
                    if dx > 0:
                        bullet_x = self.game.player.pos.x + self.get_real_speed(self.game.AGENT_SPEED) * t_1
                    else:
                        bullet_x = self.game.player.pos.x - self.get_real_speed(self.game.AGENT_SPEED) * t_1
                    bullet_y = self.game.player.pos.y - self.get_real_speed(self.game.BULLET_SPEED) * t_2
                    if in_box([bullet_x, bullet_y], circle2box([creep_x, creep_y], creep.radius+self.env.game.BULLET_RADIUS-1)):
                        if t_1 == 0:
                            if not in_box([creep_x, creep_y], [0, 0, self.game.width*1.1, self.game.height*1.1]):
                                continue
                            else:
                                self.ready2hit[creep] = t_2
                                self.target = None
                        find_out = True
                        
                        if dx > 0:
                            self.actions.extend(['right' for _ in range(t_1)] + ['fire'])
                        else:
                            self.actions.extend(['left' for _ in range(t_1)] + ['fire'])
                        break
                if find_out:
                    return True
            if not find_out:
                # print("not find out: ", creep, t1, t2, dx)
                if t1s[0] > 0:
                    if dx > 0:
                        self.actions.extend(['right'])
                    else:
                        self.actions.extend(['left'])
                else:
                    self.actions.append(['left','right'][random.randrange(0, 2)])
                return True
        else:
            # print("only one nodes")
            self.actions.append(['left','right'][random.randrange(0, 2)])
            return True

        return True

    def reset(self):
        self.times = [(creep, self.cal_time(creep)) for creep in self.env.game.creeps.sprites()]
        self.times.sort(key=lambda x:x[1][2])
  
    def exe(self):
        self.reset()
        self.action()
        action_name = self.actions[0]
        return self.env.getActionIndex(action_name)
