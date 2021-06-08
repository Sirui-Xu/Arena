import pygame
import random
import sys
import numpy as np

def circle2box(pos, radius):
    return (pos[0] - radius, pos[1] - radius, pos[0] + radius, pos[1] + radius)

def in_box(point, box):
    return point[0] <= box[2] and point[0] >= box[0] and point[1] <= box[3] and point[1] >= box[1] 

class PlanningCollectMaze:
    def __init__(self, env):
        self.n_action = len(env.getActionSet())
        self.name = env.name
        self.env = env
        assert self.name[:5] == "Water" or self.name[:8] == "Billiard"
        assert self.name[-4:] == "Maze"
        self.actions_name = ["left", "right", "up", "down", "noop"]
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]

    def init_map(self):
        self.state = self.env.getGameState()
        assert self.state["local"][0]['type'] == 'player' and self.state["local"][1]['type'] == 'creep'
        maze = self.state["global"]["maze"]
        self.map_shape = [self.state["global"]["norm_shape"][0], self.state["global"]["norm_shape"][1]]
        self.good_map = np.zeros(tuple(self.map_shape))
        self.bad_map = np.copy(maze)
        for k, info in enumerate(self.state["local"][1:]): 
            new_pos = [int(info["norm_position"][0] + info["norm_velocity"][0] + 0.5), int(info["norm_position"][1] + info["norm_velocity"][1] + 0.5)]
            if new_pos[0] == 0 or new_pos[0] == self.map_shape[0] - 1 or new_pos[1] == 0 or new_pos[1] == self.map_shape[1] - 1:
                new_pos = [int(info["norm_position"][0] + 0.5), int(info["norm_position"][1] + 0.5)]
            creep_type = info["_type"]
            if creep_type == "GOOD":
                self.good_map[new_pos[0], new_pos[1]] += 1
            elif creep_type == "BAD":
                self.bad_map[new_pos[0], new_pos[1]] += 1
        self.x, self.y = int(self.state["local"][0]["norm_position"][0] + 0.5), int(self.state["local"][0]["norm_position"][1] + 0.5)

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
                if search_map[x, y] != 0 or self.good_map[x, y] - self.bad_map[x, y] < 0:
                    continue
                # First search the path not close to the edge of the bad node
                flag = False
                for ddxy in dxys:
                    if ddxy[0] == -dxy[0] and ddxy[1] == -dxy[1]:
                        continue
                    if self.bad_map[x + ddxy[0], y + ddxy[1]] > 0:
                        flag = True
                        break
                if flag == True:
                    direction.append((x, y))
                else:
                    direction.insert(0, (x, y))

            for x, y in direction:
                search_map[x, y] = 1
                if self.good_map[x, y] > 0 and self.bad_map[x, y] == 0:
                    openlist.append((x, y, node))
                    find_out = True
                    end_point = openlist[-1]
                    break
                # If the bad node covers the good node, approach first
                elif self.good_map[x, y] > 0 and node[0] != self.x and node[1] != self.y:
                    openlist.append((x, y, node))
                    find_out = True
                    end_point = openlist[-1]
                    break
                if self.bad_map[x, y] == 0:
                    openlist.append((x, y, node))
            if find_out is True:
                break
            openlist.pop(0)

        if find_out is False:
            for dxy in dxys:
                x = self.x + dxy[0]
                y = self.y + dxy[1]
                if self.good_map[x, y] - self.bad_map[x, y] >= 0:
                    return self.env.getActionIndex(self.actions_name[self.directions.index(dxy)])
            return self.env.getActionIndex(self.actions_name[self.directions.index(dxy)])

        father = end_point[2]
        while True:
            if father[2] == None:
                dxy = (end_point[0] - father[0], end_point[1] - father[1])
                break
            end_point = father
            father = end_point[2]

        return self.env.getActionIndex(self.actions_name[self.directions.index(dxy)])

    def one_step(self):
        rewards = []
        env_state = self.env.getEnvState()
        for action in range(self.n_action):
            _, reward, game_over, _ = self.env.step(action)
            if game_over and reward < 0:
                rewards.append(-100)
            else:
                rewards.append(reward)
            self.env.loadEnvState(env_state)
            # print(self.env.getEnvState())
        actions = [i for i in range(self.n_action) if rewards[i] == max(rewards)]
        return actions

    def exe(self): 
        self.init_map()
        action = self.shortest_path()
        actions = self.one_step()
        if action in actions:
            return action
        else:        
            return random.choice(actions)


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
        assert self.fps >= 15
        self.env = env
        self.game = env.game
        self.srange = srange
        self.init()

    def get_real_speed(self, speed):
        return speed / 1000.0 * (1000.0 / self.fps)
    
    def cal_time(self, creep):
        speed_x, speed_y = self.get_real_speed(creep.direction.x*creep.speed), -self.get_real_speed(creep.direction.y*creep.speed)
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
        self.times.sort(key=lambda x:x[1][0])

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
            if t1 < 0:
                continue
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
                    if in_box([bullet_x, bullet_y], circle2box([creep_x, creep_y], creep.radius+self.env.game.BULLET_RADIUS)):
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
                    self.actions.append(['left','right','noop'][random.randrange(0, 3)])
                return True
        else:
            # print("only one nodes")
            self.actions.append(['left','right','noop'][random.randrange(0, 3)])
            return True

        return True

    def reset(self):
        self.times = [(creep, self.cal_time(creep)) for creep in self.env.game.creeps.sprites()]
        self.times.sort(key=lambda x:x[1][2])
        if self.target not in self.env.game.creeps.sprites():
            self.target = None
  
    def exe(self):
        self.reset()
        self.action()
        action_name = self.actions[0]
        return self.env.getActionIndex(action_name)
class PlanningArena:
    def __init__(self, env):
        self.n_action = len(env.getActionSet())
        self.name = env.name
        self.env = env
        self.name == "ARENA"
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
        self.actions_name = ["left", "right", "up", "down", "noop", "shoot", "fire"]


    def exe(self):
        env_state = self.env.getEnvState()
        self.map = np.zeros(tuple(env_state["state"]["global"]["shape"]))
        width, height = env_state["state"]["global"]["shape"][0], env_state["state"]["global"]["shape"][1]
        min_dis_reward_index = 0
        min_dis_reward = width + height
        threat = []
        agent_pos = [0, 0]
        agent_radius = 0
        enemy_radius = 0
        reward_radius = 0
        agent_speed = 0
        agent_vel = [0, 0]
        agent_box = [0, 0, 0, 0]
        projectile_directions = []
        projectile_speed = 0
        for i, info in enumerate(env_state["state"]["local"]):
            if info["type"] == "agent":
                agent_pos = info["position"]
                agent_radius = info["radius"]
                agent_speed = info["speed"]
                agent_box = info["box"]
                agent_vel = info["velocity"]
            if info["type"] == "reward":
                reward_pos = info["position"]
                reward_radius = info["radius"]                
                reward_box = info["box"]
                self.map[reward_box[0]:reward_box[2], reward_box[1]:reward_box[3]] = info["type_index"][0]
                dis = abs(reward_pos[0] - agent_pos[0]) + abs(reward_pos[1] - agent_pos[1])
                if dis < min_dis_reward:
                    min_dis_reward = dis
                    min_dis_reward_index = i   

            if info["type"] == "obstacle":       
                obstacle_box = info["box"]
                self.map[obstacle_box[0]:obstacle_box[2], obstacle_box[1]:obstacle_box[3]] = info["type_index"][0]

            if info["type"] == "blast":
                blast_box = info["box"]
                self.map[blast_box[0]:blast_box[2], blast_box[1]:blast_box[3]] = info["type_index"][0]


            if info["type"] == "enemy":
                enemy_pos = info["position"]
                enemy_radius = 1.2 * info["radius"]
                enemy_speed = info["speed"]
                enemy_pos_new = [enemy_pos[0] + info["velocity"][0], enemy_pos[1] + info["velocity"][1]]
                dis = abs(enemy_pos[0] - agent_pos[0]) + abs(enemy_pos[1] - agent_pos[1])
                if enemy_pos_new[0] < enemy_radius or enemy_pos_new[0] > width - enemy_radius or enemy_pos_new[1] < enemy_radius or enemy_pos_new[1] > height - enemy_radius:
                    enemy_speed = info["speed"]
                    for direction in [[0, 1], [1, 0], [0, -1], [-1, 0]]:
                        if direction[0] * info["velocity"][0] + direction[1] * info["velocity"][1] > 0:
                            continue
                        enemy_pos_new = [enemy_pos[0] + enemy_speed * direction[0], enemy_pos[1] + enemy_speed * direction[1]]
                        enemy_box_new = [int(enemy_pos_new[0] - enemy_radius - 1), int(enemy_pos_new[1] - enemy_radius - 1),
                                        int(enemy_pos_new[0] + enemy_radius + 1), int(enemy_pos_new[1] + enemy_radius + 1)]
                        self.map[enemy_box_new[0]:enemy_box_new[2], enemy_box_new[1]:enemy_box_new[3]] = info["type_index"][0]
                        new_dis = abs(enemy_pos_new[0] - agent_pos[0]) + abs(enemy_pos_new[1] - agent_pos[1])
                        new_time = (max(abs(enemy_pos_new[0] - agent_pos[0]) - agent_radius - enemy_radius, 0)
                                 + max(abs(enemy_pos_new[1] - agent_pos[1]) - agent_radius - enemy_radius, 0)) / agent_speed
                        if new_dis < dis and new_time <= 6:
                            threat.append([1 / 3, enemy_pos_new, direction])         
                else:        
                    enemy_box_new = [int(enemy_pos_new[0] - enemy_radius - 1), int(enemy_pos_new[1] - enemy_radius - 1),
                                    int(enemy_pos_new[0] + enemy_radius + 1), int(enemy_pos_new[1] + enemy_radius + 1)]
                    self.map[enemy_box_new[0]:enemy_box_new[2], enemy_box_new[1]:enemy_box_new[3]] = info["type_index"][0]
                    new_dis = abs(enemy_pos_new[0] - agent_pos[0]) + abs(enemy_pos_new[1] - agent_pos[1])
                    new_time = (max(abs(enemy_pos_new[0] - agent_pos[0]) - agent_radius - enemy_radius, 0)
                             + max(abs(enemy_pos_new[1] - agent_pos[1]) - agent_radius - enemy_radius, 0)) / agent_speed
                    # print(new_dis, dis, new_time)
                    if new_dis < dis and new_time <= 6:
                        threat.append([1, enemy_pos_new, info["velocity"]])


            if info["type"] == "bomb":
                bomb_pos = info["position"]
                bomb_life = info["type_index"][2]
                bomb_range = info["type_index"][3]
                if agent_speed * bomb_life <= bomb_range: 
                    bomb_box = [int(bomb_pos[0] - bomb_range + agent_speed * bomb_life - 1),
                                int(bomb_pos[1] - bomb_range + agent_speed * bomb_life - 1),
                                int(bomb_pos[0] + bomb_range - agent_speed * bomb_life + 1),
                                int(bomb_pos[1] + bomb_range - agent_speed * bomb_life + 1)]
                    self.map[bomb_box[0]:bomb_box[2], bomb_box[1]:bomb_box[3]] = info["type_index"][0]
            
            if info["type"] == "projectile":
                projectile_pos = info["position"]
                projectile_box = info["box"]
                projectile_speed = info["speed"]
                self.map[projectile_box[0]:projectile_box[2], projectile_box[1]:projectile_box[3]] = info["type_index"][0]
        
        self.agent_pos = agent_pos
        self.agent_radius = agent_radius
        self.enemy_radius = enemy_radius
        self.reward_radius = reward_radius
        self.agent_speed = agent_speed
        self.agent_vel = agent_vel
        self.agent_box = agent_box

        # print(threat, self.map)
        FLAG, move_action_index = self.shortest_path()
        if FLAG is False:
            heuristics = []
            choice = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
            random.shuffle(choice)
            for direction in choice:
                agent_pos_new = [agent_pos[0] + agent_speed * direction[0], agent_pos[1] + agent_speed * direction[1]]
                agent_box_new = [int(agent_pos_new[0] - agent_radius), int(agent_pos_new[1] - agent_radius),
                                int(agent_pos_new[0] + agent_radius), int(agent_pos_new[1] + agent_radius)]

                if np.sum(self.map[agent_box_new[0]:agent_box_new[2], agent_box_new[1]:agent_box_new[3]] > 2) > 0:
                    heuristics.append(1000)
                    continue
                
                heuristic = 0
                if np.sum(self.map[agent_box_new[0]:agent_box_new[2], agent_box_new[1]:agent_box_new[3]] == 2) > 0:
                    heuristic += 2
                
                reward_info = env_state["state"]["local"][min_dis_reward_index]
                heuristic = (max((abs(agent_pos_new[0] - reward_info["position"][0]) - agent_radius - reward_radius + 1), 0)
                        + max((abs(agent_pos_new[1] - reward_info["position"][1]) - agent_radius - reward_radius + 1), 0)) / agent_speed
                for enemy_info in threat:
                    c, enemy_pos_new, enemy_direction = enemy_info
                    if enemy_direction[0] == 0:
                        time = (2 * (max((abs(agent_pos_new[0] - enemy_pos_new[0]) - agent_radius - enemy_radius), 0)
                                + max((abs(agent_pos_new[1] - enemy_pos_new[1]) - agent_radius - enemy_radius), 0)) / agent_speed)
                    else:
                        time = ((max((abs(agent_pos_new[0] - enemy_pos_new[0]) - agent_radius - enemy_radius), 0)
                                + 2 * max((abs(agent_pos_new[1] - enemy_pos_new[1]) - agent_radius - enemy_radius), 0)) / agent_speed)
                    if time > 0:
                        heuristic += c / time 
                    else:
                        heuristic += 100
                heuristics.append(heuristic)
            
            move_action_index = self.directions.index(choice[heuristics.index(min(heuristics))])
        # print(heuristics, self.actions_name[move_action_index])
        if self.actions_name[move_action_index] == "noop":
            if random.random() > 0.5 + 1 / (env_state["state"]["global"]["projectiles_left"] + 2):
                return self.env.getActionIndex("shoot")
            else:
                return self.env.getActionIndex("noop")
        
        if agent_vel[0] * self.directions[move_action_index][0] + agent_vel[1] * self.directions[move_action_index][1] > 0:
            # print("judge")
            t = 1
            pos = [agent_pos[0], agent_pos[1]]
            vel = agent_vel
            while True:
                pos = [pos[0] + vel[0], pos[1] + vel[1]]
                if agent_vel[0] != 0:
                    dis = abs(pos[0] - env_state["state"]["local"][min_dis_reward_index]["position"][0])
                else:
                    dis = abs(pos[1] - env_state["state"]["local"][min_dis_reward_index]["position"][1])
                
                if pos[0] > width - agent_radius or pos[0] < agent_radius or pos[1] > height - agent_radius or pos[1] < agent_radius:
                    # print(t, "out of map")
                    return self.env.getActionIndex(self.actions_name[move_action_index])

                box = [int(pos[0] - agent_radius), int(pos[1] - agent_radius),
                       int(pos[0] + agent_radius), int(pos[1] + agent_radius)]
                if np.sum(self.map[box[0]:box[2], box[1]:box[3]] == 2) > 0:
                    # print(t, "meet obstacle")
                    if random.random() < 1 / t:
                        return self.env.getActionIndex("shoot")
                    else:
                        return self.env.getActionIndex(self.actions_name[move_action_index])
                
                if t < 6:
                    for enemy_info in threat:
                        c, enemy_pos_new, enemy_vel = enemy_info
                        # print(enemy_pos_new)
                        if abs(enemy_pos_new[0] - pos[0]) < agent_radius + enemy_radius and abs(enemy_pos_new[1] - pos[1]) < agent_radius + enemy_radius:
                            # print("meet enemy")
                            if t == 1 or random.random() < c / (t - 1):
                                return self.env.getActionIndex("shoot")  
                        enemy_pos_new = [enemy_pos_new[0] + enemy_vel[0], enemy_pos_new[1] + enemy_vel[1]]
                                                    

                if np.sum(self.map[box[0]:box[2], box[1]:box[3]] == 6) > 0:
                    # print(t, "have shoot")
                    return self.env.getActionIndex(self.actions_name[move_action_index])

                if dis < agent_radius + reward_radius:
                    # print(t, "meet reward")
                    return self.env.getActionIndex(self.actions_name[move_action_index])
                t += 1
                if t > 100:
                    break
        
        return self.env.getActionIndex(self.actions_name[move_action_index])

    def shortest_path(self):
        # bfs
        path = []
        actions = []
        dxys = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        random.shuffle(dxys)
        openlist = []
        self.x, self.y = int(self.agent_pos[0]), int(self.agent_pos[1])
        openlist.append((self.agent_pos[0], self.agent_pos[1], None))
        search_map = np.zeros_like(self.map)
        search_map[self.x, self.y] = 1
        find_out = False
        end_point = None
        while len(openlist) > 0:
            node = openlist[0]
            direction = []
            for dxy in dxys:
                x = node[0] + dxy[0] * self.agent_speed
                y = node[1] + dxy[1] * self.agent_speed
                box = [int(x - self.agent_radius), int(y - self.agent_radius), int(x + self.agent_radius), int(y + self.agent_radius)]
                if not in_box((x, y), (self.agent_radius, self.agent_radius, self.map.shape[0]-self.agent_radius-1, self.map.shape[1]-self.agent_radius-1)) or search_map[int(x), int(y)] != 0:
                    continue
                if np.sum(self.map[box[0]:box[2], box[1]:box[3]] > 1) > 0:
                    continue
                # First search the path not close to the edge of the bad node
                flag = False
                for ddxy in dxys:
                    if ddxy[0] == -dxy[0] and ddxy[1] == -dxy[1]:
                        continue
                    x_ = x + ddxy[0] * self.agent_speed
                    y_ = y + ddxy[1] * self.agent_speed
                    if not (in_box((x_, y_), (self.agent_radius, self.agent_radius, self.map.shape[0]-self.agent_radius-1, self.map.shape[1]-self.agent_radius-1))):
                        continue
                    box_ = [int(x_ - self.agent_radius), int(y_ - self.agent_radius), int(x_ + self.agent_radius), int(y_ + self.agent_radius)]
                    if np.sum(self.map[box_[0]:box_[2], box_[1]:box_[3]] > 1) > 0:
                        flag = True
                        break
                if flag == True:
                    direction.append((x, y))
                else:
                    direction.insert(0, (x, y))

            for x, y in direction:
                openlist.append((x, y, node))
                box = [int(x - self.agent_radius), int(y - self.agent_radius), int(x + self.agent_radius), int(y + self.agent_radius)]
                search_map[int(x), int(y)] = 1
                if np.sum(self.map[box[0]:box[2], box[1]:box[3]] == 1) > 0 and (np.sum(self.map[box[0]:box[2], box[1]:box[3]] > 2) == 0 or int(node[0]) != int(self.x) or int(node[1]) != int(self.y)):
                    find_out = True
                    end_point = openlist[-1]
                    break
            if find_out is True:
                break
            openlist.pop(0)

        if find_out is False:
            return False, None

        father = end_point[2]
        while True:
            if father[2] == None:
                dxy = [end_point[0] - father[0], end_point[1] - father[1]]
                if dxy[0] == 0:
                    dxy[1] /= abs(dxy[1])
                else:
                    dxy[0] /= abs(dxy[0])
                break
            end_point = father
            father = end_point[2]

        return True, self.directions.index(tuple(dxy))
