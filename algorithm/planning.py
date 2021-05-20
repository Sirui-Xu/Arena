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
        self.gap = 0.1

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
            return random.randint(0, self.n_action - 1)

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

class PlanningPacV0:
    def __init__(self, env):
        self.n_action = len(env.getActionSet())
        self.name = env.name
        self.env = env
        assert self.name[:3] == "Pac"
        assert self.name[-4:] != "Maze" or self.name[-2:] != "1d"
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
        self.actions_name = ["left", "right", "up", "down", "noop"]

    def init_map(self):
        env_state = self.env.getEnvState()
        times = env_state["state"]["global"]["times"]
        ticks = env_state["state"]["global"]["ticks"]
        rest_ticks = int(times - ticks + 0.5)
        print(rest_ticks)
        assert env_state["state"]["local"][0]["type"] == "player"
        player_pos = env_state["state"]["local"][0]["position"]
        delta = env_state["state"]["local"][0]["speed"]
        node_map = np.zeros((len(env_state["state"]["local"]), len(env_state["state"]["local"])))
        values = []
        for i in range(len(env_state["state"]["local"])):
            values.append(env_state["state"]["local"][i]["type_index"][1])
            for j in range(i + 1, len(env_state["state"]["local"])):
                pos_i = env_state["state"]["local"][i]["position"]
                pos_j = env_state["state"]["local"][j]["position"]
                vel_i = env_state["state"]["local"][i]["velocity"]
                vel_j = env_state["state"]["local"][j]["velocity"]
                if i > 0:
                    pos_i = [pos_i[0] + vel_i[0], pos_i[1] + vel_i[1]]
                pos_j = [pos_j[0] + vel_j[0], pos_j[1] + vel_j[1]]
                node_map[i, j] = node_map[j, i] = abs(pos_j[0] - pos_i[0]) // delta + abs(pos_j[1] - pos_i[1]) // delta
        values[0] = 0
        self.rest_ticks, self.node_map, self.values = rest_ticks, node_map, values
        self.num_nodes = len(env_state["state"]["local"])
        self.env_state = env_state
    
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
        if len(path) > 1:
            node = path[1]
        else:
            node = self.values.index(max(self.values))
        pos_target = self.env_state["state"]["local"][node]["position"]
        vel_target = self.env_state["state"]["local"][node]["velocity"]
        pos_target = [pos_target[0] + vel_target[0], pos_target[1] + vel_target[1]]
        pos_source = self.env_state["state"]["local"][0]["position"]
        min_dis = [pos_target[0] - pos_source[0], pos_target[1] - pos_source[1]]
        projection = [d[0] * min_dis[0] + d[1] * min_dis[1] for d in self.directions]
        names = [self.actions_name[i] for i in range(len(projection)) if projection[i] > 0]
        projection = [projection[i] for i in range(len(projection)) if projection[i] > 0]
        assert len(names) <= 2
        if projection[0] / sum(projection) >= 0.5:
            return self.env.getActionIndex(names[0])
        else:
            return self.env.getActionIndex(names[1])

class PlanningPacV1:
    def __init__(self, env):
        self.n_action = len(env.getActionSet())
        self.name = env.name
        self.env = env
        assert self.name[:3] == "Pac"
        assert self.name[-4:] != "Maze" or self.name[-2:] != "1d"
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
        self.actions_name = ["left", "right", "up", "down", "noop"]

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
                sign_1 = ((pos_j[0] - pos_i[0]) * vel_j[0]) / abs(((pos_j[0] - pos_i[0]) * vel_j[0]))
                sign_2 = ((pos_j[1] - pos_i[1]) * vel_j[1]) / abs(((pos_j[1] - pos_i[1]) * vel_j[1]))
                if (delta - sign_1 * vel_j[0]) <= 0 or (delta - sign_2 * vel_j[1]) <= 0:
                    node_map[i, j] = 1e5
                else:
                    node_map[i, j] = node_map[j, i] = abs(pos_j[0] - pos_i[0]) // (delta - sign_1 * vel_j[0]) + abs(pos_j[1] - pos_i[1]) // (delta - sign_2 * vel_j[1])
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