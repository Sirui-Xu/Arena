def get_env_kwargs_dict(setting):
    if(setting=='legacy'):
        ret_kwargs_dict={
            'width':64,
            'height':64,
            'object_size':8,
            'obstacle_size':12,
            'num_coins':[1,5],
            'num_enemies':0,
            'num_bombs':0,
            'num_projectiles':3,
            'num_obstacles':0,
            'agent_speed':2,
            'enemy_speed':2,
            'projectile_speed':2,
            'explosion_max_step':100,
            'explosion_radius':4,
            'max_step':200}
        return ret_kwargs_dict



    ret_kwargs_dict={
        'width': 128,
        'height': 128,
        'object_size': 8,
        'obstacle_size': 16,
        'num_coins': 1,
        'num_enemies': 0,
        'num_bombs': 0,
        'num_projectiles': 0,
        'num_obstacles': 0,
        'agent_speed': 2,
        'enemy_speed': 2,  # Since there is no enemy, the speed does not matter.
        'projectile_speed': 8,
        'explosion_max_step': 100,
        'explosion_radius': 32,
        'reward_decay': 1.0,
        'max_step': 200}

    if '0' in setting:
        # no enemy, no obstacle
        ret_kwargs_dict['num_obstacles'] = 0
        ret_kwargs_dict['num_enemies'] = 0

    if '1' in setting:
        # have obstacle, no enemy
        ret_kwargs_dict['num_obstacles'] = [0, 10]
        ret_kwargs_dict['num_enemies'] = 0

    if '2' in setting:
        # have obstacle and enemy
        ret_kwargs_dict['num_obstacles'] = [0, 10]
        ret_kwargs_dict['num_enemies'] = [0, 5]

    if 'A' in setting:
        # enemy wont move, no bombs/projectiles
        ret_kwargs_dict['enemy_speed'] = 0
        ret_kwargs_dict['num_bombs'] = 0
        ret_kwargs_dict['num_projectiles'] = 0

    if 'B' in setting:
        # enemy will move, no bombs/projectiles
        ret_kwargs_dict['enemy_speed'] = 2
        ret_kwargs_dict['num_bombs'] = 0
        ret_kwargs_dict['num_projectiles'] = 0

    if 'C' in setting:
        # enemy will move, with bombs/projectiles
        ret_kwargs_dict['enemy_speed'] = 2
        ret_kwargs_dict['num_bombs'] = 3
        ret_kwargs_dict['num_projectiles'] = 3

    if 'X' in setting:
        # multiple coins
        ret_kwargs_dict['num_coins'] = [1, 5]
        ret_kwargs_dict['max_step'] = 200

    if 'Y' in setting:
        # multiple coins
        ret_kwargs_dict['num_coins'] = [1, 10]
        ret_kwargs_dict['max_step'] = 350

    return ret_kwargs_dict
