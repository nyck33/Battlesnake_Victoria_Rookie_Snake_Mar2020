import json
import os
import random
import numpy as np
from math import ceil
import bottle
from bottle import HTTPResponse
# import time
from timeit import default_timer as timer

from grid_data_maker import *
from util_fns import *

# my_moves
delta = [[-1, 0],  # go up
         [0, -1],  # go left
         [1, 0],  # go down
         [0, 1]]  # go right

delta_name = ['up', 'left', 'down', 'right']

cost = 1

# vals for smaller heads, equal or big, all bodies and next heads
small_head_val = 1
my_head_val = 3
same_head_val = 2
big_head_val = 5
body_val = 4
my_body_val = 7
next_bighead_val = 9
next_samehead_val = 6
next_smhead_val = 8

next_heads = [next_smhead_val, next_samehead_val, next_bighead_val]
curr_bodies = [small_head_val, my_head_val, same_head_val, big_head_val, body_val, my_body_val]
next_ok_heads = [next_smhead_val, next_samehead_val]

def make_heuristic_map(goal, snakes_grid):
    '''
    small_head_val = 1
    my_head_val=3
    same_head_val=2
    big_head_val = 5
    body_val = 4
    my_body_val = 7
    next_bighead_val = 9
    next_samehead_val = 6
    next_smhead_val = 8
    '''
    real_heads = [same_head_val, big_head_val]
    next_heads = [next_bighead_val, next_samehead_val]
    goal_y = goal[0]
    goal_x = goal[1]
    heuristic_map = np.zeros(snakes_grid.shape, dtype=np.int)
    for i in range(heuristic_map.shape[0]):
        for j in range(heuristic_map.shape[1]):
            dy = np.abs(i - goal_y)
            dx = np.abs(j - goal_x)
            heuristic_map[i, j] = dy + dx

    return heuristic_map


def fill_food_arr(food, my_head_y, my_head_x):
    # list in order of nearest to furthest food tuples (dist, y,x)
    food_arr = []
    for z in range(len(food)):
        food_dist = heuristic([my_head_y, my_head_x],
                              [food[z]['y'], food[z]['x']])
        food_arr.append([food_dist, food[z]['y'], food[z]['x']])

    food_array = sorted(food_arr, key=lambda x: x[0])
    # #print(f'\n\nfood arr {food_arr}\n\n')
    return food_array


def mark_next_heads(head_y, head_x, snakes_grid, next_head_val):
    '''
    delta = [[-1, 0],  # go up
             [0, -1],  # go left
             [1, 0],  # go down
             [0, 1]]
    '''
    new_grid = np.copy(snakes_grid)
    for i in range(len(delta)):
        next_head_y = head_y + delta[i][0]
        next_head_x = head_x + delta[i][1]
        # if in bounds and space is free, fill with 9
        if check_in_bounds(next_head_y, next_head_x, snakes_grid):
            if new_grid[next_head_y, next_head_x] == 0 or \
                    new_grid[next_head_y, next_head_x] in next_heads:
                new_grid[next_head_y, next_head_x] += next_head_val

    return new_grid


def fill_snakes_grid(snakes, width, height, my_body_len, my_id):
    '''
    small_head_val = 1
    same_head_val=2
    my_head_val = 3
    big_head_val = 5
    body_val = 4
    my_body_val = 7
    next_bighead_val = 9
    next_samehead_val = 6
    next_smhead_val = 8
    '''
    # my_moves
    delta = [[-1, 0],  # go up
             [0, -1],  # go left
             [1, 0],  # go down
             [0, 1]]  # go right

    snake_heads = []
    snake_tails = []
    # second grid for checking open path to tail
    snakes_grid = np.zeros((width, height), dtype=np.int)
    solo_grid = np.zeros(snakes_grid.shape, dtype=np.int)

    for j in range(len(snakes)):
        curr_snake = snakes[j]
        if curr_snake['id'] == my_id:
            my_snake = True
        else:
            my_snake = False
        # fill grid
        for k in range(len(curr_snake['body'])):
            # heads of opp snakes
            if k == 0:
                head_y = curr_snake['body'][k]['y']
                head_x = curr_snake['body'][k]['x']
                # if smaller
                if len(curr_snake['body']) < my_body_len and not my_snake:
                    snakes_grid[head_y, head_x] = small_head_val
                    # append to heads list
                    snake_heads.append([small_head_val, head_y, head_x])
                    # mark smaller next heads as 8
                    snakes_grid = mark_next_heads(head_y, head_x,
                                                  snakes_grid, next_smhead_val)
                # if it's the heads of bigger or equal snakes
                elif len(curr_snake['body']) > my_body_len and not my_snake:
                    snakes_grid[head_y, head_x] = big_head_val
                    # append to heads list
                    snake_heads.append([big_head_val, head_y, head_x])
                    # mark bigger or equal next heads as 9
                    snakes_grid = mark_next_heads(head_y,
                                                  head_x, snakes_grid, next_bighead_val)
                # todo: equal size
                elif len(curr_snake['body']) == my_body_len and not my_snake:
                    snakes_grid[head_y, head_x] = same_head_val
                    # todo: append to heads list or not?
                    snake_heads.append([same_head_val, head_y, head_x])
                    # mark bigger or equal next heads as 9
                    snakes_grid = mark_next_heads(head_y,
                                                  head_x, snakes_grid,
                                                  next_samehead_val)
                # fill solo grid for crash check
                elif len(curr_snake['body']) == my_body_len and my_snake:
                    solo_grid[head_y, head_x] = my_head_val
                    snakes_grid[head_y, head_x] = my_head_val
            # all snakes body and my head and body except tail
            elif 0 < k < (len(curr_snake['body']) - 1):
                body_y = curr_snake['body'][k]['y']
                body_x = curr_snake['body'][k]['x']
                #
                if not my_snake:
                    snakes_grid[body_y, body_x] = body_val
                # fill solo grid
                elif my_snake:
                    snakes_grid[body_y, body_x] = my_body_val
                    solo_grid[body_y, body_x] = body_val
            # tails
            elif k == (len(curr_snake['body']) - 1):
                body_y = curr_snake['body'][k]['y']
                body_x = curr_snake['body'][k]['x']
                solo_grid[body_y, body_x] = my_body_val
                # all tails attached here so careful in find path to tail
                snake_tails.append([body_y, body_x])
                if curr_snake['health'] == 100:
                    snakes_grid[body_y, body_x] = body_val


    return snakes_grid, solo_grid, snake_heads, snake_tails


def check_dist_to_snakes(snake_heads, head_y, head_x):
    snake_dists = []
    for i in range(len(snake_heads)):
        snakehead = snake_heads[i]
        snake_type = snakehead[0]
        snake_y, snake_x = snakehead[1], snakehead[2]
        dist = heuristic([head_y, head_x], [snake_y, snake_x])
        snake_dists.append([dist, snake_type, snakehead[0], snakehead[1]])
    snake_arr = sorted(snake_dists, key=lambda x: x[0])

    return snake_arr


def find_free_spaces(snakes_grid, head_y, head_x):
    free_spaces = np.argwhere(snakes_grid == 0)
    free_spaces_arr = []
    for i in range(free_spaces.shape[0]):
        curr_free = free_spaces[i, :].tolist()
        dist_to_free = heuristic([head_y, head_x], curr_free)
        free_spaces_arr.append([dist_to_free, curr_free[0], curr_free[1]])

    free_arr = sorted(free_spaces_arr, key=lambda x: x[0])
    return free_arr

