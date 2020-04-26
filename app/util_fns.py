import json
import os
import random
import numpy as np
from math import ceil
import bottle
from bottle import HTTPResponse
# import time
from timeit import default_timer as timer

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

def check_in_bounds(new_y, new_x, snakes_grid):
    if 0 <= new_y < snakes_grid.shape[0] and \
            0 <= new_x < snakes_grid.shape[1]:
        return True
    return False

def set_low_y_low_x(y,x, snakes_grid, depth=1):
    '''
    Good for searching entire blocks but not strips of vertical and horizontal for which
    separate calls need to be made
    '''
    # find next possible head pos
    low_y, low_x = y - depth, x - depth
    # set at shape for slicing
    high_y, high_x = y + (depth+1), x + (depth+1)
    # if in bounds not boundary
    if y - 1 < 0:
        low_y = 0
    # if at shape that is too high
    elif y+2 > snakes_grid.shape[0]:
        high_y = snakes_grid.shape[0]
    if x-1 < 0:
        low_x = 0
    elif x+2 > snakes_grid.shape[1]:
        high_x = snakes_grid.shape[1]

    return low_y, high_y, low_x, high_x


def check_connected(y, x, grid, search_val=0, search_axis='both', search_type='any', depth=1):
    '''
    depth is how far from y, x, search_type = any, one, cumulative
    '''
    low_y, high_y, low_x, high_x = 0, 0, 0, 0
    # set lows and highs
    # check block
    if search_axis=='both':
        low_y, high_y, low_x, high_x = set_low_y_low_x(y,x, grid, depth=depth)
    #horiz
    elif search_axis=='x':
        low_y, high_y = y, y
        _,_,low_x, high_x = set_low_y_low_x(y,x,grid, depth=depth)
    elif search_axis=='y':
        low_x, high_x = x,x
        low_y, high_y, _,_ = set_low_y_low_x(y,x, grid, depth=depth)
    # search types
    # any values in the search space
    if search_type == 'any':
        if np.any(grid[low_y:high_y, low_x:high_x]==search_val):
            return True
    # count values in search space
    elif search_type=='count':
        return np.count_nonzero(grid[low_y:high_y, low_x:high_x]==search_val)

def heuristic(start_node, goal_node):
    '''
    np.linalg.norm(,1)
    '''
    start_x = start_node[1]
    start_y = start_node[0]
    goal_x = goal_node[1]
    goal_y = goal_node[0]
    dx = abs(start_x - goal_x)
    dy = abs(start_y - goal_y)
    return dx + dy

