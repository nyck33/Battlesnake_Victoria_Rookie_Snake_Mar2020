import json
import os
import random
import numpy as np
from math import ceil
import bottle
from bottle import HTTPResponse
# import time
from timeit import default_timer as timer
from grid_data_maker import  *

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

def search(goal_y, goal_x, my_head_y, my_head_x, snakes_grid, check_path=False):
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
    found_path = False
    my_move = ''
    move_num = 0

    if check_path:
        snakes_grid[snakes_grid == next_samehead_val] = next_smhead_val

    # visited array
    closed = np.zeros(snakes_grid.shape, dtype=np.int)
    closed[my_head_y, my_head_x] = 1
    # expand is final map returned with numbered spots
    expand = np.full(snakes_grid.shape, -1, dtype=np.int)

    g = 0  # each step is 1
    heuristic_map = make_heuristic_map([goal_y, goal_x],
                                       snakes_grid)
    # #print(f'heuristics_map\n{heuristic_map}')
    f = g + heuristic_map[my_head_y, my_head_x]

    open_arr = [[f, g, my_head_y, my_head_x]]
    found = False  # set when search complete
    resign = False  # set when can't expand
    count = 0
    # calculate entire path
    while not found and not resign:
        if len(open_arr) == 0:
            resign = True
        else:
            open_arr.sort()
            open_arr.reverse()
            next_arr = open_arr.pop()
            y = next_arr[2]
            x = next_arr[3]
            g = next_arr[1]
            f = g + heuristic_map[y, x]
            expand[y, x] = count
            # if count==0:
            ##print(f'first expand\ny: {y}, x: {x}\n{expand}')
            count += 1

            if y == goal_y and x == goal_x:
                found = True
            else:
                for i in range(len(delta)):
                    new_y = y + delta[i][0]
                    new_x = x + delta[i][1]

                    # if in-bounds
                    if 0 <= new_y < snakes_grid.shape[0] and \
                            0 <= new_x < snakes_grid.shape[1]:
                        # if unvisited and traversible (smaller snake's nexthead
                        # is traversible)
                        #todo: need to account for cumulative nextheads
                        # in grid spots
                        if closed[new_y, new_x] == 0 and \
                                (snakes_grid[new_y, new_x] == 0 or
                                 snakes_grid[new_y, new_x] in next_ok_heads):
                            # next_safeheads):
                            g2 = g + cost
                            f2 = g2 + heuristic_map[new_y, new_x]
                            open_arr.append([f2, g2, new_y, new_x])
                            closed[new_y, new_x] = 1

    # found goal or resigned
    #todo: return multiple moves that actually get to goal
    if found and not check_path:
        # print('here')
        init_val = expand[goal_y, goal_x]
        # print(f'initval {init_val}')
        # move nums from delta
        moves_arr = []
        move_num = 0
        path_arr = [init_val]
        small_val = init_val
        start_y, start_x = goal_y, goal_x
        next_y, next_x = 0, 0
        low_y, low_x = 0, 0
        val = init_val
        found_path = False
        rev_move = 0
        while not found_path:
            for k in range(len(delta)):
                next_y = start_y + delta[k][0]
                next_x = start_x + delta[k][1]
                if 0 <= next_y < expand.shape[0] and \
                        0 <= next_x < expand.shape[1]:
                    val = expand[next_y, next_x]
                    ##print(f'val {val}')
                    if 0 <= val < small_val:
                        small_val = val
                        move_num = k
                        ##print(f'movenum {move_num}')
                        low_y = next_y
                        low_x = next_x
                ##print('forloop')
            rev_move = (move_num + 2) % 4
            moves_arr.append(rev_move)
            if low_y == my_head_y and low_x == my_head_x:
                found_path = True
                break
            start_y = low_y
            start_x = low_x
            val = small_val
            ##print(f'moves_arr {moves_arr}')
        ##print('out')
        moves_seq = moves_arr[::-1]
        move_num = moves_seq[0]
        my_move = delta_name[move_num]


    elif check_path:
        ##print(f'check expand:\n {expand}')
        return found

    else:
        move_num = 0
        my_move = 'fudge'

    ##print('return')
    return move_num, my_move, found

def check_path_to_tail(snakes, head_y, head_x, move_num, snakes_grid, solo_grids,
                       snake_tails):
    '''
    should respect locality and look for a path to something closer or use solo grid to find own tail
    and do a flood fill to take the action that leads to more spaces locally
    '''
    # todo: use solo grid, ensure we do not trap ourselves, ie. get to own tail
    # todo: next check locality
    found_path = False
    new_head_y = head_y + delta[move_num][0]
    new_head_x = head_x + delta[move_num][1]
    if check_in_bounds(new_head_y, new_head_x, snakes_grid):
        # not a solo game
        if len(snakes) > 1:
            for i in range(len(snakes)):
                snake = snakes[i]
                if snake['health'] == 100 and len(snakes) > 1:
                    free_spaces = find_free_spaces(snakes_grid, head_y, head_x)
                    free = free_spaces[::-1]
                    for j in range(len(free)):
                        free_y = free[j][1]
                        free_x = free[j][2]

                        found_free = check_path_to_tail(snakes, )
                else:
                    tail_y = snake['body'][-1]['y']
                    tail_x = snake['body'][-1]['x']
                    # zero out tail just in case
                    snakes_grid[tail_y, tail_x] = 0
                    found_path = search(tail_y, tail_x, new_head_y,
                                        new_head_x, snakes_grid,
                                        check_path=True)
        # solo game
        elif len(snakes) == 1:
            snake = snakes[0]
            tail_y = snake['body'][-1]['y']
            tail_x = snake['body'][-1]['x']
            if snakes_grid[new_head_y, new_head_x] == my_body_val or \
                    snakes_grid[new_head_y, new_head_x] == body_val:

            if snake['health'] == 100:

                def check_path_to_free(head_y, head_x, move_num, snakes_grid, free_array):
                    '''
                    Only check path to free that is at least board width away
                    '''
                    found_path = False
                    min_dist = snakes_grid.shape[1] * 1.5
                    free_arr = free_array[::-1]
                    new_head_y = head_y + delta[move_num][0]
                    new_head_x = head_x + delta[move_num][1]
                    if 0 <= new_head_y < snakes_grid.shape[0] and \
                            0 <= new_head_x < snakes_grid.shape[1]:
                        # check that we can reach a free space
                        for i in range(len(free_arr)):
                            free_y, free_x = free_arr[i][1], free_arr[i][2]
                            if heuristic([free_y, free_x], [new_head_y, new_head_x]) >= \
                                    min_dist:

                                _, _, found_path = search(free_y, free_x, new_head_y,
                                                          new_head_x, snakes_grid)
                                if found_path:
                                    break

                    return found_path
