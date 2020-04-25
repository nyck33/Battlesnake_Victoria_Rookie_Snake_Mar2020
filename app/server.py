import json
import os
import random
import numpy as np
from math import ceil
import bottle
from bottle import HTTPResponse
# import time
from timeit import default_timer as timer
from skimage.morphology import flood

# my_moves
delta = [[-1, 0],  # go up
         [0, -1],  # go left
         [1, 0],  # go down
         [0, 1]]  # go right

diag_delta = [[-1,-1], # nw
              [1,-1], # sw
              [1,1], # se
              [-1,1]] # ne

rev_delta = [[1, 0],  # back down
         [0, 1],  # back right
         [-1, 0],  # back up
         [0, -1]]  # back left

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
next_samehead_val = 11
next_smhead_val = 8

next_heads = [next_smhead_val, next_samehead_val, next_bighead_val]
curr_bodies = [small_head_val, same_head_val, big_head_val, body_val, my_body_val]
curr_heads = [small_head_val, same_head_val, big_head_val]


@bottle.route("/")
def index():
    return "I'm nasty."


@bottle.post("/ping")
def ping():
    """
    Used by the Battlesnake Engine to make sure your snake is still working.
    """
    return HTTPResponse(status=200)


@bottle.post("/start")
def start():
    """
    Called every time a new Battlesnake game starts and your snake is in it.
    Your response will control how your snake is displayed on the board.
    """
    data = bottle.request.json
    # #print(f"start_data:\n{json.dumps(data, indent=2)}")

    response = {"color": "#f2d933", "headType": "fang", "tailType": "bolt"}
    return HTTPResponse(
        status=200,
        headers={"Content-Type": "application/json"},
        body=json.dumps(response),
    )


def search(goal_y, goal_x, my_head_y, my_head_x, snakes_grid,
           check_path=False):
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

                        if closed[new_y, new_x] == 0 and \
                                (snakes_grid[new_y, new_x] == 0 or
                                 snakes_grid[new_y, new_x] % next_smhead_val == 0
                                 or snakes_grid[new_y, new_x] ==
                                 next_samehead_val):
                            # next_safeheads):
                            g2 = g + cost
                            f2 = g2 + heuristic_map[new_y, new_x]
                            open_arr.append([f2, g2, new_y, new_x])
                            closed[new_y, new_x] = 1

    # found goal or resigned
    if check_path:
        ##print(f'check expand:\n {expand}')
        return found

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


    else:
        move_num = 0
        my_move = 'fuzz'
        path_found = False
    ##print('return')
    return move_num, my_move, found

    '''
    # possible multipaths from start
    # found goal or resigned
    found_path = False
    moves_idxs = []
    if check_path:
        return found
    if found and not check_path:
        # using np.min so make -1 all np.inf
        expand[expand==-1]=np.inf
        # start at goal and work backwards to start, adding multiple paths
        start_y, start_x = goal_y, goal_x
        high_y, high_x = 0,0
        low_y, low_x = 0,0
        y,x = goal_y, goal_x
        count = 0
        val = expand[goal_y, goal_x]
        # nodes that can be traversed back to start
        curr_nodes = np.array([goal_y, goal_x])
        # indices of path to goal with the value of the node
        path_arrs = [[val, curr_nodes]]
        next_paths= np.zeros(1)
        while True:
            if len(path_arrs)>0:
                # get the current level nodes
                curr_nodes = path_arrs[count]
                for j in range(len(curr_nodes)):
                    # get the value to beat and node idx from current level
                    val = curr_nodes[j][0]
                    curr_idx = curr_nodes[j][1]
                    y,x = curr_idx
                    act_val = expand[y,x]
                    assert val== act_val, f'val {val} vs. act_val {act_val}'
                    # set boundaries for search of connected node from y,x
                    low_y, low_x, high_y, high_x = set_low_y_low_x(y,x, snakes_grid)
                    #get indicies of values lower than val in 4 connected to node
                    next_paths = np.argwhere(expand[low_y: high_y, low_x:high_x] < val)
                    # if there are such nodes with lower values proceed
                    next_level = []
                    if next_paths.shape[0]!= 0:
                        for k in range(next_paths.shape[0]):
                            next_idx = next_paths[k]
                            next_y, next_x = next_idx
                            # find the value at that next node
                            next_val = expand[next_y, next_x]
                            next_level.append([next_val, np.array([next_y, next_x])])
                    # sort by value of the node with lowest up front
                    next_level_sorted = sorted(next_level, key=lambda x: x[0])
                    path_arrs.append(next_level_sorted)

                count+=1
            else:
                break
        # sort path_arrs by value, want 0 at front
        # only attach
        path_arrs_sorted = sorted(path_arrs, key=lambda x: x[0][0])
        # array of next indices from the start head_y head_x
        for k in range(len(path_arrs_sorted)):
            level = path_arrs_sorted[k]
            for m in range(len(level)):
                curr = level[m]
                if curr[0]==0:
                  moves_idxs.append(curr[1])


            # multipaths for possible multimoves at start
            
            #shortest path
            #val = np.min(expand[low_y: high_y, low_x:high_x])
            # index of val
            idx = np.argwhere(expand==val)[0]
            y = idx[0]
            x = idx[1]
            path_arr.append(idx)

            if idx[0]==my_head_y and idx[1]==my_head_x:
                found_path = True
                break
    

    ##print('return')
    return expand, moves_idxs, found_path
    '''

def fill_food_arr(food, snakes_grid, my_head_y, my_head_x):
    # list in order of nearest to furthest food tuples (dist, y,x)
    food_arr = []
    food_value = 99
    food_grid = np.zeros(snakes_grid.shape, dtype=np.int)
    for z in range(len(food)):
        food_y, food_x = food[z]['y'], food[z]['x']
        food_dist = heuristic([my_head_y, my_head_x],
                              [food_y, food_x])
        food_grid[food_y, food_x] = food_value
        food_arr.append([food_dist, food_y, food_x])

    food_array = sorted(food_arr, key=lambda x: x[0])
    # #print(f'\n\nfood arr {food_arr}\n\n')
    return food_array, food_grid


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
        if 0 <= next_head_y < snakes_grid.shape[0] \
                and 0 <= next_head_x < snakes_grid.shape[1]:
            if new_grid[next_head_y, next_head_x] not in curr_bodies:
                new_grid[next_head_y, next_head_x]+= next_head_val

    return new_grid


def fill_snakes_grid(snakes, width, height, my_body_len, my_id):
    '''
    body_val = 1
    head_val = 2
    next_bighead_val = 10
    next_samehead_val = 3
    '''
    # my_moves
    delta = [[-1, 0],  # go up
             [0, -1],  # go left
             [1, 0],  # go down
             [0, 1]]  # go right

    snake_heads = []
    snake_tails = []
    growing = False
    # second grid for checking open path to tail
    snakes_grid = np.zeros((width, height), dtype=np.int)
    # each remaining snake gets its own grid for predictions

    solo_grids = []

    for j in range(len(snakes)):
        solo_grid = np.zeros(snakes_grid.shape, dtype=np.int)
        curr_snake = snakes[j]
        if curr_snake['id'] == my_id:
            my_snake = True
        else:
            my_snake = False
        # fill grid with bodies
        for k in range(len(curr_snake['body'])):
            # heads of opp snakes
            if k == 0:
                head_y = curr_snake['body'][k]['y']
                head_x = curr_snake['body'][k]['x']
                # if smaller
                if len(curr_snake['body']) < my_body_len and not my_snake:
                    snakes_grid[head_y, head_x] = small_head_val
                    solo_grid[head_y, head_x] = my_head_val
                    snake_heads.append([next_smhead_val, head_y, head_x])

                # if it's the heads of bigger or equal snakes
                elif len(curr_snake['body']) > my_body_len and not my_snake:
                    snakes_grid[head_y, head_x] = big_head_val
                    solo_grid[head_y, head_x] = my_head_val
                    # append to heads list
                    snake_heads.append([next_bighead_val, head_y, head_x])

                # todo: equal size
                elif len(curr_snake['body']) == my_body_len and not my_snake:
                    snakes_grid[head_y, head_x] = same_head_val
                    solo_grid[head_y, head_x] = same_head_val
                    # todo: append to heads list or not?
                    snake_heads.append([next_samehead_val, head_y, head_x])

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
                    solo_grid[body_y, body_x] = body_val
                # fill solo grid
                elif my_snake:
                    snakes_grid[body_y, body_x] = body_val
                    solo_grid[body_y, body_x] = body_val
            # tails
            elif k == (len(curr_snake['body']) - 1):
                body_y = curr_snake['body'][k]['y']
                body_x = curr_snake['body'][k]['x']

                if curr_snake['health'] == 100:
                    growing = True
                    snake_tails.append([growing, body_y, body_x])
                    snakes_grid[body_y, body_x] = body_val
                    solo_grid[body_y, body_x] = body_val
                else:
                    # only include tails of non growing snakes
                    snake_tails.append([growing, body_y, body_x])
                growing = False
        solo_grids.append([len(curr_snake['body']),
                           curr_snake['id'], np.copy(solo_grid)])
    # done iterating
    bodies_grid = np.copy(snakes_grid)
    # mark next heads after bodies filled
    for i in range(len(snake_heads)):
        curr_head = snake_heads[i]
        curr_next_val = curr_head[0]
        curr_y = curr_head[1]
        curr_x = curr_head[2]
        snakes_grid = mark_next_heads(curr_y, curr_x, snakes_grid, curr_next_val)

    # asc from shortest to longest
    solo_grids = sorted(solo_grids, key=lambda x: x[0])

    return snakes_grid, bodies_grid, solo_grids, snake_heads, snake_tails

def check_in_bounds(new_y, new_x, snakes_grid):
    if 0 <= new_y < snakes_grid.shape[0] and \
            0 <= new_x < snakes_grid.shape[1]:
        return True
    return False

def check_path_to_tail(head_y, head_x, move_num, snakes_grid,
                       snake_tails):
    found_path = False
    new_head_y = head_y + delta[move_num][0]
    new_head_x = head_x + delta[move_num][1]
    if check_in_bounds(new_head_y, new_head_x, snakes_grid):
        # check that we can reach a tail of a non growing snake
        for q in range(len(snake_tails)):
            growing = snake_tails[q][0]
            if growing:
                for i in range(len(delta)):
                    alt_y = snake_tails[q][1] + delta[i][0]
                    alt_x = snake_tails[q][2] + delta[i][1]
                    if check_in_bounds(alt_y, alt_x, snakes_grid):
                        if snakes_grid[alt_y, alt_x] not in curr_bodies:
                            found_path = search(alt_y, alt_x, new_head_y,
                                                new_head_x, snakes_grid,
                                                check_path=True)
            else:
                tail_y = snake_tails[q][1]
                tail_x = snake_tails[q][2]
                snakes_grid[tail_y, tail_x] = 0
                found_path = search(tail_y, tail_x, new_head_y,
                                    new_head_x, snakes_grid,
                                    check_path=True)

            if found_path:
                break
            else:
                found_path = False
                # print('check tail fail')
    return found_path


def board_control_heuristics(solo_grids, snakes_grid, bodies_grid, me):
    '''
    find the best move for each snake based on total num of reachable squares
    keep the best grid that shows this next pos
    append the reach_pts, next idx and grid to list
    todo: currently only moving my head into possible future pos's but should move all heads
    '''
    reach = np.inf
    seed_point = 1000
    dist_grids = []
    highest = -np.inf
    my_id = me['id']
    total_squares = 121
    best_grid = np.zeros(snakes_grid.shape, dtype=np.float64)
    h_list = []
    for i in range(len(solo_grids)):
        snake_len = solo_grids[i][0]
        snake_id = solo_grids[i][1]
        solo_grid = solo_grids[i][2]
        if snake_id == my_id:
            # current head
            idx = np.argwhere(solo_grid == my_head_val)
            y, x = idx[0,:]
            low_y, low_x, high_y, high_x = set_low_y_low_x(y,x, snakes_grid)

            # get all next poses in bounds and free:
            is_zero = np.argwhere(snakes_grid[low_y:high_y, low_x:high_x] \
                                     ==0)
            is_next_small = np.argwhere(snakes_grid[low_y:high_y, low_x:high_x] \
                                     == next_smhead_val)
            is_next_same = np.argwhere(snakes_grid[low_y:high_y, low_x:high_x] \
                                     == next_samehead_val)
            is_next_big = np.argwhere(snakes_grid[low_y:high_y, low_x:high_x] \
                                     == next_samehead_val)
            next_poses = np.vstack((is_zero, is_next_small, is_next_same,
                                    is_next_big))
            for j in range(next_poses.shape[0]):
                next_grid = np.copy(snakes_grid)
                next_y, next_x = next_poses[j, :]

                # old head is body
                next_grid[y, x] = my_body_val
                # new head set as 0 for flood fill, needs same value
                next_grid[next_y, next_x] = 0
                # find accessible points but want to zero out nextheads 
                next_grid[next_grid == next_bighead_val] = 0
                next_grid[next_grid == next_samehead_val] = 0
                next_grid[next_grid == next_smhead_val] = 0
                # mask is bool array
                mask = flood(next_grid, (next_y, next_x), connectivity=1)
                next_flooded = np.copy(next_grid)
                next_flooded = next_flooded.astype(np.float64)
                # set values on arr to np.inf based on bool
                next_flooded[mask] = reach
                # set new head value to something else and get the heuristics
                next_flooded[next_y, next_x] = seed_point
                next_flooded[next_flooded == reach] = \
                    np.abs(np.argwhere(next_flooded == reach) -
                           np.argwhere(next_flooded == seed_point)).sum(1)
                # set curr_bodies to np.inf so only reachable index numbers remain as non np.inf
                next_flooded[next_flooded == body_val] = np.inf
                next_flooded[next_flooded == my_body_val] = np.inf
                next_flooded[next_flooded == my_head_val] = np.inf
                next_flooded[next_flooded == small_head_val] = np.inf
                next_flooded[next_flooded == same_head_val] = np.inf
                next_flooded[next_flooded == big_head_val] = np.inf
                
                # reverse the points so closer is higher pts, 0's were unreachable
                points_grid = np.copy(next_flooded)
                points_grid[points_grid==np.inf]=0
                # calculate points and store in grid
                points_grid[0<points_grid] = total_squares - points_grid[0<points_grid]
                # calc points
                points_grid[points_grid==np.inf]=0
                # only the move pts remain
                reach_pts = points_grid.sum()
                # append the next index, reach_pts and grid to list
                h_list.append([reach_pts, [next_y, next_x], next_flooded, points_grid])
                '''
                # only keep max of the grids
                if next_grid.sum() > highest:
                    highest = next_grid.sum()
                    best_grid = next_grid
                '''
        # sort from lowest to highest snake
        h_list_sorted = sorted(h_list, key=lambda x: x[0])
        # highest points first
        h_list = h_list_sorted[::-1]
    return h_list

def set_low_y_low_x(y,x, snakes_grid):
    # find next possible head pos
    low_y, low_x = y - 1, x - 1
    # set at shape for slicing
    high_y, high_x = y + 2, x + 2
    # if in bounds not boundary
    if y - 1 < 0:
        low_y = 0
    # if at shape that is too high
    elif y >= snakes_grid.shape[0] - 1:
        high_y = snakes_grid.shape[0] - 1
    if x <= 0:
        low_x = 0
    elif x >= snakes_grid.shape[1] - 1:
        high_x = snakes_grid.shape[1] - 1

    return low_y, low_x, high_y, high_x

def check_proximity(heuristics_list,snakes_grid, y, x):
    '''
    my_head_y, my_head_x
    Look for bigheads
    small_head_val = 1
    my_head_val = 3
    same_head_val = 2
    big_head_val = 5
    body_val = 4
    my_body_val = 7
    next_bighead_val = 9
    next_samehead_val = 11
    next_smhead_val = 8
    h_list is [reach_pts, [next_y, next_x], next_flooded, points_grid])
    Count number of next head types up to 4 moves away
    '''
    bighead_count=0
    eq_head_count=0
    sm_head_count=0
    next_eqhead_val = 23
    next_predator_val = 191
    next_prey_val = 2

    tactic = 'neutral'
    # iterate h_list get scores and find move with lowest next_bighead_count or highest_next_small_head_count
    scores_per_move = []
    next_pos = []
    # check up to 3 moves away
    depth = 3

    # check 2 away, not just one so add 1 to head y and x
    low_x, low_y, _,_ = set_low_y_low_x(y-1,x-1, snakes_grid)
    _, _, high_x, high_y = set_low_y_low_x(y+1, x+1, snakes_grid)

    next_head_poses = np.argwhere(snakes_grid[low_y:high_y, low_x: high_x]== next_bighead_val)
    curr_head_poses = np.argwhere(snakes_grid[low_y:high_y, low_x: high_x]==big_head_val)
    next_smhead_poses = np.argwhere(snakes_grid[low_y:high_y, low_x: high_x] == next_smhead_val)
    curr_smhead_poses = np.argwhere(snakes_grid[low_y:high_y, low_x: high_x] == small_head_val)

    # check diagonals
    for i in range(len(diag_delta)):
        next_y = diag_delta[i][0]
        next_x = diag_delta[i][1]
        if check_in_bounds(next_y, next_x, snakes_grid):
            if snakes_grid[next_y, next_x] == next_bighead_val \
                    or snakes_grid[next_y, next_x] == big_head_val:
                bighead_count+=1
            elif snakes_grid[next_y, next_x] == next_smhead_val \
                    or snakes_grid[next_y, next_x] == small_head_val:
                sm_head_count+=1

    bighead_count+=next_head_poses.shape[0] + curr_head_poses.shape[0]
    sm_head_count+= next_smhead_poses.shape[0] + curr_smhead_poses.shape[0]
    if bighead_count >0:
        tactic = 'evade'
    elif bighead_count==0 and sm_head_count>0:
        tactic='attack'
    else:
        tactic='neutral'
    return tactic

    '''
    for j in range(len(heuristics_list)):
        curr_h = heuristics_list[j]
        reach_pts = curr_h[0]
        next_pos = curr_h[1]
        flood_grid = curr_h[2]
        #"points" or
        points_grid = curr_h[3]
        # iterate for depth-1, set iterations to depth+1
        # check flood grid for current move
        for k in range(1,depth+1,1):
            idxs = np.argwhere(flood_grid==k)
            if idxs.shape[0]>0:
                for m in range(idxs.shape[0]):
                    y,x = idxs[m,:]
                    #(5-k) means 1 move away is 4*, 2 moves away is 5-2=3*...
                    if snakes_grid[y,x] == next_bighead_val:
                        bighead_count+=  (5-k) * next_predator_val
                    elif snakes_grid[y,x] == eq_head_count:
                        eq_head_count += (5-k) * next_eqhead_val
                    elif snakes_grid[y,x] == next_smhead_val:
                        sm_head_count+= (5-k) * next_prey_val

            total_score = bighead_count + eq_head_count+sm_head_count
            scores_per_move.append([total_score, next_pos])
    # lowest score to front is what we want here
    scores_per_move_sorted = sorted(scores_per_move, key=lambda x: x[0])
    return scores_per_move_sorted
    
    # find how many next_big, next_same are within 4 moves
    
    for i in range(next_poses.shape[0]):
        next_y, next_x = next_poses[i,:]
        # check two moves away
        low_y, low_x, _, _ = set_low_y_low_x(next_y-1, next_x-1, snakes_grid)
        _, _, high_y, high_x = set_low_y_low_x(next_y+1, next_x+1, snakes_grid)
        # y-axis
        bighead_count+= big_head_val * np.count_nonzero(snakes_grid[low_y:high_y, next_x] == next_bighead_val)
        eq_head_count+= next_eqhead_val * np.count_nonzero(snakes_grid[low_y:high_y, next_x] == next_samehead_val)
        # x-axis
        bighead_count += big_head_val * np.count_nonzero(snakes_grid[next_y, low_x:high_x] == next_bighead_val)
        eq_head_count += next_eqhead_val * np.count_nonzero(snakes_grid[next_y, low_x:high_x] == next_samehead_val)
        # get diagonals
    '''
@bottle.post("/move")
def move():
    """
    Called when the Battlesnake Engine needs to know your next my_move.
    The data parameter will contain information about the board.
    Your response must include your my_move of up, down, left, or right.
    """
    start = timer()

    # my_moves
    delta = [[-1, 0],  # go up
             [0, -1],  # go left
             [1, 0],  # go down
             [0, 1]]  # go right

    delta_name = ['up', 'left', 'down', 'right']

    # call for data
    data = bottle.request.json
    turn = data['turn']
    # pretty #print
    ##print(f"turn: {turn}\n{json.dumps(data, indent=2)}")
    # board size
    width = data['board']['width']
    height = data['board']['height']

    # my head and body locations
    snakes = data['board']['snakes']
    me = data['you']
    # my health
    my_health = me['health']
    ##print(f'me\n{me}')
    my_head_y = me['body'][0]['y']
    my_head_x = me['body'][0]['x']

    my_tail_y = me['body'][-1]['y']
    my_tail_x = me['body'][-1]['x']

    # find next tail
    my_next_tail_y = me['body'][-2]['y']
    my_next_tail_x = me['body'][-2]['x']

    next_tails = []
    for i in range(len(snakes)):
        next_tail_y = snakes[i]['body'][-2]['y']
        next_tail_x = snakes[i]['body'][-2]['x']

        next_tails.append([next_tail_y, next_tail_x])

    ##print(f'tail yx = {my_tail_y},{my_tail_x}\n'
    #     f'nexttail_yx: {next_tail_y},{next_tail_x}')
    my_id = me['id']

    # for comparison with opponent's snakes
    my_body_len = len(me['body'])

    # moves info
    which_move = ''
    my_move = ''
    tactic = ''
    # flags
    path_found = False
    
    # make state info
    snakes_grid, bodies_grid, solo_grids, snake_heads, snake_tails = \
        fill_snakes_grid(snakes, width, height, my_body_len, my_id)
    
    # [reach_pts, next_idx, flood grid, points_grid] from highest to lowest reach points
    heuristics_list = board_control_heuristics(solo_grids, snakes_grid, bodies_grid, me)

    # get ranked moves for locality
    # returns list of lists ([total_score, next_pos])
    tactic = check_proximity(heuristics_list, snakes_grid, my_head_y, my_head_x)

    if tactic=='evade':
        for j in range(len(heuristics_list)):
            move_idx = heuristics_list[0][1]
            next_y, next_x = move_idx
            for i in range(len(rev_delta)):
                prev_y = next_y + rev_delta[i][0]
                prev_x = next_x + rev_delta[i][1]
                if check_in_bounds(prev_y, prev_x, snakes_grid):
                    if prev_y == my_head_y and prev_x == my_head_x:
                        move_num = i
                        my_move = delta_name[i]
                        path_found = True

                    if path_found:
                        break


    # list of dicts of food locations
    food = data['board']['food']
    # list in order of nearest to furthest food tuples (dist, y,x)
    food_arr = []
    food_grid = np.copy(snakes_grid)
    # if there is food
    if len(food) > 0:
        food_arr, food_grid = fill_food_arr(food, snakes_grid, my_head_y,
                                            my_head_x)
    # there is a food so A star for route to food using snake grid for g
    food_count = 0
    # closest food location
    nearest_food_dist = food_arr[0][0]
    # if less than 1, get food
    health_food_ratio = my_health/nearest_food_dist
    if health_food_ratio <=2.0:
        tactic='get food'

    attack = False
    # if me_longest, chase 8s
    if tactic=='attack' and not path_found:
        next_small_targets = np.argwhere(snakes_grid%8==0)
        for i in range(next_small_targets.shape[0]):
            victim_y, victim_x = next_small_targets[i,:]
            for j in range(len(delta)):
               target_y = victim_y + delta[j][0]
               target_x = victim_x + delta[j][1]
               if check_in_bounds(target_y, target_x, snakes_grid):
                   if snakes_grid[target_y, target_x] ==0 or \
                        snakes_grid[target_y, target_x]%8 ==0:
                       # todo: use move_idxs and check against heuristics list reach score, pick move that matches, otherwise favor reach score and
                       # find path to tail
                       move_num, my_move, path_found = search(victim_y, victim_x, my_head_y, my_head_x,
                               snakes_grid)
                       if path_found:
                            found_free = check_path_to_tail(my_head_y, my_head_x,
                                                        move_num, snakes_grid,
                                                        snake_tails)
                            if found_free:
                                break
                       else:
                           which_move=' failed attack'

    # get food
    eating = False
    count = 0
    get_it = False
    # if get food or attack failed
    if (tactic=='neutral' or tactic=='get food') and not path_found:
        # print('food')
        while not eating and count < len(food_arr):
            curr_food = food_arr[count]
            food_dist = curr_food[0]
            food_y = curr_food[1]
            food_x = curr_food[2]
            food_count += 1
            if len(snakes) > 1:
                for i in range(len(snake_heads)):
                    curr_head = snake_heads[i]
                    head_type = curr_head[0]
                    snakehead_y = curr_head[1]
                    snakehead_x = curr_head[2]

                    opp_dist = heuristic([snakehead_y, snakehead_x],
                                         [food_y, food_x])
                    if food_dist < opp_dist:
                        get_it = True
                    elif head_type == small_head_val and \
                            food_dist <= opp_dist:
                        get_it = True
                    else:
                        get_it = False
                        break
            else:
                get_it = True

            if get_it:
                move_num, my_move, path_found = \
                    search(food_y, food_x, my_head_y, my_head_x,
                           snakes_grid)
                if path_found:

                    found_free = check_path_to_tail(my_head_y, my_head_x,
                                                    move_num, snakes_grid,
                                                    snake_tails)

                    if found_free:
                        which_move = 'get food'
                        eating = True
                    else:
                        path_found = False
                else:
                    path_found = False
                    which_move='failed food'

            count += 1

    # shorten food_arr
    # food_arr = food_arr[food_count:]
    count = 0
    # chase my tail
    if not path_found:
        # print('my tail')
        # chase tail if nothing in food_arr
        move_num, my_move, path_found = search(my_tail_y, my_tail_x,
                                               my_head_y, my_head_x,
                                               snakes_grid)
        if path_found:
            '''
            found_free = check_path_to_free(my_head_y, my_head_x,
                                move_num, snakes_grid, free_spaces_arr)
            '''
            found_free = check_path_to_tail(my_head_y, my_head_x,
                                            move_num, snakes_grid,
                                            snake_tails)
            if found_free:
                which_move = 'my tail'
            else:
                path_found = False
        else:
            path_found = False
            which_move = 'failed tail'

    count = 0

    # use h_list
    if my_move =='fuzz':
        #print('here')
        for j in range(len(heuristics_list)):
            move_idx = heuristics_list[j][1]
            next_y, next_x = move_idx
            for i in range(len(rev_delta)):
                prev_y = next_y + rev_delta[i][0]
                prev_x = next_x + rev_delta[i][1]
                if check_in_bounds(prev_y, prev_x, snakes_grid):
                    if prev_y == my_head_y and prev_x == my_head_x:
                        move_num = i
                        my_move = delta_name[i]
                        which_move='h_list last'
                        break


    shout = "get in my belly!"

    response = {"move": my_move, "shout": shout}
    end = timer()
   #print(f'\n\nturn: {turn}\ntime: {end-start}\nmy_move: {my_move}\n '
    #f'which_move: {which_move}\n\n')
    ##print(f'snakes_grid\n {snakes_grid}\nsolo_grid\n {solo_grid}\n')
    return HTTPResponse(
        status=200,
        headers={"Content-Type": "application/json"},
        body=json.dumps(response),
    )


def heuristic(start_node, goal_node):
    start_x = start_node[1]
    start_y = start_node[0]
    goal_x = goal_node[1]
    goal_y = goal_node[0]
    dx = abs(start_x - goal_x)
    dy = abs(start_y - goal_y)
    return dx + dy


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




@bottle.post("/end")
def end():
    """
    Called every time a game with your snake in it ends.
    """
    data = bottle.request.json
    # #print(f"end data:\n{json.dumps(data, indent=2)}")
    return HTTPResponse(status=200)


def main():
    bottle.run(
        application,
        host=os.getenv("IP", "0.0.0.0"),
        port=os.getenv("PORT", "8080"),
        debug=os.getenv("DEBUG", True),
    )


# Expose WSGI app (so gunicorn can find it)
application = bottle.default_app()

if __name__ == "__main__":
    main()