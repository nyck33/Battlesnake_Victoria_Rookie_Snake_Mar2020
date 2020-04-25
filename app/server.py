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
           check_path=False, count_path=False):
    '''
    small_head_val = 1
    my_head_val = 3
    same_head_val = 2
    big_head_val = 5
    body_val = 4
    my_body_val = 7
    next_bighead_val = 9
    next_samehead_val = 11
    next_smhead_val = 8
    '''
    found_path = False
    my_move = ''
    move_num = 0
    # for count_path
    num_moves = 0

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
                                 or snakes_grid[new_y, new_x] ==next_samehead_val or \
                                 snakes_grid[new_y, new_x]==next_bighead_val):
                            # next_safeheads):
                            g2 = g + cost
                            f2 = g2 + heuristic_map[new_y, new_x]
                            open_arr.append([f2, g2, new_y, new_x])
                            closed[new_y, new_x] = 1

    # found goal or resigned
    if found and check_path:
        ##print(f'check expand:\n {expand}')
        return found

    elif found and not check_path:
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
                if check_in_bounds(next_y, next_x, snakes_grid):
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
            num_moves+=1
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

        if count_path:
            return num_moves

    else:
        move_num = 0
        my_move = 'fuzz'
        found = False


    return move_num, my_move, found


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
            # heads of snakes
            if k == 0:
                head_y = curr_snake['body'][k]['y']
                head_x = curr_snake['body'][k]['x']
                # if opp smaller
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

                # fill solo grid for checking flood fill in queue
                # desc from largest
                elif len(curr_snake['body']) == my_body_len and my_snake:
                    solo_grid[head_y, head_x] = my_head_val
                    snakes_grid[head_y, head_x] = my_head_val
            # all snakes body and my head and body except tail
            elif 0 < k < (len(curr_snake['body']) - 1):
                body_y = curr_snake['body'][k]['y']
                body_x = curr_snake['body'][k]['x']
                if my_snake:
                    snakes_grid[body_y, body_x] = my_body_val
                    solo_grid[body_y, body_x] = my_body_val
                else:
                    snakes_grid[body_y, body_x] = body_val
                    solo_grid[body_y, body_x] = body_val
            # tails
            elif k == (len(curr_snake['body']) - 1):
                body_y = curr_snake['body'][k]['y']
                body_x = curr_snake['body'][k]['x']

                if curr_snake['health'] == 100:
                    growing = True
                    snake_tails.append([growing, body_y, body_x])
                    if my_snake:
                        snakes_grid[body_y, body_x] = my_body_val
                        solo_grid[body_y, body_x] = my_body_val
                    else:
                        snakes_grid[body_y, body_x] = body_val
                        solo_grid[body_y, body_x] = body_val
                else:
                    # only include tails of non growing snakes
                    # only these can be targeted for next move
                    snake_tails.append([growing, body_y, body_x])
                growing = False
        solo_grids.append([len(curr_snake['body']),
                           curr_snake['id'], np.copy(solo_grid)])
    # no next heads in bodies grid
    bodies_grid = np.copy(snakes_grid)
    # mark next heads after bodies filled
    for i in range(len(snake_heads)):
        curr_head = snake_heads[i]
        curr_next_val = curr_head[0]
        curr_y = curr_head[1]
        curr_x = curr_head[2]
        snakes_grid = mark_next_heads(curr_y, curr_x, snakes_grid, curr_next_val)

    # asc from longest to shortest
    temp = sorted(solo_grids, key=lambda x: x[0])
    solo_grids = temp[::-1]

    return snakes_grid, bodies_grid, solo_grids, snake_heads, snake_tails



def check_path_to_tail(head_y, head_x, move_num, snakes_grid,
                       snake_tails):
    found_path = False
    new_head_y = head_y + delta[move_num][0]
    new_head_x = head_x + delta[move_num][1]
    if check_in_bounds(new_head_y, new_head_x, snakes_grid):
        # check that we can reach a tail of a non growing snake
        for q in range(len(snake_tails)):
            growing = snake_tails[q][0]
            tail_y, tail_x = snake_tails[q][1], snake_tails[q][2]
            if growing: # find alternate
                low_y, low_x, high_y, high_x = set_low_y_low_x(tail_y,
                        tail_x, snakes_grid)
                # can only be 0 or next head
                free_spaces = np.argwhere((snakes_grid[low_y:high_y,
                                          low_x:high_x]!= my_body_val) |
                                          (snakes_grid[low_y:high_y,
                                          low_x:high_x] != my_head_val) |
                                           (snakes_grid[low_y:high_y,
                                            low_x:high_x] != body_val) |
                                           (snakes_grid[low_y:high_y,
                                           low_x:high_x] != small_head_val) |
                                          (snakes_grid[low_y:high_y,
                                           low_x:high_x] != same_head_val) |
                                          (snakes_grid[low_y:high_y,
                                          low_x:high_x] != big_head_val))

                '''
                not_my_body = np.argwhere(snakes_grid[low_y:high_y,
                                          low_x, high_x]!= my_body_val)
                not_my_head = np.argwhere(snakes_grid[low_y:high_y,
                                          low_x, high_x]!= my_head_val)
                not_body = np.argwhere(snakes_grid[low_y:high_y,
                                          low_x, high_x]!= body_val)
                not_sm = np.argwhere(snakes_grid[low_y:high_y,
                                          low_x, high_x]!= small_head_val)
                not_eq = np.argwhere(snakes_grid[low_y:high_y,
                                          low_x, high_x]!= same_head_val)
                not_big = np.argwhere(snakes_grid[low_y:high_y,
                                          low_x, high_x]!= big_head_val)
                free_spaces = np.vstack((not_my_body, not_my_head,
                                         not_body, not_sm, not_eq, not_big))
                '''
                if free_spaces.shape[0] > 0:
                    for i in range(free_spaces.shape[0]):
                        curr_free = free_spaces[i,:]
                        alt_y, alt_x = curr_free
                        found_path = search(alt_y, alt_x, new_head_y,
                                            new_head_x, snakes_grid,
                                            check_path=True)
                        if found_path:
                            break
            else:
                tail_y = snake_tails[q][1]
                tail_x = snake_tails[q][2]
                # just to make sure
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

def check_in_bounds(new_y, new_x, snakes_grid):
    if 0 <= new_y < snakes_grid.shape[0] and \
            0 <= new_x < snakes_grid.shape[1]:
        return True
    return False

def board_control_heuristics(turn, solo_grids, snakes_grid, bodies_grid, me):
    '''
    find the best move for each snake based on total num of reachable squares
    keep the best grid that shows this next pos
    append the reach_pts, next idx and grid to list
    todo: currently only moving my head into possible future pos's but should move all heads
    small_head_val = 1
    my_head_val = 3
    same_head_val = 2
    big_head_val = 5
    body_val = 4
    my_body_val = 7
    next_bighead_val = 9
    next_samehead_val = 11
    next_smhead_val = 8
    '''
    reach = 99
    reach_pts=0
    seed_point = np.inf
    dist_grids = []
    highest = -np.inf
    my_id = me['id']
    total_squares = 121
    best_grid = np.zeros(snakes_grid.shape, dtype=np.float64)
    h_list = []

    # current head
    if turn == 0:
        idx = np.argwhere(snakes_grid == my_body_val)
    else:
        idx = np.argwhere(snakes_grid == my_head_val)
    #print(f'snakesgrid\n{snakes_grid} myhead{my_head_val}')
    #print(f'idx {idx}')
    y, x = idx[0,:]
    #print(f'yx {y}, {x}')
    for k in range(len(delta)):
        new_y = y + delta[k][0]
        new_x = x + delta[k][1]
        if check_in_bounds(new_y, new_x, snakes_grid):
            if snakes_grid[new_y, new_x]==0 or \
                    snakes_grid[new_y, new_x] in next_heads:
                next_grid = np.copy(snakes_grid)
                next_y, next_x = new_y, new_x
                #print(f'nexty, nextx {next_y}{next_x}')
                # old head is body
                next_grid[y, x] = my_body_val
                # new head set as 0 for flood fill, needs same value to work
                next_grid[next_y, next_x] = 0
                # find accessible points but want to zero out nextheads
                next_grid[(next_grid == next_bighead_val) |
                          (next_grid == next_samehead_val) |
                          (next_grid == next_smhead_val)] = 0
                #print(f'nextgrid\n {next_grid}')
                # mask is bool array
                mask = flood(next_grid, (next_y, next_x),  connectivity=1)
                #print(f'mask\n{mask}')
                next_flooded = np.copy(next_grid)
                next_flooded = next_flooded.astype(np.float64)
                # set values on arr to np.inf based on bool
                next_flooded[mask] = reach

                # set new head value to something else and get the heuristics
                next_flooded[next_y, next_x] = seed_point
                # zero out all the bodies and heads
                next_flooded[(next_flooded==my_body_val)|(next_flooded==
                                body_val)|(next_flooded==same_head_val)|
                             (next_flooded==small_head_val)|(next_flooded
                                ==big_head_val)]=0
                #print(f'nextflood,reach\n{next_flooded}')
                # obstacles in way so l1 distance won't work
                '''
                # todo: timeout so just check locality
                # within 2 vertically and horiz and diag
                low_y, low_x, _,_= set_low_y_low_x(next_y-1, next_x-1,
                                                   snakes_grid)
                _, _, high_y, high_x = set_low_y_low_x(next_y+1, next_x+1,
                                                       snakes_grid)
                '''
                '''
                check_dist_arr = np.argwhere(next_flooded==reach)
                most_steps = 0
                if check_dist_arr.shape[0] > 0:
                    for m in range(check_dist_arr.shape[0]):
                        check_y, check_x = check_dist_arr[m,:]
                        num_steps = search(check_y, check_x,
                                next_y, next_x, snakes_grid,
                                           count_path=True)
                        next_flooded[check_y, check_x]=num_steps
                        if num_steps > most_steps:
                            most_steps = num_steps
                '''
                #print(f'flood before\n{next_flooded}')
                # zero out the new head for points
                # todo: not seeing thin paths
                points_grid = np.copy(next_flooded)
                points_grid[next_y, next_x] = 0
                reach_pts = np.count_nonzero(points_grid)
                '''
                reach_pts+=np.sum(np.argwhere(points_grid[low_y: high_y, x]>0))
                reach_pts += np.sum(np.argwhere(points_grid[y,
                                                low_x: high_x] > 0))
                '''

                # append the next index, reach_pts and grid to list
                #print(f'nexty,nextx {next_y},{next_x}\n '
                      #f'flooded\n{next_flooded}\n'
                      #f'reachpts {reach_pts}')

                # move to get to next_y, next_x
                move = (np.array([next_y, next_x]) - np.array([y,x])).tolist()
                move_idx = delta.index(move)
                my_move = delta_name[move_idx]
                print(f'reachpts {reach_pts}'
                      f'move{move}, moveidx {move_idx} mymove {my_move}')
                h_list.append([reach_pts, my_move, next_flooded])
                reach_pts=0
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
    elif y+2 > snakes_grid.shape[0]:
        high_y = snakes_grid.shape[0]
    if x-1 < 0:
        low_x = 0
    elif x+2 > snakes_grid.shape[1]:
        high_x = snakes_grid.shape[1]

    return low_y, low_x, high_y, high_x

def check_proximity(snakes, me, snakes_grid, y, x):
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
    todo: this is current head pos only, could use for each move in h_list
    '''
    next_bighead_val = 9
    next_samehead_val = 11
    next_smhead_val = 8
    bighead_count=0
    sm_head_count=0
    diag_bighead_count= 0
    diag_small_count = 0
    diag_next_big=0
    diag_next_small =0
    #next_eqhead_val = 23
    #next_predator_val = 191
    #next_prey_val = 2

    tactic = ''
    # iterate h_list get scores and find move with lowest next_bighead_count or highest_next_small_head_count
    scores_per_move = []
    next_pos = []
    # check up to 2 moves away
    depth = 1

    # check 2 away, not just one so add 1 to head y and x
    low_y, low_x, _,_ = set_low_y_low_x(y-depth,x-depth, snakes_grid)
    _, _, high_y, high_x = set_low_y_low_x(y+depth, x+depth, snakes_grid)

    # check vertical 2 away
    bighead_count += (np.argwhere(snakes_grid[low_y:high_y, x]== next_bighead_val)).shape[0]
    bighead_count += (np.argwhere(snakes_grid[low_y:high_y, x]==big_head_val)).shape[0]
    # horizontal 2 away
    bighead_count+=  (np.argwhere(snakes_grid[y, low_x: high_x] == next_bighead_val)).shape[0]
    bighead_count+= (np.argwhere(snakes_grid[y, low_x: high_x] == big_head_val)).shape[0]
    # vertical
    sm_head_count += (np.argwhere(snakes_grid[low_y:high_y, x] == next_smhead_val)).shape[0]
    sm_head_count += (np.argwhere(snakes_grid[low_y:high_y, x] == small_head_val)).shape[0]
    # horizontal
    sm_head_count += (np.argwhere(snakes_grid[y, low_x: high_x] == next_smhead_val)).shape[0]
    sm_head_count += (np.argwhere(snakes_grid[y, low_x: high_x] == small_head_val)).shape[0]

    # check diagonals
    for i in range(len(diag_delta)):
        next_y = diag_delta[i][0]
        next_x = diag_delta[i][1]
        if check_in_bounds(next_y, next_x, snakes_grid):
            if snakes_grid[next_y, next_x] == next_bighead_val:
                bighead_count+=1
            elif snakes_grid[next_y, next_x] == big_head_val:
                bighead_count+=1
            elif snakes_grid[next_y, next_x] == next_smhead_val:
                sm_head_count+=1
            elif snakes_grid[next_y, next_x] == small_head_val:
                sm_head_count+=1

    if bighead_count>0:
        tactic='evade'
    elif sm_head_count>0:
        tactic='attack'
    elif len(snakes)==2:
        for j in range(len(snakes)):
            curr_snake = snakes[j]
            if curr_snake['id']!= me['id']:
                if len(curr_snake['body']) < len(me['body']):
                    tactic='attack'
    else:
        tactic='neutral'

    return tactic


@bottle.post("/move")
def move():
    """
    Called when the Battlesnake Engine needs to know your next my_move.
    The data parameter will contain information about the board.
    Your response must include your my_move of up, down, left, or right.
    """
    #todo: tune healthfood ratio
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
    
    # reach_pts < 4 moves, move, next_flooded from
    # highest to lowest reach points
    # next pos idx ranked from most accessible spots to least
    heuristics_list = board_control_heuristics(turn, solo_grids, snakes_grid,
                                               bodies_grid, me)
    h_moves = []
    for j in range(len(heuristics_list)):
        curr_move = heuristics_list[j][1]
        h_moves.append(curr_move)


    # get ranked moves for locality
    # returns list of lists ([total_score, next_pos])
    tactic = check_proximity(snakes,me, snakes_grid, my_head_y, my_head_x)
    res_move = ''
    if len(h_moves)>0:
        res_move = h_moves[0]
    if tactic=='evade':
        if len(h_moves)>0:
            res_move = h_moves[0]
            for i in range(len(h_moves)):
                my_move = h_moves[i]
                # check path to tail
                move_num = delta_name.index(my_move)
                found_free = check_path_to_tail(my_head_y, my_head_x,
                                                move_num, snakes_grid,
                                                snake_tails)
                if found_free:
                    which_move = 'evade'
                    path_found=True
                    break
                else:
                    my_move = res_move
                    path_found=False

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
    health_food_ratio = my_health - nearest_food_dist
    # must be between 0 and 1 or die unless random spawn within myhealth spots
    min_ratio = 20
    if health_food_ratio < min_ratio:
        tactic='get food'

    attack = False
    # if me_longest, chase 8s
    if tactic=='attack' and not path_found:
        next_small_targets = np.argwhere(snakes_grid%8==0)
        for i in range(next_small_targets.shape[0]):
            target_y, target_x = next_small_targets[i,:]
            # todo: use move_idxs and check against heuristics list reach score, pick move that matches, otherwise favor reach score and
            # find path to tail
            move_num, my_move, path_found = search(target_y, target_x,
                                                   my_head_y, my_head_x,
                                                        snakes_grid)
            if path_found:
                found_free = check_path_to_tail(my_head_y, my_head_x,
                                            move_num, snakes_grid,
                                            snake_tails)
                if found_free:
                    which_move = 'attack'
                    break
            else:
                which_move=' failed attack'
                path_found=False


    # get food
    eating = False
    count = 0
    get_it = False
    # if get food or attack failed
    if ((tactic=='neutral' or tactic=='get food') and not path_found) \
            or my_move =='fuzz':
        # print('food')
        while not eating and count < len(food_arr):
            curr_food = food_arr[count]
            food_dist = curr_food[0]
            food_y = curr_food[1]
            food_x = curr_food[2]
            food_count += 1
            if len(snakes) > 1:
                # must be closerish than all snakes
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
                        break
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
    if not path_found and my_health!=100:
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

    if not path_found or my_move =='fuzz':
        if len(h_moves)>0:
            res_move = h_moves[0]
            for i in range(len(h_moves)):
                my_move = h_moves[i]
                # check path to tail
                move_num = delta_name.index(my_move)
                found_free = check_path_to_tail(my_head_y, my_head_x,
                                                move_num, snakes_grid,
                                                snake_tails)
                if found_free:
                    path_found=True
                    which_move = 'last'
                    break
                else:
                    my_move = res_move
                    which_move = 'res'
                #path_found=True

    shout = "get in my belly!"

    response = {"move": my_move, "shout": shout}
    end = timer()
    print(f'pathfound {path_found}')
    print(f'\n\nturn: {turn}\ntime: {end-start}\nmy_move: {my_move}\n '
    f'which_move: {which_move}\n\n')
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

'''
            low_y, low_x, high_y, high_x = set_low_y_low_x(y,x, snakes_grid)
            print(f'prev y,x {y},{x}\n low: yx {low_y},{low_x}\n'
                  f'high yx {high_y},{high_x}')
            # get all next poses of small and equal in bounds and free spots
            # todo: subarray indices returned

            next_poses = np.argwhere((snakes_grid[low_y:high_y, x]
                                     ==0) |
                                     (snakes_grid[low_y:high_y, x]
                                     == next_smhead_val) |
                                     (snakes_grid[low_y:high_y, x]
                                      == next_samehead_val)|
                                     (snakes_grid[low_y:high_y, x]
                                      == next_bighead_val) |
                                     (snakes_grid[y, low_x:high_x]
                                      == 0) |
                                     (snakes_grid[y, low_x:high_x]
                                      == next_smhead_val) |
                                     (snakes_grid[y, low_x:high_x]
                                      == next_samehead_val) |
                                     (snakes_grid[y, low_x:high_x]
                                      == next_bighead_val))
            print(f'next poses {next_poses}')

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
            '''