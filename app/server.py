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
same_head_val = 1
my_head_val = 1
big_head_val = 1
body_val = 1
my_body_val = 1
next_smhead_val = 2
next_samehead_val = 3
next_bighead_val = 9

next_heads = [next_smhead_val, next_samehead_val, next_bighead_val]


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

    response = {"color": "#fc0313", "headType": "fang", "tailType": "bolt"}
    return HTTPResponse(
        status=200,
        headers={"Content-Type": "application/json"},
        body=json.dumps(response),
    )


def search(goal_y, goal_x, my_head_y, my_head_x, snakes_grid,
           check_grid, check_path=False):
    '''
    small_head_val = 1
    same_head_val = 1
    my_head_val = 1
    big_head_val = 1
    body_val = 1
    my_body_val = 1
    #todo: for hunting need to label next_smhead_val =
    next_smhead_val = 2
    next_samehead_val = 5
    next_bighead_val = 9
    '''
    found_path = False
    my_move = ''
    move_num = 0

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
    y,x = 0,0
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
                                snakes_grid[new_y, new_x] != body_val:

                            g2 = g + cost
                            f2 = g2 + heuristic_map[new_y, new_x]
                            open_arr.append([f2, g2, new_y, new_x])
                            closed[new_y, new_x] = 1

    # found goal or resigned

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
            small_val = expand[low_y, low_x]
            start_y = low_y
            start_x = low_x

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
        my_move = 'fail'
        found = False

    ##print('return')
    return move_num, my_move, found


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
        if 0 <= next_head_y < snakes_grid.shape[0] \
                and 0 <= next_head_x < snakes_grid.shape[1]:
            if snakes_grid[next_head_y, next_head_x] != body_val:
                new_grid[next_head_y, next_head_x] += next_head_val

    return new_grid


def fill_snakes_grid(snakes, width, height, my_body_len, my_id):
    '''
    small_head_val = 1
    same_head_val = 1
    my_head_val = 1
    big_head_val = 1
    body_val = 1
    my_body_val = 1
    #todo: for hunting need to label next_smhead_val =
    next_smhead_val = 2
    next_samehead_val = 5
    next_bighead_val = 9
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
        # fill grid with bodies
        for k in range(len(curr_snake['body'])):
            # heads of opp snakes
            if k == 0:
                head_y = curr_snake['body'][k]['y']
                head_x = curr_snake['body'][k]['x']
                # if smaller
                if len(curr_snake['body']) < my_body_len and not my_snake:
                    snakes_grid[head_y, head_x] += small_head_val
                    # append to heads list
                    snake_heads.append([next_smhead_val, head_y, head_x])

                # if it's the heads of bigger or equal snakes
                elif len(curr_snake['body']) > my_body_len and not my_snake:
                    snakes_grid[head_y, head_x] += big_head_val
                    # append to heads list
                    snake_heads.append([next_bighead_val, head_y, head_x])

                # todo: equal size
                elif len(curr_snake['body']) == my_body_len and not my_snake:
                    snakes_grid[head_y, head_x] += same_head_val
                    # todo: append to heads list or not?
                    snake_heads.append([next_samehead_val, head_y, head_x])

                # fill solo grid for crash check
                elif len(curr_snake['body']) == my_body_len and my_snake:
                    solo_grid[head_y, head_x] += my_head_val
                    snakes_grid[head_y, head_x] += my_head_val
            # all snakes body and my head and body except tail
            elif 0 < k < (len(curr_snake['body']) - 1):
                body_y = curr_snake['body'][k]['y']
                body_x = curr_snake['body'][k]['x']
                #
                if not my_snake:
                    snakes_grid[body_y, body_x] += body_val
                # fill solo grid
                elif my_snake:
                    snakes_grid[body_y, body_x] += my_body_val
                    solo_grid[body_y, body_x] += body_val
            # tails
            elif k == (len(curr_snake['body']) - 1):
                body_y = curr_snake['body'][k]['y']
                body_x = curr_snake['body'][k]['x']
                solo_grid[body_y, body_x] += my_body_val
                snake_tails.append([body_y, body_x])
                if curr_snake['health'] == 100:
                    snakes_grid[body_y, body_x] += body_val

    # mark next heads after bodies filled
    for i in range(len(snake_heads)):
        curr_head = snake_heads[i]
        curr_nextval = curr_head[0]
        curr_y = curr_head[1]
        curr_x = curr_head[2]
        snakes_grid = mark_next_heads(curr_y, curr_x, snakes_grid, curr_nextval)

    return snakes_grid, solo_grid, snake_heads, snake_tails

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
    same_head_val = 1
    my_head_val = 1
    big_head_val = 1
    body_val = 1
    my_body_val = 1
    #todo: for hunting need to label next_smhead_val =
    next_smhead_val = 2
    next_samehead_val = 3
    next_bighead_val = 9
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
            # if spot is possible next for equal or bigger
            if next_samehead_val <= snakes_grid[i, j] <= next_bighead_val:
                heuristic_map[i, j] += snakes_grid[i, j]
            # if space is next for 3 or multiple big snakes or combo
            elif next_bighead_val < snakes_grid[i, j]:
                heuristic_map[i, j] += (snakes_grid[i, j] * 2)
            # if space is next for many smaller, then it's good
            elif snakes_grid[i, j] % 2 == 0:
                heuristic_map[i, j] -= snakes_grid[i, j]

    return heuristic_map


def find_free_spaces(snakes_grid, head_y, head_x):
    '''
    small_head_val = 1
    same_head_val = 1
    my_head_val = 1
    big_head_val = 1
    body_val = 1
    my_body_val = 1
    #todo: for hunting need to label next_smhead_val =
    next_smhead_val = 2
    next_samehead_val = 3
    next_bighead_val = 9
    Free spaces ranked according to grid value and num empty spaces of at least
    my len
    '''
    free_spaces = np.argwhere(snakes_grid < body_val)
    free_spaces_arr = []
    biggest_dist = 0
    furthest = []
    for i in range(free_spaces.shape[0]):
        curr_y, curr_x = free_spaces[i, 0], free_spaces[i, 1]
        dist_to_free = heuristic([head_y, head_x], [curr_y, curr_x])

        free_spaces_arr.append([dist_to_free, curr_y, curr_x])
    # nearest to front
    free_arr = sorted(free_spaces_arr, key=lambda x: x[0])
    return free_arr


def check_snakes_info(snakes, snake_heads, head_y, head_x, me):
    '''
    Get info on snakes and return array and the num snakes nearby?
    '''
    my_id = me['id']
    my_len = len(me['body'])
    snake_info_arr = []
    for j in range(len(snakes)):
        curr_snake = snakes[j]
        id = curr_snake['id']
        if id != my_id:
            health = curr_snake['health']
            body_len = len(curr_snake['body'])
            snakehead_y = curr_snake['body'][0]['y']
            snakehead_x = curr_snake['body'][0]['x']
            tail_y = curr_snake['body'][-1]['y']
            tail_x = curr_snake['body'][-1]['x']
            dist_to_head = heuristic([head_y, head_x], [snakehead_y, snakehead_x])
            snake_info_arr.append([id, health, body_len, [snakehead_y,
                                                          snakehead_x], [tail_y, tail_x],
                                   dist_to_head])

    return snake_info_arr

def check_path_to_tail(head_y, head_x, snakes, move_num, snakes_grid,
                       check_grid,snake_tails):
    found_path = False
    danger_tail = 0
    #todo: change this
    num_free = 0
    new_head_y = head_y + delta[move_num][0]
    new_head_x = head_x + delta[move_num][1]
    curr_tail_y = 0
    curr_tail_x = 0

    if 0 <= new_head_y < snakes_grid.shape[0] and \
            0 <= new_head_x < snakes_grid.shape[1]:

        # check that we can reach a tail and health is not 100
        for i in range(len(snakes)):
            curr_snake = snakes[i]
            curr_tail_y = snakes[i]['body'][-1]['y']
            curr_tail_x = snakes[i]['body'][-1]['x']
            if curr_snake['health'] == 100:
                # check four connected from new postion to find min 2
                for j in range(len(delta)):
                    check_y = curr_tail_y + delta[j][0]
                    check_x = curr_tail_x + delta[j][1]
                    if 0 <= check_y < snakes_grid.shape[0] and \
                            0 <= check_x < snakes_grid.shape[1]:
                        if snakes_grid[check_y, check_x]!=body_val:
                            found_path = search(curr_tail_y, curr_tail_x,
                                                new_head_y,
                                                new_head_x, snakes_grid,
                                                check_grid,
                                                check_path=True)
                            if found_path:
                                break
                            else:
                                found_path = False
                                # print('check tail fail')
            else:

                found_path = search(curr_tail_y, curr_tail_x, new_head_y,
                                    new_head_x, snakes_grid,
                                    check_grid,
                                    check_path=True)
                if found_path:
                    break
                else:
                    found_path = False
                    # print('check tail fail')

    return found_path




def move_attack(my_head_y, my_head_x,snakes, snakes_grid, check_grid,
                            snake_tails, targets):
    # print('attack')
    target_arr = []
    move_num = 0
    my_move = ''
    path_found = False
    # calculate distances and sort
    # snake type given by next vals
    for j in range(targets.shape[0]):
        curr_target = targets[j]
        target_y = curr_target[0]
        target_x = curr_target[1]
        dist = heuristic([target_y, target_x], [my_head_y, my_head_x])
        target_arr.append([dist, target_y, target_x])
    targets = sorted(target_arr, key=lambda x: x[0])
    # iterate targets
    for i in range(len(targets)):
        victim = targets[i]
        move_num, my_move, path_found = \
            search(victim[1], victim[2], my_head_y, my_head_x, snakes_grid,
                   check_grid)

        if path_found:
            found_free = check_path_to_tail(my_head_y, my_head_x,snakes,
                                            move_num, snakes_grid,
                                            check_grid,
                                            snake_tails)
            if found_free:
                break
            else:
                path_found = False


    return move_num, my_move, path_found


def find_food(food_arr, my_head_y, my_head_x, snakes, snakes_grid, check_grid,
              snake_heads, snake_tails):
    '''
    small_head_val = 1
    same_head_val = 1
    my_head_val = 1
    big_head_val = 1
    body_val = 1
    my_body_val = 1
    #todo: for hunting need to label next_smhead_val =
    next_smhead_val = 2
    next_samehead_val = 3
    next_bighead_val = 9
    '''

    count = 0
    move_num = 0
    my_move = ''
    path_found = False
    eating = False
    next_food = False
    get_it = False

    while not eating and count < len(food_arr):
        curr_food = food_arr[count]
        food_dist = curr_food[0]
        food_y = curr_food[1]
        food_x = curr_food[2]
        # check four connected for next_bighead val
        for j in range(len(delta)):
            check_y = food_y + delta[j][0]
            check_x = food_x + delta[j][1]
            if 0 <= check_y < snakes_grid.shape[0] and \
                    0 <= check_x < snakes_grid.shape[1]:
                if snakes_grid[check_y, check_x] >= next_samehead_val:
                    next_food = True
                    break

        # more than one snake, check distances to food
        if len(snakes) is not 1 and not next_food:
            for i in range(len(snake_heads)):
                curr_head = snake_heads[i]
                head_type = curr_head[0]
                snakehead_y = curr_head[1]
                snakehead_x = curr_head[2]
                # either equal or bigger
                if head_type > next_smhead_val:
                    opp_dist = heuristic([snakehead_y, snakehead_x],
                                         [food_y, food_x])
                    if food_dist < opp_dist:
                        get_it = True
                    else:
                        get_it = False
                        break
        else:
            get_it = True

        if get_it and not next_food:
            move_num, my_move, path_found = \
                search(food_y, food_x, my_head_y, my_head_x,
                       snakes_grid, check_grid)
            if path_found:
                found_free = check_path_to_tail(my_head_y, my_head_x,
                                                snakes,
                                                move_num, snakes_grid,
                                                check_grid,
                                                snake_tails)

                if found_free:
                    which_move = 'get food'
                    eating = True
                else:
                    path_found = False
                    get_it = False
                    next_food = False
            else:
                path_found = False
                get_it = False
                next_food = False

        count += 1

    return move_num, my_move, path_found


def chase_opp_tails(snakes, snakes_info, tail_y, tail_x, my_head_y, my_head_x, snakes_grid,
                    check_grid, snake_tails):
    '''

    # id, health, body_len, [snakehead_y,snakehead_x], [tail_y, tail_x],
    #                           dist_to_head]
    '''

    move_num = 0
    my_move = ''
    path_found = False

    for i in range(len(snakes_info)):
        curr_snake = snakes_info[i]
        id = curr_snake[0]
        health = curr_snake[1]
        body_len = curr_snake[2]
        head_y, head_x = curr_snake[3][0], curr_snake[3][1]
        tail_y, tail_x = curr_snake[4][0], curr_snake[4][1]
        dist_to_head = curr_snake[5]
        if health == 100:
            for j in range(len(delta)):
                adj_y = tail_y + delta[i][0]
                adj_x = tail_x + delta[i][1]
                if 0 <= adj_y < snakes_grid.shape[0] and \
                        0 <= adj_x < snakes_grid.shape[1]:
                    if snakes_grid[adj_y, adj_x] != 1:
                        move_num, my_move, path_found = \
                            search(tail_y, tail_x,
                                   my_head_y, my_head_x, snakes_grid,
                                   check_grid)
                    if path_found:
                        '''
                        found_free = check_path_to_free(my_head_y, my_head_x,
                                                        move_num, snakes_grid, free_spaces_arr)
                        '''
                        found_free = check_path_to_tail(
                            my_head_y, my_head_x,
                            snakes, move_num, snakes_grid,
                            check_grid, snake_tails)

                        if found_free:
                            break
                        else:
                            path_found = False

    return move_num, my_move, path_found


def chase_my_tail(my_health, my_tail_y, my_tail_x, my_head_y, my_head_x, snakes,
                  snakes_grid, check_grid, snake_tails):
    move_num = 0
    my_move = ''
    path_found = False
    # going to tail will collide due to extension
    if my_health == 100:
        for i in range(len(delta)):
            adj_y = my_tail_y + delta[i][0]
            adj_x = my_tail_x + delta[i][1]
            if 0 <= adj_y < snakes_grid.shape[0] and \
                    0 <= adj_x < snakes_grid.shape[1]:
                if snakes_grid[adj_y, adj_x] != 1:
                    move_num, my_move, path_found = \
                        search(adj_y, adj_x,
                               my_head_y, my_head_x, snakes_grid,
                               check_grid)
                if path_found:
                    '''
                    found_free = check_path_to_free(my_head_y, my_head_x,
                                        move_num, snakes_grid, free_spaces_arr)
                    '''
                    found_free = check_path_to_tail(my_head_y, my_head_x,
                                                    snakes, move_num,
                                                    snakes_grid,
                                                    check_grid, snake_tails)
                    if found_free:
                        which_move = 'my tail'
                    else:
                        path_found = False
                else:
                    path_found = False

    else:
        move_num, my_move, path_found = \
            search(my_tail_y, my_tail_x,
                   my_head_y, my_head_x, snakes_grid, check_grid)
        if path_found:
            '''
            found_free = check_path_to_free(my_head_y, my_head_x,
                                move_num, snakes_grid, free_spaces_arr)
            '''
            found_free = check_path_to_tail(my_head_y, my_head_x,
                                            snakes,
                                            move_num, snakes_grid,
                                            check_grid, snake_tails)
            if found_free:
                which_move = 'my tail'
            else:
                path_found = False

    return move_num, my_move, path_found

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
    move_num = 0

    # flags
    path_found = False
    ready = False
    risky = False
    riskier = False

    # make state info
    snakes_grid, solo_grid, snake_heads, snake_tails = \
        fill_snakes_grid(snakes, width, height, my_body_len, my_id)


    if my_health==100:
        #print(f'100 tail val {snakes_grid[my_tail_y, my_tail_x]}')
        snakes_grid[my_tail_y, my_tail_x]=body_val
    # check_grid
    check_grid = np.copy(snakes_grid)

    # id, health, body_len, [snakehead_y,snakehead_x], [tail_y, tail_x],
    #                           dist_to_head]
    snakes_info = check_snakes_info(snakes, snake_heads, my_head_y, my_head_x
                                   , me)

    # find free spaces and dists
    # dist, freey, freex
    #free_spaces_arr = find_free_spaces(snakes_grid, my_head_y, my_head_x)

    attack = False
    attack_num = 77
    attack_dist = width
    # attack
    attack_grid = np.copy(snakes_grid)

    if len(snakes)>1 and my_health > (width * 2):
        for i in range(len(snakes_info)):
            curr_snake = snakes_info[i]
            curr_len = curr_snake[2]
            curr_y, curr_x = curr_snake[3][0], curr_snake[3][1]
            dist = curr_snake[5]
            if len(snakes) > 2:
                if curr_len < my_body_len and dist <= attack_dist:
                    attack_grid[curr_y, curr_x] = attack_num
                    attack = True
            elif len(snakes)==2:
                attack_grid[curr_y, curr_x] = attack_num
                attack=True


    # leave walls
    leave_walls = False
    '''
    if ((my_head_x == 0 or my_head_x == (snakes_grid.shape[1] - 1)) or \
        (my_head_y == 0 or my_head_y == (snakes_grid.shape[0] - 1))) and \
            my_health > 10:
        # print('walls')
        my_move, path_found = get_away_walls(my_head_y, my_head_x,snakes,
                                             snakes_grid, check_grid, 
                                             snake_tails)
        if path_found:
            leave_walls = True
        
    '''
    # if attack
    if attack and not leave_walls and my_health:
        targets = np.argwhere(attack_grid == attack_num)
        move_num, my_move, path_found = move_attack(my_head_y, my_head_x,
                                                    snakes,
                                                    snakes_grid, check_grid,
                                                    snake_tails, targets)
        if not path_found:
            attack = False

    # list of dicts of food locations
    food = data['board']['food']
    # list in order of nearest to furthest food tuples (dist, y,x)
    food_arr = []
    # if there is food
    if len(food) > 0:
        food_arr = fill_food_arr(food, my_head_y, my_head_x)
    # there is a food so A star for route to food using snake grid for g
    food_count = 0

    found_path = False
    # get food
    eating = False
    count = 0
    get_it = False
    next_food=False
    if not path_found and not leave_walls and not attack:
        # print('food')
        move_num, my_move, path_found = find_food(food_arr, my_head_y,
                                    my_head_x, snakes, snakes_grid,
                                    check_grid,
                                    snake_heads, snake_tails)
        if path_found:
            which_move = 'food'


    # chase my tail
    if not path_found and not leave_walls and not attack:
        move_num, my_move, path_found = chase_my_tail(my_health,
                        my_tail_y, my_tail_x, my_head_y, my_head_x, snakes,
                          snakes_grid, check_grid, snake_tails)
        if path_found:
            which_move = 'chase my tail'


    # chase other snakes' tails
    if not path_found and not leave_walls and not attack:
        # print('other tails')
        move_num, my_move, path_found = chase_opp_tails(snakes, snakes_info,
                                my_tail_y, my_tail_x,
                                my_head_y, my_head_x, snakes_grid,
                                check_grid, snake_tails)
        if path_found:
            which_move = 'opp tails'

    # go to furthest free space
    if not path_found and not leave_walls and not attack:
        free_spaces = find_free_spaces(snakes_grid, my_head_y, my_head_x)
        # furthest front
        safe_arr = free_spaces[::-1]
        #print(f'safearr {safe_arr}')
        for i in range(len(safe_arr)):
            curr_safe = safe_arr[i]
            safe_dist = curr_safe[0]
            safe_y = curr_safe[1]
            safe_x = curr_safe[2]

            move_num, my_move, path_found = search(safe_y, safe_x, my_head_y,
                                my_head_x, snakes_grid, check_grid,
                                check_path=False)
            if path_found:
                found_free = check_path_to_tail(safe_y, safe_x,
                                                snakes,
                                                move_num, snakes_grid,
                                                check_grid, snake_tails)
                if found_free:
                    which_move = 'meadows'
                    break
                else:
                    path_found = False


    shout = "get in my belly!"

    response = {"move": my_move, "shout": shout}
    end = timer()
    print(f'\n\nturn: {turn}\ntime: {end-start}\nmy_move: {my_move}\n '
    f'which_move: {which_move}\n\n')
    ##print(f'snakes_grid\n {snakes_grid}\nsolo_grid\n {solo_grid}\n')
    return HTTPResponse(
        status=200,
        headers={"Content-Type": "application/json"},
        body=json.dumps(response),
    )


"""
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
"""
def get_away_walls(my_head_y, my_head_x,snakes,
                        snakes_grid, check_grid, snake_tails):
    path_found = False

    move_num = 0
    my_move = ''
    count = 0
    found_free = False

    free_spaces = find_free_spaces(snakes_grid, my_head_y, my_head_x)
    # furthest front
    safe_arr = free_spaces[::-1]

    for i in range(len(safe_arr)):
        curr_safe = safe_arr[i]
        safe_dist = curr_safe[0]
        safe_y = curr_safe[1]
        safe_x = curr_safe[2]

        move_num, my_move, path_found = search(safe_y, safe_x, my_head_y,
                                               my_head_x, snakes_grid, check_grid,
                                               check_path=False)

        if path_found:
            found_free = check_path_to_tail(my_head_y, my_head_x, snakes,
                                            move_num, snakes_grid,
                                            check_grid,
                                            snake_tails)
            if found_free:
                break
            else:
                path_found = False

    return move_num, my_move, path_found

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