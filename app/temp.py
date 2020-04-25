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

delta_name = ['up', 'left', 'down', 'right']

cost = 1

# vals for smaller heads, equal or big, all bodies and next heads
body_val = 1
head_val = 2
next_bighead_val = 10
next_samehead_val = 3

curr_bodies = [body_val, head_val]
next_heads = [next_samehead_val, next_bighead_val]


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
    body_val = 1
    head_val = 2
    next_bighead_val = 10
    next_samehead_val = 3
    '''
    found_path = False
    my_move = ''
    move_num = 0

    if check_path:
        snakes_grid[snakes_grid == next_samehead_val] = 0
        snakes_grid[snakes_grid == next_bighead_val] = 0

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
            count += 1

            if y == goal_y and x == goal_x:
                found = True
                expand[y, x] = count
            else:
                for i in range(len(delta)):
                    new_y = y + delta[i][0]
                    new_x = x + delta[i][1]

                    # if in-bounds
                    if check_in_bounds(new_y, new_x, snakes_grid):
                        # if unvisited and traversible (> head_val, body_val)
                        if closed[new_y, new_x] == 0 and \
                                (snakes_grid[new_y, new_x] == 0 or
                                 snakes_grid[new_y, new_x] not in curr_bodies):
                            g2 = g + cost
                            f2 = g2 + heuristic_map[new_y, new_x]
                            open_arr.append([f2, g2, new_y, new_x])
                            closed[new_y, new_x] = 1

    # found goal or resigned
    if found and not check_path:
        return expand, found
    elif check_path:
        return found
    else:
        return 'fail', found


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
        if check_in_bounds(next_head_y, next_head_x, snakes_grid):
            # if not head nor body
            if snakes_grid[next_head_y, next_head_x] != body_val \
                    and snakes_grid[next_head_y, next_head_x] != head_val:
                new_grid[next_head_y, next_head_x] += next_head_val

    return new_grid


def check_in_bounds(new_y, new_x, snakes_grid):
    if 0 <= new_y < snakes_grid.shape[0] and \
            0 <= new_x < snakes_grid.shape[1]:
        return True
    return False


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
                    snakes_grid[head_y, head_x] = head_val
                    solo_grid[head_y, head_x] = head_val

                # if it's the heads of bigger or equal snakes
                elif len(curr_snake['body']) > my_body_len and not my_snake:
                    snakes_grid[head_y, head_x] = head_val
                    solo_grid[head_y, head_x] = head_val
                    # append to heads list
                    snake_heads.append([next_bighead_val, head_y, head_x])

                # todo: equal size
                elif len(curr_snake['body']) == my_body_len and not my_snake:
                    snakes_grid[head_y, head_x] = head_val
                    solo_grid[head_y, head_x] = head_val
                    # todo: append to heads list or not?
                    snake_heads.append([next_samehead_val, head_y, head_x])

                # fill solo grid for crash check
                elif len(curr_snake['body']) == my_body_len and my_snake:
                    solo_grid[head_y, head_x] = head_val
                    snakes_grid[head_y, head_x] = head_val
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

    return snakes_grid, bodies_grid, solo_grids, snake_tails


def check_path_to_tail(head_y, head_x, move_num, snakes_grid, check_grid,
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
                    if check_in_bounds(alt_y, alt_x, snake_tails):
                        if snakes_grid[alt_y, alt_x] not in curr_bodies:
                            found_path = search(alt_y, alt_x, new_head_y,
                                                new_head_x, snakes_grid,
                                                check_grid,
                                                check_path=True)
            else:
                tail_y = snake_tails[q][1]
                tail_x = snake_tails[q][2]
                snakes_grid[tail_y, tail_x] = 0
                found_path = search(tail_y, tail_x, new_head_y,
                                    new_head_x, snakes_grid, check_grid,
                                    check_path=True)

            if found_path:
                break
            else:
                found_path = False
                # print('check tail fail')
    return found_path


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


def board_control_heuristics(solo_grids, snakes_grid, bodies_grid):
    '''
    find the best move for each snake based on total num of reachable squares
    keep the best grid that shows this next pos
    todo: currently ownly moving my head to possible next pos's
    '''
    reach = np.inf
    seed_point = 1000
    dist_grids = []
    highest = -np.inf
    best_grid = np.zeros(snakes_grid.shape, dtype=np.int)
    #iterate each snakes' solo grids
    for i in range(solo_grids):
        snake_len = solo_grids[i][0]
        snake_id = solo_grids[i][1]
        solo_grid = solo_grids[i][2]
        # current head
        y, x = np.argwhere(solo_grid == head_val)[0]
        # find next possible head pos
        low_y, low_x = y-1, x-1
        # set at shape for slicing
        high_y, high_x = y+2, x+2
        # if in bounds not boundary
        if y - 1 < 0:
            low_y = 0
        # if at shape that is too high
        elif y >= solo_grid.shape[0]-1:
            high_y = solo_grid.shape[0]-1
        if x <= 0:
            low_x = 0
        elif x >= solo_grid.shape[1]-1:
            high_x = solo_grid.shape[1]-1
        # get all next poses in bounds and free:
        next_poses = np.argwhere(bodies_grid[low_y:high_y, low_x:high_x] \
                                 not in curr_bodies)
        for j in range(next_poses.shape[0]):
            next_grid = np.copy(bodies_grid)
            next_y, next_x = next_poses[j, :]
            # old head is body
            next_grid[y, x] = body_val
            # new head set as 0 for flood fill
            next_grid[next_y, next_x] = 0
            # find accessible points
            # mask is bool array
            mask = flood(next_grid, (next_y, next_x), connectivity=1)
            next_flooded = np.copy(next_grid)
            # set values on arr based on bool
            next_flooded[mask] = reach
            # set new head value to seed point to count steps from it
            next_flooded[next_y, next_x] = seed_point
            # count steps from seed point (next head)
            next_flooded[next_flooded == reach] = \
                np.abs(np.argwhere(next_flooded == reach) -
                       np.argwhere(next_flooded == seed_point)).sum(1)
            # todo: adjust scores so nearer nodes are worth more
            # only keep max of the grids
            if next_grid.sum() > highest:
                highest = next_grid.sum()
                best_grid = next_grid

    # sort from shortest to longest snake
    dist_grids_sorted = sorted(dist_grids, key=lambda x: x[0])
    return dist_grids


def adjust_boards(dist_grids, snakes_grid, me):
    '''
    dist_grids.append([snake_len, snake_id, solo_grid])
    set unreachable to zero and use np.nonzero
    for each of my available moves, find total number that I can access
    leads into predicting next moves at depth=n
    '''
    # longest to front
    rev_dist_grids = dist_grids[::-1]
    for i in range(0, len(rev_dist_grids), 1):
        curr_snake = rev_dist_grids[i]
        curr_snake_len = curr_snake[0]
        curr_snake_id = curr_snake[1]
        curr_grid = curr_snake[2]
        for j in range(i + 1, len(rev_dist_grids), 1):
            next_snake = rev_dist_grids[j]
            next_snake_len = next_snake[0]
            next_snake_id = next_snake[1]
            next_grid = next_snake[2]
            # first snake longer
            if curr_snake_len > next_snake_len:
                # takes over any equal distance spots
                # curr_grid[curr_grid==next_grid]=curr_grid[curr_grid==next_grid]
                # where bigger snake is further todo: inf?
                curr_grid[curr_grid > next_grid] = 0
                # leave the ones closer as they are
            # equal length snakes
            elif curr_snake_len == next_snake_len:
                # equal dist leave them
                # further
                curr_grid[curr_grid > next_grid] = 0


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
    move_num = 0

    # flags
    path_found = False

    # make state info
    # make snakes_grid
    snakes_grid, bodies_grid, solo_grids, snake_tails = \
        fill_snakes_grid(snakes, width, height, my_body_len, my_id)

    # check_grid
    check_grid = np.copy(snakes_grid)
    for i in range(len(next_tails)):
        next_tail_y = next_tails[i][0]
        next_tail_x = next_tails[i][1]
        check_grid[next_tail_y, next_tail_x] = 0
    # todo: use this? get distances to snake heads
    # dists, snaketype, y, x
    # snake_dists = check_dist_to_snakes(snake_heads, my_head_y, my_head_x)

    # find free spaces and dists
    # dist, freey, freex
    # check path to free only considers those beyond min_dist
    # free_spaces_arr = find_free_spaces(snakes_grid, my_head_y, my_head_x)

    attack = False
    # todo: if longest, start moving towards next_smhead_val on snakes grid

    num_to_attack = 2

    # todo: on risky, could attack with more snakes left
    # attack when only one snake left
    if len(snakes) == num_to_attack:
        for i in range(len(snakes)):
            if len(snakes[i]['body']) < my_body_len:
                attack = True
            else:
                attack = False
                break

    # if me_longest, chase 8s
    if attack:
        # print('attack')
        target_arr = []
        # calculate distances and sort
        for j in range(len(snake_heads)):
            snake_type = snake_heads[j][0]
            target_y = snake_heads[j][1]
            target_x = snake_heads[j][2]
            dist = heuristic([target_y, target_x], [my_head_y, my_head_x])
            target_arr.append([dist, target_y, target_x])
        targets = sorted(target_arr, key=lambda x: x[0])
        for i in range(len(targets)):
            victim = targets[i]
            move_num, my_move, path_found = \
                search(victim[1], victim[2], my_head_y, my_head_x,
                       snakes_grid)
            if path_found and my_move != 'snakeshit':

                found_free = check_path_to_tail(my_head_y, my_head_x,
                                                move_num, snakes_grid,
                                                check_grid,
                                                snake_tails)
                if found_free:
                    break
                else:
                    path_found = False
            elif my_move == 'snakeshit':
                path_found = False

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
    if not path_found and not attack:
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
                           snakes_grid, check_grid)
                if path_found:

                    found_free = check_path_to_tail(my_head_y, my_head_x,
                                                    move_num, snakes_grid,
                                                    check_grid,
                                                    snake_tails)

                    if found_free:
                        which_move = 'get food'
                        eating = True
                    else:
                        path_found = False
                else:
                    path_found = False

            count += 1

    # shorten food_arr
    # food_arr = food_arr[food_count:]
    count = 0
    # chase my tail
    if not path_found and not leave_walls and not attack:
        # print('my tail')
        # chase tail if nothing in food_arr
        move_num, my_move, path_found = search(my_tail_y, my_tail_x,
                                               my_head_y, my_head_x, snakes_grid,
                                               check_grid)
        if path_found:
            '''
            found_free = check_path_to_free(my_head_y, my_head_x,
                                move_num, snakes_grid, free_spaces_arr)
            '''
            found_free = check_path_to_tail(my_head_y, my_head_x,
                                            move_num, snakes_grid,
                                            check_grid, snake_tails)
            if found_free:
                which_move = 'my tail'
            else:
                path_found = False
        else:
            path_found = False

    count = 0
    # chase other snakes' tails
    if not path_found and not leave_walls and not attack:
        # print('other tails')
        for q in range(len(snake_tails)):
            curr_tail = snake_tails[q]
            move_num, my_move, path_found = search(curr_tail[0], curr_tail[1],
                                                   my_head_y, my_head_x, snakes_grid,
                                                   check_grid)
            if path_found:
                '''
                found_free = check_path_to_free(my_head_y, my_head_x,
                                                move_num, snakes_grid, free_spaces_arr)
                '''
                found_free = check_path_to_tail(my_head_y, my_head_x,
                                                move_num, snakes_grid,
                                                check_grid, snake_tails)
                if found_free:
                    which_move = 'opponent tail'
                    break
                else:
                    path_found = False

            else:
                path_found = False

    # sorta random
    # todo: change 9s to 8s
    if not path_found and not leave_walls and not attack:
        # print('random')
        next_heads = [next_smhead_val, next_samehead_val, next_bighead_val]
        for t in range(len(delta)):
            next_y = my_head_y + delta[t][0]
            next_x = my_head_x + delta[t][1]
            if 0 <= next_y < snakes_grid.shape[0] and \
                    0 <= next_x < snakes_grid.shape[1]:
                if snakes_grid[next_y, next_x] == 0 or \
                        snakes_grid[next_y, next_x] in next_heads:
                    my_move = delta_name[t]
                    which_move = 'last resort'
                    # #print(f'my_move: {my_move}')
                    path_found = True
                    break

    shout = "get in my belly!"

    response = {"move": my_move, "shout": shout}
    end = timer()
    # print(f'\n\nturn: {turn}\ntime: {end-start}\nmy_move: {my_move}\n '
    # f'which_move: {which_move}\n\n')
    ##print(f'snakes_grid\n {snakes_grid}\nsolo_grid\n {solo_grid}\n')
    return HTTPResponse(
        status=200,
        headers={"Content-Type": "application/json"},
        body=json.dumps(response),
    )


def heuristic(start_node, goal_node):
    start = np.array(start_node)
    goal = np.array(goal_node)

    return np.sum(np.abs(start - goal))


def make_heuristic_map(goal, snakes_grid):
    # goal index
    goal_y = goal[0]
    goal_x = goal[1]
    h_map = np.zeros(snakes_grid.shape, dtype=np.int)
    h_map[goal_y, goal_x] = 100
    h_map[h_map == 0] = np.abs((np.argwhere(h_map != 0) -
                                np.argwhere(h_map == 100)).sum(1))

    return h_map




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