import json
import os
import random
import numpy as np
from math import ceil
import bottle
from bottle import HTTPResponse
# import time
from timeit import default_timer as timer

from util_fns import *
from grid_data_maker import *
from search import *

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
    ready = False
    risky = False
    riskier = False

    # make state info
    # make snakes_grid
    snakes_grid, solo_grid, snake_heads, snake_tails = \
        fill_snakes_grid(snakes, width, height, my_body_len, my_id)

    # check_grid
    check_grid = np.copy(snakes_grid)
    for i in range(len(next_tails)):
        next_tail_y = next_tails[i][0]
        next_tail_x = next_tails[i][1]
        check_grid[next_tail_y, next_tail_x] = 0
    # todo: use this? get distances to snake heads
    # dists, snaketype, y, x
    #snake_dists = check_dist_to_snakes(snake_heads, my_head_y, my_head_x)

    # find free spaces and dists
    # dist, freey, freex
    # check path to free only considers those beyond min_dist
    #free_spaces_arr = find_free_spaces(snakes_grid, my_head_y, my_head_x)



    attack = False
    # todo: if longest, start moving towards next_smhead_val on snakes grid

    # todo: on risky, could attack with more snakes left
    # attack when only one snake left
    if len(snakes) > 1:
        for i in range(len(snakes)):
            if len(snakes[i]['body']) < my_body_len:
                attack = True
            else:
                attack = False
                break

    max_dist_for_food = (width + height) * 2


    # if me_longest, chase 8s
    if attack:
        # where the next smallheads are on grid
        target_arr = np.argwhere(snakes_grid[snakes_grid % next_smhead_val==0])
        # calculate distances and sort
        for j in range(target_arr.shape[0]):
            target_y = target_arr[j][0]
            target_x = target_arr[j][1]
            dist = heuristic([target_y, target_x], [my_head_y, my_head_x])
            target_arr.append([dist, target_y, target_x])
            move_num, my_move, path_found = \
                search(target_y, target_x, my_head_y, my_head_x,
                       snakes_grid)
            if path_found and my_move != 'snakeshit':

                found_free = check_path_to_tail(snakes, my_head_y, my_head_x,
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
    if not path_found and not leave_walls and not attack:
        # print('food')
        while not eating and count < len(food_arr):
            curr_food = food_arr[count]
            food_dist = curr_food[0]
            food_y = curr_food[1]
            food_x = curr_food[2]
            #if food along the wall and big head or next bighead around it,
            # pass
            '''
            if food_y==0 or food_y==(snakes_grid.shape[0]-1) or food_x==0 or\
                food_x ==(snakes_grid.shape[1]-1):
            '''
            low_y, low_x, _, _ = \
                set_low_y_low_x(food_y-2, food_x-2, snakes_grid)
            _,_, high_y, high_x = \
                set_low_y_low_x(food_y + 2, food_x + 2, snakes_grid)
            if np.any((snakes_grid[low_y:high_y, low_x:high_x]==
                       next_bighead_val) | (snakes_grid[low_y:high_y,
                        low_x:high_x]==big_head_val)):
                count+=1
                continue
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

                    found_free = check_path_to_tail(snakes, my_head_y, my_head_x,
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
            found_free = check_path_to_tail(snakes, my_head_y, my_head_x,
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
                found_free = check_path_to_tail(snakes,my_head_y, my_head_x,
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
    print(f'\n\nturn: {turn}\ntime: {end-start}\nmy_move: {my_move}\n '
    f'which_move: {which_move}\n\n')
    ##print(f'snakes_grid\n {snakes_grid}\nsolo_grid\n {solo_grid}\n')
    return HTTPResponse(
        status=200,
        headers={"Content-Type": "application/json"},
        body=json.dumps(response),
    )



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