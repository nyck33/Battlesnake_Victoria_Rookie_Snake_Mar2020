import json
import os
import random
import numpy as np
from math import ceil
import bottle
from bottle import HTTPResponse
#import time
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
my_head_val=3
big_head_val = 5
body_val = 4
my_body_val = 7
next_bighead_val = 9
next_smhead_val = 8


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
    # print(f"start_data:\n{json.dumps(data, indent=2)}")

    response = {"color": "#fc0313", "headType": "fang", "tailType": "bolt"}
    return HTTPResponse(
        status=200,
        headers={"Content-Type": "application/json"},
        body=json.dumps(response),
    )


def search(goal_y, goal_x, my_head_y, my_head_x, snakes_grid, check_path=False):

    my_move = ''
    move_num=0

    # visited array
    closed = np.zeros(snakes_grid.shape, dtype=np.int)
    closed[my_head_y, my_head_x] = 1
    # expand is final map returned with numbered spots
    expand = np.full(snakes_grid.shape, -1, dtype=np.int)

    g = 0  # each step is 1
    heuristic_map = make_heuristic_map([goal_y, goal_x],
                                       snakes_grid)
    # print(f'heuristics_map\n{heuristic_map}')
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
            #if count==0:
                #print(f'first expand\ny: {y}, x: {x}\n{expand}')
            count += 1

            if y == goal_y and x == goal_x:
                found = True
                #expand[y,x] = -99
                if check_path:
                    #print(f'checkpath expand\n{expand}')
                    return found
            else:
                for i in range(len(delta)):
                    new_y = y + delta[i][0]
                    new_x = x + delta[i][1]

                    # if in-bounds
                    if 0 <= new_y < snakes_grid.shape[0] and \
                            0 <= new_x < snakes_grid.shape[1]:
                        # if unvisited and traversible (smaller snake's nexthead
                        #is traversible)

                        if closed[new_y, new_x]==0 and \
                                (snakes_grid[new_y, new_x]==0 or
                                snakes_grid[new_y, new_x]==next_smhead_val):
                            g2 = g + cost
                            f2 = g2 + heuristic_map[new_y, new_x]
                            open_arr.append([f2, g2, new_y, new_x])
                            closed[new_y, new_x] = 1

    # found goal or resigned
    if found:
        move_num = 0

        # todo: can work backwards from where expand is >0 and compare to
        # todo: start y and x and find move to get there
        found_path = False

        for i in range(len(delta)):
            next_y = my_head_y + delta[i][0]
            next_x = my_head_x + delta[i][1]
            if 0 <= next_y < expand.shape[0] and \
                    0 <= next_x < expand.shape[1]:
                if expand[next_y, next_x]>0:
                    if snakes_grid[next_y, next_x] == 0 or \
                            snakes_grid[next_y, next_x] == next_smhead_val:

                        # check four connected for a pos int
                        for j in range(len(delta)):
                            n_next_y = next_y + delta[j][0]
                            n_next_x = next_x + delta[j][1]
                            if 0 <= n_next_y < expand.shape[0] and \
                                    0 <= n_next_x < expand.shape[1]:
                                if expand[next_y, next_x] > 0:
                                    if snakes_grid[n_next_y, n_next_x]==0 or\
                                            snakes_grid[n_next_y,n_next_x]\
                                                == next_smhead_val:
                                        move_num = i
                                        my_move = delta_name[i]
                                        found_path=True
                                        break

            if found_path:
                break

    else:
        move_num = 0
        my_move = 'snakeshit'
    #print(f'expand:\n{expand}')

    return move_num, my_move, found


def fill_food_arr(food, my_head_y, my_head_x):
    # list in order of nearest to furthest food tuples (dist, y,x)
    food_arr = []
    for z in range(len(food)):
        food_dist = heuristic([my_head_y, my_head_x],
                              [food[z]['y'], food[z]['x']])
        food_arr.append([food_dist, food[z]['y'], food[z]['x']])

    food_array = sorted(food_arr, key=lambda x: x[0])
    # print(f'\n\nfood arr {food_arr}\n\n')
    return food_array

def mark_next_heads(head_y, head_x, snakes_grid,next_head_val):
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
            if new_grid[next_head_y, next_head_x]==0:
                new_grid[next_head_y, next_head_x] = next_head_val

    return new_grid

def fill_snakes_grid(snakes, width, height, my_body_len, my_id):
    '''
    small_head_val = 1
    my_head_val = 3
    big_head_val = 5
    body_val = 4
    my_body_val = 7
    next_bighead_val = 9
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
            my_snake=True
        else:
            my_snake=False
        # fill grid
        for k in range(len(curr_snake['body'])):
            # heads of opp snakes
            if k == 0:
                head_y = curr_snake['body'][k]['y']
                head_x = curr_snake['body'][k]['x']
                # if smaller
                if len(curr_snake['body']) < my_body_len and not my_snake:
                    snakes_grid[head_y, head_x]= small_head_val
                    # append to heads list
                    snake_heads.append([small_head_val, head_y, head_x])
                    # mark smaller next heads as 8
                    snakes_grid = mark_next_heads(head_y, head_x,
                                            snakes_grid, next_smhead_val)
                # if it's the heads of bigger or equal snakes
                elif len(curr_snake['body']) >= my_body_len and not my_snake:
                    snakes_grid[head_y,head_x]= big_head_val
                    # append to heads list
                    snake_heads.append([big_head_val, head_y, head_x])
                    # mark bigger or equal next heads as 9
                    snakes_grid = mark_next_heads(head_y,
                                    head_x,snakes_grid,next_bighead_val)

                #fill solo grid for crash check
                elif len(curr_snake['body']) == my_body_len and my_snake:
                    solo_grid[head_y, head_x] = my_head_val
                    snakes_grid[head_y, head_x] = my_head_val
            # all snakes body and my head and body except tail
            elif 0 < k < (len(curr_snake['body'])-1):
                body_y = curr_snake['body'][k]['y']
                body_x = curr_snake['body'][k]['x']
                #
                if not my_snake:
                    snakes_grid[body_y,body_x] = body_val
                # fill solo grid
                elif my_snake:
                    snakes_grid[body_y, body_x] = my_body_val
                    solo_grid[body_y, body_x] = body_val
            # tails
            elif k==(len(curr_snake['body'])-1):
                body_y = curr_snake['body'][k]['y']
                body_x = curr_snake['body'][k]['x']
                solo_grid[body_y, body_x] = my_body_val
                snake_tails.append([body_y,body_x])

    return snakes_grid, solo_grid, snake_heads, snake_tails

def check_path_to_tail(head_y, head_x, move_num, snakes_grid, snake_tails):
    found_path=False
    new_head_y = head_y + delta[move_num][0]
    new_head_x = head_x + delta[move_num][1]
    if 0 <= new_head_y < snakes_grid.shape[0] and \
            0 <= new_head_x < snakes_grid.shape[1]:
        # check that we can reach a tail
        for q in range(len(snake_tails)):
            found_path = search(snake_tails[q][0],
                                snake_tails[q][1], new_head_y,
                                new_head_x, snakes_grid, check_path=True)

            if found_path:
                break
    return found_path

def find_free_spaces(snakes_grid, head_y, head_x):
    free_spaces = np.argwhere(snakes_grid==0)
    free_spaces_arr = []
    for i in range(free_spaces.shape[0]):
        curr_free = free_spaces[i,:].tolist()
        dist_to_free = heuristic([head_y, head_x], curr_free)
        free_spaces_arr.append([dist_to_free, curr_free[0], curr_free[1]])

    free_arr = sorted(free_spaces_arr, key=lambda x: x[0])
    return free_arr

def get_away_walls(my_head_y, my_head_x,
                                    snakes_grid,solo_grid,snake_tails):
    path_found = False
    adjust_y = False
    adjust_x = False
    move_num = 0
    my_move = ''
    count = 0
    free_spaces_arr = find_free_spaces(snakes_grid, my_head_y, my_head_x)
    free_arr = free_spaces_arr[::-1]

    while not path_found and count < len(free_arr):
        free_space = free_arr[count]
        goal_y = free_space[1]
        goal_x = free_space[2]

        move_num, my_move, path_found = search(goal_y,goal_x, my_head_y,
                                               my_head_x, snakes_grid)
        # todo: check path out to own tail, don't trap myself
        if path_found and my_move != "snakeshit":
            found_tail = check_path_to_tail(my_head_y, my_head_x,
                                move_num, solo_grid, snake_tails)
            if found_tail:
                break
            else:
                path_found=False


        count+=1

    return my_move, path_found


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
    # pretty print
    #print(f"turn: {turn}\n{json.dumps(data, indent=2)}")
    # board size
    width = data['board']['width']
    height = data['board']['height']

    # my head and body locations
    snakes = data['board']['snakes']
    me = data['you']
    #print(f'me\n{me}')
    my_head_y = me['body'][0]['y']
    my_head_x = me['body'][0]['x']

    my_tail_y = me['body'][-1]['y']
    my_tail_x = me['body'][-1]['x']

    next_tail_y = me['body'][-2]['y']
    next_tail_x = me['body'][-2]['x']
    # means we just ate food so grew
    if my_tail_y==next_tail_y and my_tail_x==next_tail_x:
        next_tail_y = me['body'][-3]['y']
        next_tail_x = me['body'][-3]['x']

    #print(f'tail yx = {my_tail_y},{my_tail_x}\n'
     #     f'nexttail_yx: {next_tail_y},{next_tail_x}')
    my_id = me['id']

    # for comparison with opponent's snakes
    my_body_len = len(me['body'])
    attack = False
    # todo: if longest, start moving towards next_smhead_val on snakes grid
    # my health
    my_health = me['health']
    if my_health > 30:
        for i in range(len(snakes)):
            if len(snakes[i]['body'])<my_body_len:
                attack=True
            else:
                attack=False
                break

    max_dist_for_food = width+height
    #calc_max_dist_for_food(my_health, width, factor=width)

    # flags
    path_found = False

    # debugging
    which_move = ''
    my_move = ''

    # make snakes_grid
    snakes_grid, solo_grid, snake_heads, snake_tails = \
        fill_snakes_grid(snakes, width, height, my_body_len, my_id)

    # find empty spaces
    #free_grid = find_free_land(snakes_grid)

    # leave walls asap
    leave_walls=False

    if (my_head_x==0 or my_head_x==(snakes_grid.shape[1]-1) or
            my_head_y==0 or my_head_y==(snakes_grid.shape[0]-1)) and\
            my_health > width + height:
        leave_walls=True
        my_move, path_found = get_away_walls(my_head_y, my_head_x,
                                 snakes_grid,solo_grid, snake_tails)
        if path_found and my_move!='snakeshit':
            which_move = 'get away walls'
        else:
            leave_walls=False
            path_found = False

    # if me_longest, chase 8s
    if attack and not leave_walls:

        targets = np.argwhere(snakes_grid==8)
        for i in range(targets.shape[0]):
            victim = targets[i,:]
            move_num, my_move, path_found = \
                search(victim[0], victim[1], my_head_y,my_head_x, snakes_grid)
            if path_found and my_move != 'snakeshit':
                found_tail = check_path_to_tail(my_head_y, my_head_x, move_num,
                                                solo_grid, snake_tails)
                if found_tail:
                    which_move = 'hunting'
                else:
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
    #get food
    eating = False
    count=0
    closer=False
    if not path_found and not leave_walls and not attack:
        while not eating and count < len(food_arr):
            # todo: go after food within dist and I'm closer
            curr_food = food_arr[count]
            #if curr_food[0] <= max_dist_for_food:
            food_y = curr_food[1]
            food_x = curr_food[2]
            food_count += 1
            # only go for food I'm closer than other snakes
            for i in range(len(snake_heads)):
                if heuristic([my_head_y, my_head_x], [food_y,food_x])\
                        < heuristic(snake_heads[i], [food_y, food_x]):
                    closer = True
                else:
                    closer= False
                    break


            if closer:
                move_num, my_move, path_found = \
                        search(food_y, food_x, my_head_y,my_head_x, snakes_grid)

                # todo: check path out to own tail, don't trap myself
                if path_found and my_move!='snakeshit':

                    found_tail = check_path_to_tail(my_head_y, my_head_x, move_num,
                                                 solo_grid, snake_tails)
                    if found_tail:
                        which_move = 'get food'
                        eating=True
                    else:
                        path_found=False

            count+=1

    # shorten food_arr
    food_arr = food_arr[food_count:]
    count=0
    #chase my tail
    if not path_found and not leave_walls and not attack:
        # chase tail if nothing in food_arr
        move_num, my_move, path_found = search(my_tail_y, my_tail_x, my_head_y,
                                     my_head_x, snakes_grid)
        if path_found and my_move != 'snakeshit':
            found_tail = check_path_to_tail(my_head_y, my_head_x, move_num,
                                            solo_grid, snake_tails)
            if found_tail:
                which_move = 'my tail'
            else:
                path_found = False

    # find the spot on the map that has most emtpy space
    count = 0
    #if not path_found and not leave_walls and not attack and my_health > 50:



    '''
        free_grid = find_free_land(snakes_grid)
        most_free_y, most_free_x = np.where(free_grid == free_grid.min())
        nearest = np.inf
        while not path_found and count < most_free_y.shape[0]:
            goal_y = most_free_y[count]
            goal_x = most_free_x[count]
            move_num, my_move, path_found = search(goal_y,
                                                   goal_x, my_head_y,
                                                   my_head_x, snakes_grid)

            if path_found and my_move != 'snakeshit':
                found_tail = check_path_to_tail(my_head_y, my_head_x,
                                                move_num, solo_grid, snake_tails)
                if found_tail:
                    which_move = 'get free'
                    break
                else:
                    path_found = False
            count += 1
    '''
    count = 0


    # chase other snakes' tails
    if not path_found and not leave_walls:
        for q in range(len(snake_tails)):
            move_num, my_move, path_found = search(snake_tails[q][0], snake_tails[q][1],
                                         my_head_y,my_head_x,snakes_grid)
            if path_found and my_move!= 'snakeshit':
                new_head_y = my_head_y + delta[move_num][0]
                new_head_x = my_head_x + delta[move_num][1]

                if 0 <= new_head_y < snakes_grid.shape[0] and \
                        0 <= new_head_x < snakes_grid.shape[1]:
                    # check that we can reach a tail
                    for q in range(len(snake_tails)):
                        found_path = search(snake_tails[q][0],
                                            snake_tails[q][1], new_head_y,
                                            new_head_x, solo_grid, check_path=True)

                        if found_path:

                            # print(f'my_move: {my_move}')
                            break
            if path_found:
                which_move = 'opponent tail'
                break

    # chasing tail nor search for food worked
    if not path_found:
        for t in range(len(delta)):
            next_y = my_head_y + delta[t][0]
            next_x = my_head_x + delta[t][1]
            if 0 <= next_y < snakes_grid.shape[0] and \
                    0 <= next_x < snakes_grid.shape[1]:
                if snakes_grid[next_y, next_x]==0 or \
                        snakes_grid[next_y, next_x]==next_smhead_val: #or \

                    my_move = delta_name[t]
                    which_move = 'last resort'
                    path_found=True
                    #print(f'my_move: {my_move}')
                    break



    # Shouts are messages sent to all the other snakes in the game.
    # Shouts are not displayed on the game board.
    shout = "get in my belly!"


    response = {"move": my_move, "shout": shout}
    end = timer()
    print(f'\n\nturn: {turn}\ntime: {end-start}\nmy_move: {my_move}\n '
         f'which_move: {which_move}\n\n')
    #print(f'snakes_grid\n {snakes_grid}\nsolo_grid\n {solo_grid}\n')
    return HTTPResponse(
        status=200,
        headers={"Content-Type": "application/json"},
        body=json.dumps(response),
    )

def find_free_land(snakes_grid):
    '''
    small_head_val = 1
    my_head_val=3
    big_head_val = 5
    body_val = 4
    my_body_val = 7
    next_bighead_val = 9
    next_smhead_val = 8
    '''
    # 4 for snake bodies
    my_body = np.argwhere(snakes_grid==my_body_val)
    opp_bodies = np.argwhere(snakes_grid==body_val)
    # as is
    big_heads = np.argwhere(snakes_grid==big_head_val)
    small_heads = np.argwhere(snakes_grid==small_head_val)
    my_head = np.argwhere(snakes_grid==my_head_val)
    # as is
    next_bheads = np.argwhere(snakes_grid==next_bighead_val)
    # heuristic only
    next_smhead = np.argwhere(snakes_grid==next_smhead_val)

    total_grid = np.zeros(snakes_grid.shape, dtype=np.int)
    for i in range(snakes_grid.shape[0]):
        for j in range(snakes_grid.shape[1]):
            for k in range(my_body.shape[0]):
                total_grid[i,j] += int(4/heuristic(my_body[k,:], [i,j]))
            for z in range(opp_bodies.shape[0]):
                total_grid[i,j] += int(4/heuristic(opp_bodies[z,:], [i,j]))
            for m in range(big_heads.shape[0]):
                total_grid[i,j] += int(5/ heuristic(big_heads[m,:], [i,j]))
                        #snakes_grid[big_heads[m,0], big_heads[m,1]]
            for c in range(small_heads.shape[0]):
                total_grid[i, j] += int(4/heuristic(small_heads[c,:], [i, j]))
                                    #snakes_grid[small_heads[c, 0],
                                     #           small_heads[c, 1]]
            for n in range(next_bheads.shape[0]):
                total_grid[i,j] += int(3/heuristic(next_bheads[n,:], [i,j]))
                    #snakes_grid[next_bheads[n,0], next_bheads[n,1]]
            for p in range(next_smhead.shape[0]):
                total_grid[i, j] += int(1/heuristic(next_smhead[p,:], [i, j]))

            total_grid[i,j] += 1/int(heuristic([my_head[0, 0], my_head[0,1]],
                                                [i,j]))
    return total_grid

def heuristic(start_node, goal_node):
    start_x = start_node[1]
    start_y = start_node[0]
    goal_x = goal_node[1]
    goal_y = goal_node[0]
    dx = abs(start_x - goal_x)
    dy = abs(start_y - goal_y)
    return (dx + dy)


def make_heuristic_map(goal, snakes_grid):
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
    # print(f"end data:\n{json.dumps(data, indent=2)}")
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