import json
import os
import random
import numpy as np
from math import ceil
import bottle
from bottle import HTTPResponse

# my_moves
delta = [[-1, 0],  # go up
         [0, -1],  # go left
         [1, 0],  # go down
         [0, 1]]  # go right

delta_name = ['up', 'left', 'down', 'right']

cost = 1

# vals for smaller heads, equal or big, all bodies and next heads
small_head_val = 1
big_head_val = 5
body_val = 6
next_head_val = 9


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




def search(goal_y, goal_x, my_head_y, my_head_x, snakes_grid, snakes_grid_two):

    my_move = ''
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
            next = open_arr.pop()
            y = next[2]
            x = next[3]
            g = next[1]
            f = g + heuristic_map[y, x]
            expand[y, x] = count
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
                        # if unvisited and traversible (smaller snake's head
                        #is traversible)
                        if closed[new_y, new_x] == 0 and \
                                snakes_grid[new_y, new_x] == 0:
                            g2 = g + cost
                            f2 = g2 + heuristic_map[new_y, new_x]
                            open_arr.append([f2, g2, new_y, new_x])
                            closed[new_y, new_x] = 1
    # found goal or resigned
    if found:
        # find next my_move, how to get to spot that's not -1 in expand
        # but choose a spot that has continuation, not just an explored one
        # todo: can work backwards from where expand is >0 and compare to
        # todo: start y and x and find move to get there
        for i in range(len(delta)):
            next_y = my_head_y + delta[i][0]
            next_x = my_head_x + delta[i][1]
            if 0 <= next_y < expand.shape[0] and \
                    0 <= next_x < expand.shape[1] and \
                    expand[next_y, next_x] > 0:
                curr_spot_val = expand[next_y, next_x]
                # find next move that is the continuation
                for j in range(len(delta)):
                    n_next_y = next_y + delta[j][0]
                    n_next_x = next_x + delta[j][1]
                    if (0 <= n_next_y < expand.shape[0] and
                            0 <= n_next_x < expand.shape[1]) and \
                            snakes_grid[n_next_y, n_next_x] == 0:
                            #expand[n_next_y, n_next_x] > curr_spot_val:

                            #snakes_grid_two[n_next_y, n_next_x]==1):
                        # print(f'expand\n {expand}')
                        my_move = delta_name[i]
                        break
                    else:
                        continue
                return my_move, found
    else:
        my_move = 'up'
        return my_move, found


def fill_food_arr(food, my_head_y, my_head_x):
    # list in order of nearest to furthest food tuples (dist, y,x)
    food_arr = []
    for z in range(len(food)):
        food_dist = heuristic([my_head_y, my_head_x],
                              [food[z]['y'], food[z]['x']])
        food_arr.append([food_dist, food[z]['y'], food[z]['x']])


    food_arr = sorted(food_arr, key=lambda x: x[0])
    # print(f'\n\nfood arr {food_arr}\n\n')
    return food_arr

def mark_next_heads(delta, curr_snake, snakes_grid, snakes_grid_two,
                    next_head_val):
    for s in range(len(delta)):
        next_head_y = curr_snake['body'][0]['y'] \
                      + delta[s][0]
        next_head_x = curr_snake['body'][0]['x'] \
                      + delta[s][1]
        # if in bounds and space is free
        if 0 <= next_head_y < snakes_grid.shape[0] \
                and 0 <= next_head_x < snakes_grid.shape[1] \
                and snakes_grid[next_head_y, next_head_x] == 0:
            snakes_grid[next_head_y, next_head_x] = next_head_val
            snakes_grid_two[next_head_y, next_head_x] = next_head_val

        return snakes_grid, snakes_grid_two

def fill_snakes_grid(snakes, width, height, my_body_len, my_id):
    # my_moves
    delta = [[-1, 0],  # go up
             [0, -1],  # go left
             [1, 0],  # go down
             [0, 1]]  # go right
    '''
    # vals for smaller heads, equal or big, all bodies and next heads
    small_head_val = 1
    big_head_val = 5
    body_val = 6
    next_head_val = 9
    '''
    snake_heads = []
    snake_tails = []
    #flag for me or other snake
    my_snake = False

    # second grid for 2 moves in
    snakes_grid = np.zeros((width, height), dtype=np.int)
    snakes_grid_two = np.zeros(snakes_grid.shape, dtype=np.int)
    solo_grid = np.zeros(snakes_grid.shape, dtype=np.int)

    for j in range(len(snakes)):
        curr_snake = snakes[j]
        if curr_snake['id'] == my_id:
            my_snake=True
        else:
            my_snake=False
        # fill grid
        for k in range(0, len(curr_snake['body']), 1):
            # heads of opp snakes
            if k == 0 and not my_snake:
                # if opp smaller, don't fill grid
                if len(curr_snake['body']) < my_body_len:
                    # append to heads list
                    snake_heads.append([small_head_val, curr_snake['body'][k]['y'],
                                    curr_snake['body'][k]['x']])
                # if bigger or equal snakes
                if len(curr_snake['body']) >= my_body_len:
                    snakes_grid[curr_snake['body'][k]['y'],
                                curr_snake['body'][k]['x']] = big_head_val
                    snakes_grid_two[curr_snake['body'][k]['y'],
                                curr_snake['body'][k]['x']] = big_head_val
                    # append to heads list
                    snake_heads.append([big_head_val, curr_snake['body'][k]['y'],
                                        curr_snake['body'][k]['x']])
                # mark all next heads as 9, even smaller ones
                snakes_grid, snakes_grid_two = mark_next_heads(delta,
                            curr_snake, snakes_grid,
                            snakes_grid_two, next_head_val)
            # snakes body and my head and body except tail
            elif 0 < k < len(curr_snake['body'])-1:
                snakes_grid[curr_snake['body'][k]['y'],
                            curr_snake['body'][k]['x']] = body_val
                # fill up to second to last body segment for grid two
                if 0 < k < len(curr_snake['body'])-2:
                    snakes_grid_two[curr_snake['body'][k]['y'],
                            curr_snake['body'][k]['x']] = body_val
                # fill solo grid todo: delete?
                if my_snake:
                    solo_grid[curr_snake['body'][k]['y'],
                                curr_snake['body'][k]['x']] = body_val
            # tails
            elif k==len(curr_snake['body'])-1:
                snake_tails.append([curr_snake['body'][k]['y'],
                                    curr_snake['body'][k]['x']])


    return snakes_grid, snakes_grid_two, solo_grid, snake_heads, snake_tails

#todo: maybe just need to chase food I'm closer than other snakes?
def calc_max_dist_for_food(my_health, width, factor=1):
    # make it inverse to health
    max_dist_for_food = width*2
    if my_health > 90:
        max_dist_for_food = width/width
    elif my_health > 75:
        max_dist_for_food = min(factor*3, width)
    elif my_health > 50:
        max_dist_for_food = min(factor*5, width)
    elif my_health >40:
        max_dist_for_food = width
    elif my_health > width*2:
        max_dist_for_food = width*2
    elif my_health > width:
        max_dist_for_food = width*2
    else:
        max_dist_for_food = width*2

    return max_dist_for_food

@bottle.post("/move")
def move():
    """
    Called when the Battlesnake Engine needs to know your next my_move.
    The data parameter will contain information about the board.
    Your response must include your my_move of up, down, left, or right.
    """
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
    my_head_y = data['you']['body'][0]['y']
    my_head_x = data['you']['body'][0]['x']

    my_tail_y = data['you']['body'][-1]['y']
    my_tail_x = data['you']['body'][-1]['x']
    my_id = data['you']['id']
    # my health
    my_health = data['you']['health']
    # calculate max distance we go for food
    max_dist_for_food = calc_max_dist_for_food(my_health, width)

    # for comparison with opponent's snakes
    my_body_len = len(data['you']['body'])

    # flags
    path_found = False

    # todo: debugging
    which_move = ''
    my_move = ''

    # make snakes_grid
    snakes_grid, snakes_grid_two, solo_grid, snake_heads, snake_tails = \
        fill_snakes_grid(snakes, width, height, my_body_len, my_id)

    # list of dicts of food locations
    food = data['board']['food']
    # list in order of nearest to furthest food tuples (dist, y,x)
    food_arr = []
    # if there is food
    if len(food) > 0:
        food_arr = fill_food_arr(food, my_head_y, my_head_x)
    # there is a food so A star for route to food using snake grid for g
    food_count = 0

    get_food=False
    nearby_snakes = 0
    if not path_found:
        for q in range(len(food_arr)):
            # todo: go after food within dist and I'm closer
            #if food_arr[q][0] <= max_dist_for_food:
            # iterate snakeheads
            for r in range(len(snake_heads)):
                # if other snakes farther get food
                #todo: I'm closer or also equal?
                '''
                #smaller snake
                if snake_heads[r][0]==1:
                    get_food=True
                # big or equal snake
                elif snake_heads[r][0]==5:
                '''
                if heuristic([my_head_y, my_head_x],
                         [food_arr[q][1], food_arr[q][2]])\
                            < heuristic(snake_heads[r], [food_arr[q][1],
                                                         food_arr[q][2]]):
                    continue
                else:
                    nearby_snakes+=1
                    #get_food=False
                    #break
            food_count += 1
            if nearby_snakes<2 and \
                    snakes_grid[food_arr[q][1], food_arr[q][2]]==0:
                # goal y and x
                goal_y = food_arr[q][1]
                goal_x = food_arr[q][2]
                my_move, path_found = search(goal_y, goal_x, my_head_y,
                                             my_head_x, snakes_grid, snakes_grid_two)
                #todo: check path out to own tail

            #else:
             # continue
            if path_found:
                which_move = 'food near'
                break
    # shorten food_arr
    food_arr = food_arr[food_count:]

    #chase my tail
    if not path_found:
        # chase tail if nothing in food_arr
        my_move, path_found = search(my_tail_y, my_tail_x, my_head_y,
                                     my_head_x, snakes_grid, snakes_grid_two)
        if path_found:
            which_move = 'tail'

   # chase other snakes' tails
    if not path_found:
        for q in range(len(snake_tails)):
            my_move, path_found = search(snake_tails[q][0], snake_tails[q][1],
                                         my_head_y,my_head_x,
                                         snakes_grid, snakes_grid_two)
            if path_found:
                which_move='other tail'

    # chasing tail nor search for food worked so just go two deep
    if not path_found:
        for t in range(len(delta)):
            next_y = my_head_y + delta[t][0]
            next_x = my_head_x + delta[t][1]
            if 0 <= next_y < snakes_grid.shape[0] and \
                    0 <= next_x < snakes_grid.shape[1] and \
                    (snakes_grid[next_y, next_x]==0
                    or snakes_grid[next_y, next_x]==9):
                my_move = delta_name[t]
                which_move = 'last resort'
                for v in range(len(delta)):
                    n_next_y = next_y + delta[v][0]
                    n_next_x = next_x + delta[v][1]
                    if 0 <= n_next_y < snakes_grid.shape[0] and \
                            0 <= n_next_x < snakes_grid.shape[1] and \
                            (snakes_grid_two[n_next_y, n_next_x] == 0
                                or snakes_grid_two[n_next_y, n_next_x]==9):
                        my_move = delta_name[t]
                        which_move = 'last resort'
                        path_found = True
                        break
            if path_found:
                break

    # Shouts are messages sent to all the other snakes in the game.
    # Shouts are not displayed on the game board.
    shout = "namenayo!"

    print(f'\n\nturn: {turn}\nmy_move: {my_move}\n '
                    f'which_move: {which_move}\n\n')
    response = {"move": my_move, "shout": shout}
    return HTTPResponse(
        status=200,
        headers={"Content-Type": "application/json"},
        body=json.dumps(response),
    )

def find_stretch_goal(width, height):
    pass
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