import json
import os
import random
import numpy as np

import bottle
from bottle import HTTPResponse


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
    #print(f"start_data:\n{json.dumps(data, indent=2)}")

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
    '''
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
    '''
    # my_moves
    delta = [[-1, 0],  # go up
             [0, -1],  # go left
             [1, 0],  # go down
             [0, 1]]  # go right

    delta_name = ['up', 'left','down', 'right']
    # call for data
    data = bottle.request.json
    #board size
    width = data['board']['width']
    height = data['board']['height']
    # my head
    # my head and body locations
    snakes = data['board']['snakes']
    my_head_x = data['you']['body'][0]['x']
    my_head_y = data['you']['body'][0]['y']
    my_id = data['you']['id']

    # get list of snakes and make snakes_grid
    # my snake is 3 (head), 4 (body), opponents are 0,0 for smaller snake,
    # 5,6 for larger snake
    snakes_grid = np.zeros((width, height), dtype=np.int)
    # for comparison with opponent's snakes
    my_body_len = len(data['you']['body'])
    # vals for larger or equal size opponents
    head_val = 5
    body_val = 6

    #todo: debugging
    which_move = ''
    my_move = ''

    for j in range(len(snakes)):
        curr_snake = snakes[j]
        # iterate to last since tail is gone next move
        for k in range(0, len(curr_snake['body']) - 1, 1):
            head_val = 5
            body_val = 6
            # head of all snakes including me
            if k == 0:
                snakes_grid[curr_snake['body'][k]['y'],
                            curr_snake['body'][k]['x']] = head_val
                # todo: pick 1/4 connected for next_head?
                #if snake is not me and equal or bigger,  next heads are marked
                if len(curr_snake['body']) >= my_body_len and \
                        curr_snake['id'] != my_id:
                    next_head_candidates = []
                    for s in range(len(delta)):
                        next_head_y = curr_snake['body'][k]['y'] \
                                      + delta[s][0]
                        next_head_x = curr_snake['body'][k]['x'] \
                                      + delta[s][1]
                        # if in bounds and not its own body
                        if 0 <= next_head_y < snakes_grid.shape[0] \
                                and 0 <= next_head_x < snakes.grid.shape[1]\
                                and snakes_grid[next_head_y, next_head_x]==0:
                            snakes_grid[next_head_y, next_head_x] = head_val
                            #next_head_candidates.append([next_head_y,next_head_x])
                    # random choice on candidates
                    #next_head = random.choice(next_head_candidates)
                    #snakes_grid[next_head[0], next_head[1]] = head_val

            # snakes body
            else:
                snakes_grid[curr_snake['body'][k]['y'],
                            curr_snake['body'][k]['x']] = body_val

    #todo: hits own tail so include on snakes grid
    snakes_grid[data['you']['body'][-1]['y'], data['you']['body'][-1]['x']] = 3

    #list of dicts of food locations
    food = data['board']['food']
    # list in order of nearest to furthest food tuples (dist, y,x)
    food_arr = []
    get_food = False
    #if there is food
    if len(food)>0:
        for z in range(len(food)):
            food_dist = heuristic((my_head_y, my_head_x),
                                  (food[z]['y'],food[z]['x']))
            # dont' go for food further than width away
            #todo: 0.7 is param
            if food_dist < width * 1.0:
                food_arr.append([food_dist, food[z]['y'], food[z]['x']])
                get_food=True
        food_arr = sorted(food_arr, key=lambda x: x[0])
    print(f'\n\nfood arr {food_arr}\n\n')

    #print(f"snakes_grid\n {snakes_grid}")
    cost = 1
    #first_my_move = 0

    path_found=False
    # there is a food so A star for route to food using snake grid for g
    if get_food:
        for q in range(len(food_arr)):
            # goal y and x
            goal_y = food_arr[q][1]
            goal_x = food_arr[q][2]
            # visited array
            closed = np.zeros(snakes_grid.shape, dtype=np.int)
            closed[my_head_y, my_head_x] = 1
            # expand is final map returned with numbered spots
            expand = np.full(snakes_grid.shape, -1, dtype=np.int)

            g = 0 # each step is 1
            heuristic_map = make_heuristic_map((goal_y, goal_x),
                                               snakes_grid)

            f = g + heuristic_map[my_head_y, my_head_x]

            open_arr = [[f,g,my_head_y, my_head_x]]
            found = False # set when search complete
            resign = False # set when can't expand
            count = 0
            # only need to return the next my_move? or calculate entire path?
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
                    expand[y,x] = count
                    count+=1

                    if y == goal_y and x == goal_x:
                        found = True
                    else:
                        for i in range(len(delta)):
                            new_y = y + delta[i][0]
                            new_x = x + delta[i][1]

                            # if in-bounds, also mark as head
                            if 0 <= new_y < snakes_grid.shape[0] and \
                                0 <= new_x < snakes_grid.shape[1]:
                                # if unvisited and traversible
                                if closed[new_y,new_x] ==0 and \
                                        snakes_grid[new_y,new_x]==0:
                                    # todo: the first my_move is where we are going

                                    g2 = g + cost
                                    f2 = g2 + heuristic_map[new_y,new_x]
                                    open_arr.append([f2,g2,new_y,new_x])
                                    closed[new_y,new_x]=1
            # found goal or resigned
            if found:

                # find next my_move, how to get to any non -1 in expand
                # todo: but also 2 deep?
                for i in range(len(delta)):
                    next_y = my_head_y + delta[i][0]
                    next_x = my_head_x + delta[i][1]
                    #if expand is "pos" and in bounds
                    if 0 <= next_y < expand.shape[0] and \
                            0 <= next_x < expand.shape[1] and \
                            expand[next_y, next_x]==1:
                        my_move = delta_name[i]
                        path_found=True
                        which_move='food'
                        break
            if path_found:
                print(f'heuristics map\n{heuristic_map}')
                break

    '''
                        curr_spot = expand[next_y, next_x]
                        for j in range(len(delta)):
                            n_next_y = next_y + delta[j][0]
                            n_next_x = next_x + delta[j][1]
                            if 0 <= n_next_y < expand.shape[0] and \
                                    0 <= n_next_x < expand.shape[1] and \
                                    expand[n_next_y, n_next_x]>curr_spot:
                                #print(f'expand\n {expand}')
                                my_move = delta_name[i]
                                path_found = True
                                which_move = 'food'
                                break
                    # found path so break
                    if path_found:
                        break
    '''

    # if food A-star had no path or no food within reach to begin with
    # so chase tail
    if not path_found:
        # chase tail if nothing in food_arr
        goal_y = snakes[0]['body'][-1]['y']
        goal_x = snakes[0]['body'][-1]['x']
        # visited array
        closed = np.zeros(snakes_grid.shape, dtype=np.int)
        closed[my_head_y, my_head_x] = 1
        # expand is final map returned with numbered spots
        expand = np.full(snakes_grid.shape, -1, dtype=np.int)

        g = 0  # each step is 1
        heuristic_map = make_heuristic_map([goal_y, goal_x],
                                           snakes_grid)
        print(f'heuristics_map\n{heuristic_map}')
        f = g + heuristic_map[my_head_y, my_head_x]

        open_arr = [[f, g, my_head_y, my_head_x]]
        found = False  # set when search complete
        resign = False  # set when can't expand
        count = 0
        # calculate entire path but only use first my_move
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

                        # if in-bounds, also mark as head
                        if 0 <= new_y < snakes_grid.shape[0] and \
                                0 <= new_x < snakes_grid.shape[1]:
                            # if unvisited and traversible
                            if closed[new_y,new_x] == 0 and \
                                    snakes_grid[new_y,new_x] == 0:
                                # todo: the first my_move is where we are going

                                g2 = g + cost
                                f2 = g2 + heuristic_map[new_y,new_x]
                                open_arr.append([f2, g2, new_y, new_x])
                                closed[new_y,new_x] = 1
        # found goal or resigned
        if found:

            # find next my_move, how to get to spot that's not -1 in expand
            # but choose a spot that has continuation, not just an explored one
            for i in range(len(delta)):
                next_y = my_head_y + delta[i][0]
                next_x = my_head_x + delta[i][1]
                if 0 <= next_y < expand.shape[0] and \
                        0 <= next_x < expand.shape[1] and \
                        expand[next_y, next_x]==1:
                    my_move = delta_name[i]
                    path_found = True
                    which_move = 'tail'
                    break

                '''
                    curr_spot = expand[next_y, next_x]
                    for j in range(len(delta)):
                        n_next_y = next_y + delta[j][0]
                        n_next_x = next_x + delta[j][1]
                        if 0 <= n_next_y < expand.shape[0] and \
                            0 <= n_next_x < expand.shape[1] and \
                            expand[n_next_y, n_next_x]>curr_spot:
                            #print(f'expand\n {expand}')
                            my_move = delta_name[i]
                            path_found = True
                            which_move = 'tail'
                            break
                if path_found:
                    break
                '''
    # pretty print
    #print(f"my_move_data:\n{json.dumps(data, indent=2)}")

    #chasing tail nor search for food worked so random?
    if not path_found:
        for t in range(len(delta)):
            next_y = my_head_y + delta[t][0]
            next_x = my_head_x + delta[t][1]
            if 0 <= next_y < snakes_grid.shape[0] and \
                    0 <= next_x < snakes_grid.shape[1] and \
                    snakes_grid[next_y, next_x]==0:
                my_move = delta_name[t]
                for v in range(len(delta)):
                    n_next_y = next_y + delta[v][0]
                    n_next_x = next_x + delta[v][1]
                    if 0 <= n_next_y < snakes_grid.shape[0] and \
                            0 <= n_next_x < snakes_grid.shape[1] and \
                            snakes_grid[n_next_y,n_next_x]==0:
                        my_move = delta_name[t]
                        which_move = 'last resort'
                        path_found=True
                        break
            if path_found:
                break
    # Choose a random direction to my_move in
    #if not path_found:
        #directions = ["up", "down", "left", "right"]
      #  my_move = random.choice(directions)
       # path_found=True

    # Shouts are messages sent to all the other snakes in the game.
    # Shouts are not displayed on the game board.
    shout = "tssss!"
    turn = data['turn']
    print(f'\n\nturn: {turn}\nmy_move: {my_move}\n which_move: {which_move}\n\n')
    response = {"move": my_move, "shout": shout}
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
    return (dx+dy)

def make_heuristic_map(goal, snakes_grid):
    goal_y = goal[0]
    goal_x = goal[1]
    heuristic_map = np.zeros(snakes_grid.shape, dtype=np.int)
    for i in range(heuristic_map.shape[0]):
        for j in range(heuristic_map.shape[1]):
            dy = np.abs(i-goal_y)
            dx = np.abs(j-goal_x)
            heuristic_map[i,j] = dy + dx

    return heuristic_map


@bottle.post("/end")
def end():
    """
    Called every time a game with your snake in it ends.
    """
    data = bottle.request.json
    #print(f"end data:\n{json.dumps(data, indent=2)}")
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
