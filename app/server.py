import json
import os
import random
import numpy as np

import bottle
from bottle import HTTPResponse


@bottle.route("/")
def index():
    return "I'm a nasty snake!"


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
    print(f"start_data:\n{json.dumps(data, indent=2)}")

    response = {"color": "#fc0313", "headType": "fang", "tailType": "bolt"}
    return HTTPResponse(
        status=200,
        headers={"Content-Type": "application/json"},
        body=json.dumps(response),
    )


@bottle.post("/move")
def move():
    """
    Called when the Battlesnake Engine needs to know your next move.
    The data parameter will contain information about the board.
    Your response must include your move of up, down, left, or right.
    """
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

    # moves
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
    my_head_x = snakes[0]['body'][0]['x']
    my_head_y = snakes[0]['body'][0]['y']

    #list of dicts of food locations
    food = data['board']['food']
    # list in order of nearest to furthest food
    food_arr = []
    for z in range(len(food)):
        food_dist = heuristic((my_head_y, my_head_x),
                              (food[z]['y'],food[z]['x']))
        #if food_dist > width:
         #   continue

        food_arr.append([food_dist, food[z]['y'], food[z]['x']])

    food_arr = sorted(food_arr, key=lambda x: x[0])
    print(f'nearest food {food_arr[0]}')


    # get list of snakes and make snakes_grid
    # my snake is 3 (head), 4 (body), opponents are 0,0 for smaller snake,
    # 5,6 for larger snake
    snakes_grid = np.zeros((width, height), dtype=np.int)
    # for comparison with opponent's snakes
    my_body_len = len(snakes[0]['body'])
    # vals for larger or equal size opponents
    head_val = 5
    body_val = 6

    for j in range(len(snakes)):
        curr_snake = snakes[j]
        #iterate to last since tail is gone next move
        for k in range(0,len(curr_snake['body'])-1,1):
            # if opponent bigger than or equal to me avoid
            if j!=0 and len(curr_snake['body']) >= my_body_len:
                head_val = 5
                body_val = 6
                # if head
                if k==0:
                    snakes_grid[curr_snake['body'][k]['y'],
                                curr_snake['body'][k]['x']] = head_val
                    # mark the 4 connected
                    for i in range(len(delta)):
                        next_head_y = curr_snake['body'][k]['y'] \
                                    + delta[i][0]
                        next_head_x = curr_snake['body'][k]['x'] \
                                    + delta[i][1]
                        # if in bounds
                        if 0 <= next_head_y < snakes_grid.shape[0] \
                                and 0 <= next_head_x < snakes.grid.shape[1]:
                            snakes_grid[next_head_y, next_head_x] = head_val
            # opponent's body
            elif j!= 0:
                snakes_grid[curr_snake['body'][k]['y'],
                            curr_snake['body'][k]['x']] = body_val

            # if opponent smaller than me, don't care so leave as 0
            elif j!=0 and len(curr_snake['body']) < my_body_len:
                break
            # if it's my head
            elif j==0 and k==0:
                snakes_grid[curr_snake['body'][k]['y'],
                        curr_snake['body'][k]['x']] = 3
            # my body, can't hit it anyways
            elif j==0:
                snakes_grid[curr_snake['body'][k]['y'],
                        curr_snake['body'][k]['x']] = body_val

        # set them to default larger or equal to me
        head_val = 5
        body_val = 6

    print(f"snakes_grid\n {snakes_grid}")
    cost = 1
    first_move = 0
    move = ''
    # there is a food so A star for route with snake grid
    if len(food_arr) > 0:

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
            print(f'heuristics map\n{heuristic_map}')
            f = g + heuristic_map[my_head_y, my_head_x]

            open_arr = [[f,g,my_head_y, my_head_x]]
            found = False # set when search complete
            resign = False # set when can't expand
            count = 0
            # only need to return the next move? or calculate entire path?
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
                                    # the first move is where we are going
                                    if count == 1:
                                        first_move = i
                                    g2 = g + cost
                                    f2 = g2 + heuristic_map[new_y,new_x]
                                    open_arr.append([f2,g2,new_y,new_x])
                                    closed[new_y,new_x]=1
            # found goal or resigned
            if not resign:
                # find next move, how to get to "1" in expand
                for i in range(len(delta)):
                    next_y = my_head_y + delta[i][0]
                    next_x = my_head_x + delta[i][1]
                    #if expand is "1" and in bounds
                    if 0 <= next_y < expand.shape[0] and \
                            0 <= next_x < expand.shape[1] and \
                            expand[next_y, next_x] == 1:
                        print(f'expand\n {expand}')
                        move = delta_name[i]
                break
    # chase tail if nothing in food_arr
    else:
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
        # only need to return the next move? or calculate entire path?
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
                                # the first move is where we are going
                                if count == 1:
                                    first_move = i
                                g2 = g + cost
                                f2 = g2 + heuristic_map[new_y,new_x]
                                open_arr.append([f2, g2, new_y, new_x])
                                closed[new_y][new_x] = 1
        # found goal or resigned
        if not resign:
            # find next move, how to get to "1" in expand
            for i in range(len(delta)):
                next_y = my_head_y + delta[i][0]
                next_x = my_head_x + delta[i][1]
                if 0 <= next_y < expand.shape[0] and \
                        0 <= next_x < expand.shape[1] and \
                        expand[next_y, next_x]==1:
                    print(f'expand\n {expand}')
                    move = delta_name[i]

    # pretty print
    #print(f"move_data:\n{json.dumps(data, indent=2)}")

    # Choose a random direction to move in
    #directions = ["up", "down", "left", "right"]

    #move = random.choice(directions)

    # Shouts are messages sent to all the other snakes in the game.
    # Shouts are not displayed on the game board.
    shout = "Kurae!"
    print(f'move: {move}')
    response = {"move": move, "shout": shout}
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
    print(f"end data:\n{json.dumps(data, indent=2)}")
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
