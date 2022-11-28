from tree import Node
import turtle
from math import dist
from numpy import cumsum
import numpy as np
import funtions as fn


class Graph:
    def __init__(self):
        self.edges = {}
        self.weights = {}

    def neighbors(self, id):
        return self.edges[id]


def cost_of_Path(path, graph):
    # Returns the cumulated cost of path
    cum_costs = [0]
    for i in range(len(path) - 1):
        cum_costs += [1]
    return cumsum(cum_costs)


def draw_square(node_id, color="medium sea green", scale=1,
                correction=345, ts=None, text=None):
    if ts is None:
        ts = turtle.Turtle(shape="square")
    ts.shapesize(0.5, 0.5)
    ts.color(color)
    ts.penup()
    x, y = geo_pos(node_id)
    ts.goto(x * scale, correction - y * scale)
    if text is not None:
        ts.write(str(text), font=("Arial", 20, "normal"))


def draw_Maze_Section(t):
    correction = 345
    scale = 1
    for i in range(1, 51):
        x, y = geo_pos(str(i))
        t.goto(x * scale, correction - y * scale)


def find_least_F(oL):
    """
    finds the node with least F in oL (queue)
    This is equivalent to build a priority queue
    """
    mF = min(get_F(oL))
    for ni in range(len(oL)):
        if oL[ni][1] == mF:
            return oL.pop(ni)[0]


def get_F(oL):
    """
    Returns costs of queue F = C + H
    C: cost of route
    H: heuristic cost
    """
    return [i[1] for i in oL]


def path_from_Origin(origin, n, parents):
    # Builds shortest path from search result (parents)
    if origin == n:
        return []
    pathO = [n]
    i = n
    while True:

        i = parents[i]
        pathO.insert(0, i)
        if i == origin:
            return pathO


def a_Star(graph, start, goal):
    openL = []
    openL.append((start, 0))
    parents = {}
    costSoFar = {}
    parents[start] = None
    costSoFar[start] = 0

    while bool(len(openL)):
        current = find_least_F(openL)
        draw_square(current)  # Draw search expansion
        if current == goal:
            break
        for successor in graph.neighbors(current):
            newCost = costSoFar[current] + 1
            if successor not in costSoFar or newCost < costSoFar[successor]:
                costSoFar[successor] = newCost
                priority = newCost + heuristic(successor)
                openL.append((successor, priority))
                parents[successor] = current
        print(openL)
    return parents


# function greedy algorithm
def greedy(graph, start, goal):
    openL = []
    openL.append((start, 0))
    parents = {}
    costSoFar = {}
    parents[start] = None
    costSoFar[start] = 0

    while bool(len(openL)):
        current = find_least_F(openL)
        draw_square(current)  # Draw search expansion

        if current == goal:
            break

        for successor in graph.neighbors(current):
            newCost = costSoFar[current] + 1
            if successor not in costSoFar or newCost < costSoFar[successor]:
                costSoFar[successor] = newCost
                priority = heuristic(successor)
                openL.append((successor, priority))
                parents[successor] = current

        print(openL)

    return parents


def dfs_non_recursive(graph, source, endNode):
    parents = {}
    parents[source] = None
    # if source is None or source not in graph.neighbors(source):
    #    return "Invalid input"
    path = []
    stack = [source]
    while len(stack) != 0:
        s = stack.pop()
        draw_square(s)
        if s == endNode:
            break
        if s not in path:
            path.append(s)
        if s not in graph.edges.keys():
            # leaf node
            continue

        for neighbor in graph.neighbors(s):
            if neighbor not in path:
                stack.append(neighbor)
                parents[neighbor] = s
    return parents


def bfs_non_recursive(graph, source, endNode):
    parents = {}
    parents[source] = None
    # if source is None or source not in graph.neighbors(source):
    #    return "Invalid input"
    path = []
    stack = [source]
    while len(stack) != 0:
        s = stack.pop(0)
        if s == endNode:
            draw_square(s)
            break
        if s not in path:
            path.append(s)
            draw_square(s)
        if s not in graph.edges.keys():
            # leaf node
            continue

        for neighbor in graph.neighbors(s):
            if neighbor not in path:
                stack.append(neighbor)
                parents[neighbor] = s
    return parents


def geo_pos(id):
    """
    Builds Maze's cities positional information
    The map is a png image used as backgroud,
    the position corresponds to an approximated pixel
    for each city
    """
    G = {
        '1': (56, 56),
        '2': (112, 56),
        '3': (168, 56),
        '4': (168, 112),
        '5': (112, 112),
        '6': (56, 112),
        '7': (56, 168),
        '8': (112, 168),
        '9': (112, 224),
        '10': (168, 224),
        '11': (168, 168),
        '12': (224, 168),
        '13': (56, 224),
        '14': (56, 280),
        '15': (112, 280),
        '16': (168, 280),
        '17': (224, 280),
        '18': (224, 224),
        '19': (280, 224),
        '20': (280, 280),
        '21': (336, 280),
        '22': (280, 168),
        '23': (280, 112),
        '24': (224, 112),
        '25': (224, 56),
        '26': (292, 56),
        '27': (349, 56),
        '28': (349, 112),
        '29': (405, 56),
        '30': (390, 112),
        '31': (402, 168),
        '32': (453, 168),
        '33': (460, 112),
        '34': (461, 56),
        '35': (526, 56),
        '36': (561, 56),
        '37': (562, 112),
        '38': (512, 112),
        '39': (515, 168),
        '40': (518, 224),
        '41': (568, 224),
        '42': (570, 168),
        '43': (471, 224),
        '44': (464, 280),
        '45': (412, 280),
        '46': (512, 280),
        '47': (564, 280),
        '48': (346, 168),
        '49': (352, 224),
        '50': (401, 224)
    }

    return G[id]


def heuristic(id):
    """
    Builds Maze heuristic
    """
    H = {
        '1': 555.1936599061628,
        '2': 504.4601074416093,
        '3': 454.9637348185018,
        '4': 430.1627598944381,
        '5': 482.21157182299146,
        '6': 535.0588752651432,
        '7': 520.1999615532472,
        '8': 465.66941063376714,
        '9': 455.45581563967323,
        '10': 399.93999549932494,
        '11': 411.53371672318667,
        '12': 357.9720659492861,
        '13': 511.0772935672255,
        '14': 508.0,
        '15': 452.0,
        '16': 396.0,
        '17': 340.0,
        '18': 344.58090486850836,
        '19': 289.4684784220901,
        '20': 284.0,
        '21': 228.0,
        '22': 305.28675044947494,
        '23': 329.9696955782455,
        '24': 379.24134795668044,
        '25': 407.1559897631373,
        '26': 352.3634487287239,
        '27': 310.4851043125902,
        '28': 272.85344051340087,
        '29': 274.6943756249844,
        '30': 241.8677324489565,
        '31': 196.9466932954194,
        '32': 157.6863976378432,
        '33': 197.5854245636555,
        '34': 246.54614172604687,
        '35': 227.20035211240318,
        '36': 224.0200883849482,
        '37': 168.01190434013893,
        '38': 175.86358349584486,
        '39': 122.24974437601085,
        '40': 72.47068372797375,
        '41': 56.1426753904728,
        '42': 112.16059914247961,
        '43': 108.55873986004076,
        '44': 100.0,
        '45': 152.0,
        '46': 52.0,
        '47': 0.0,
        '48': 245.08773939142694,
        '49': 219.2715211786519,
        '50': 172.35138525698017
    }

    return H[id]

def get_elements(matrix, element):
    aux = []
    item = np.where(matrix == element)
    x, y = item[0].tolist(), item[1].tolist()

    for itemx, itemy in zip(x, y):
        aux.append([itemx, itemy])
    return aux

def main2():
    path_image = 'image/image.png'
    matrix = fn.get_mtx_gb(path_image)
    start_Node = get_elements(matrix, 'ch')
    endNode = get_elements(matrix, 'sT')

    Board = Graph()

    # Board.edges =


# def get_edges(matrix):




def main(argv):
    """
        EJEMPLO DE USO:
          python deber2.py 1 1
          argumento 1: Nodo inicial
          argumento 2: Algoritmo a usar>>
          1>> A*
          2>> DFS
          3>> BFS
          4>> Greddy
        """
    if len(argv) != 2:
        print(main.__doc__)
    else:
        startNode = argv[0]
        algorithm = argv[1]

        # Always Bucarest due to heuristic is given for this end city
        endNode = '47'
        Maze = Graph()  # Builds Maze Graph

        # Adding edges (adjacency list)
        Maze.edges = {
            '1': ['2', '6'],
            '2': ['1', '3'],
            '3': ['2', '4'],
            '4': ['3', '5'],
            '5': ['4'],
            '6': ['1', '7'],
            '7': ['6', '8'],
            '8': ['7', '9'],
            '9': ['8', '10', '13'],
            '10': ['9', '11'],
            '11': ['10', '12'],
            '12': ['11'],
            '13': ['9', '14'],
            '14': ['13', '15'],
            '15': ['14', '16'],
            '16': ['15', '17'],
            '17': ['16', '18'],
            '18': ['17', '19'],
            '19': ['18', '20', '22'],
            '20': ['19', '21'],
            '21': ['20'],
            '22': ['19', '23', '48'],
            '23': ['22', '24'],
            '24': ['23', '25'],
            '25': ['24', '26'],
            '26': ['25', '27'],
            '27': ['26', '28', '29'],
            '28': ['27', '30'],
            '29': ['27', '30', '34'],
            '30': ['28', '29', '31'],
            '31': ['30', '32'],
            '32': ['31', '33'],
            '33': ['32'],
            '34': ['29', '35'],
            '35': ['34', '36'],
            '36': ['35', '37'],
            '37': ['36', '38'],
            '38': ['37', '39'],
            '39': ['38', '40'],
            '40': ['39', '41', '43'],
            '41': ['40', '42'],
            '42': ['41'],
            '43': ['40', '44'],
            '44': ['43', '45', '46'],
            '45': ['44'],
            '46': ['44', '47'],
            '47': ['46'],
            '48': ['22', '49'],
            '49': ['48', '50'],
            '50': ['49']
        }

        if argv[0] not in Maze.edges.keys():
            print("Nodo no existe")
            return

        # Define screen and World Wide coordinates
        screen = turtle.Screen()
        screen.setup(600, 327)
        turtle.setworldcoordinates(0, 0, 600, 327)

        # Use image as backgroud (image is 600x327 pixels)
        turtle.bgpic('Picture1.png')

        # Get image anchored to left-bottom corner (sw: southwest)
        canvas = screen.getcanvas()
        canvas.itemconfig(screen._bgpic, anchor="sw")

        if algorithm == '1':
            # Building aStar path of parents
            parents = a_Star(Maze, startNode, endNode)
        if algorithm == '2':
            parents = dfs_non_recursive(Maze, startNode, endNode)
        if algorithm == '3':
            parents = bfs_non_recursive(Maze, startNode, endNode)
        if algorithm == '4':
            parents = greedy(Maze, startNode, endNode)
            # parents = dfs(visited, Maze, '1',parentsDfs)

        # Calculating and printing the shortest path
        shortest_path = path_from_Origin(startNode, endNode, parents)
        print(shortest_path)

        # Calculating the cost of the shortest path
        cost_tsp = cost_of_Path(shortest_path, Maze)

        # Draw shortest path
        for ni in shortest_path:
            draw_square(ni, color="salmon")

        # Animate shortest path agent and include cost
        tsp = turtle.Turtle(shape="square")

        for i, ni in enumerate(shortest_path):
            draw_square(ni, color="dodger blue", ts=tsp, text=cost_tsp[i])

        turtle.exitonclick()  # Al hacer clic sobre la ventana grafica se cerrara

#
# if __name__ == "__main__":
#     import sys
#
#     main(sys.argv[1:])