import time
import pyautogui as robot
import numpy as np
from numpy import cumsum
import cv2


def open_nav(url_game):
    robot.hotkey('win', 's')
    robot.write('brave')
    robot.hotkey('return')
    time.sleep(1)
    # robot.hotkey('ctrl', 't')
    robot.hotkey('esc')
    robot.hotkey('f6')
    robot.write(url_game)
    robot.hotkey('return')


def get_img(image_board, image_save, image_name):
    robot.moveTo(image_board)
    robot.rightClick()
    robot.keyDown('rigth')
    time.sleep(0.03)
    robot.keyUp('left')
    time.sleep(0.03)
    robot.moveTo(image_save)
    time.sleep(0.03)
    robot.leftClick()
    time.sleep(0.03)
    robot.hotkey('alt', 'm')
    time.sleep(0.03)
    robot.write(image_name)
    time.sleep(0.03)
    robot.hotkey('return')
    time.sleep(0.03)


def get_bank_img():
    button = cv2.imread('image/blocks/grises/button.png', 0)
    character = cv2.imread('image/blocks/grises/character.png', 0)
    diamond = cv2.imread('image/blocks/grises/diamond.png', 0)
    doorMetal = cv2.imread('image/blocks/grises/doorMetal.png', 0)
    doorRock = cv2.imread('image/blocks/grises/doorRock.png', 0)
    fertilityIdol = cv2.imread('image/blocks/grises/fertilityIdol.png', 0)
    freeSpace = cv2.imread('image/blocks/grises/freeSpace.png', 0)
    hat = cv2.imread('image/blocks/grises/hat.png', 0)
    hole = cv2.imread('image/blocks/grises/hole.png', 0)
    key = cv2.imread('image/blocks/grises/key.png', 0)
    magma = cv2.imread('image/blocks/grises/magma.png', 0)
    rock = cv2.imread('image/blocks/grises/rock.png', 0)
    spikes = cv2.imread('image/blocks/grises/spikes.png', 0)
    stair = cv2.imread('image/blocks/grises/stair.png', 0)
    wall1 = cv2.imread('image/blocks/grises/wall1.png', 0)
    wall2 = cv2.imread('image/blocks/grises/wall2.png', 0)

    return np.array(
        [button, character, diamond, doorMetal, doorRock, fertilityIdol, freeSpace, hat, hole, key, magma, rock, spikes,
         stair, wall1, wall2])


def get_mtx_gb(path_image):
  img_rgb = cv2.imread(path_image)
  img_gray = cv2.imread(path_image, 0)

  bankImages = get_bank_img()

  #block level
  tilesImage = np.array([img_gray[x:x+64,y:y+64] for x in range(0,img_gray.shape[0],64) for y in range(0,img_gray.shape[1],64)])

  #clasification game board
  classificationImage = np.array([""]*150, dtype = object)

  #name blocks/grises
  letterTiles = np.array(['b','ch','d','dM','dR','fI','fS','ha','ho','k','m','r','sP','sT','w1','w2'],dtype = object)

  #Compare
  for i in range(0, len(tilesImage)):
    for j in range(0, len(bankImages)):
      errorL2 = cv2.norm(bankImages[j], tilesImage[i], cv2.NORM_L2 )
      similarity = 1 - errorL2 / ( 64 * 64 )

      #restriction
      if(similarity>=0.7):
        classificationImage[i] = letterTiles[j]

    #restriction
    if(classificationImage[i]==""):
      classificationImage[i] = 'NF'

  return classificationImage.reshape(15,10)


def print_mtx_gb(matrix):
  s = [[str(e) for e in row] for row in matrix]
  lens = [max(map(len, col)) for col in zip(*s)]
  fmt = '  '.join('{{:{}}}'.format(x) for x in lens)
  table = [fmt.format(*row) for row in s]
  print('\n'.join(table))


def send_click(pos, clicks=1):
    robot.moveTo(pos)
    robot.click(clicks=clicks)


def press_and_release(key, wait=0.03, pressed=1):
    for i in range(pressed):
        robot.keyDown(key)
        time.sleep(wait)
        robot.keyUp(key)
        time.sleep(wait)


class Graph:
    def __init__(self):
        self.edges = {}
        self.weights = {}

        def neigbors(self, id):
            return self.edges[id]


def cost_path(path, graph):
    cum_costs = [0]
    for i in range(len(path) -1):
        cum_costs += [1]
    return cumsum(cum_costs)


def findleastF(oL):
    """
    finds the node with least F in oL (queue)
    This is equivalent to build a priority queue
    """
    mF = min(getF(oL))
    for ni in range(len(oL)):
        if oL[ni][1] == mF:
            return oL.pop(ni)[0]


def getF(oL):
    """
    Return cost of queue F = C + H
    C: cost of route
    H: heuristic cost
    """
    return [i[1] for i in oL]

def pathFromOrigin(origin, n, parents):
    #Builds shortest path from search result (parents)
    if origin == n:
        return []

    path0 = [n]
    i = n
    while True:
        i = parents[i]
        path0.inert(0, i)
        if i == origin:
            return path0


def aStar(graph, start, goal):
    openL = []
    openL.append((start, 0))
    parents = {}
    costSoFar = {}
    parents[start] = None
    costSoFar[start] = 0

    while bool(len(openL)):
        current = findleastF(openL)
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


def get_blockImages(setTile):
    img_rgb = cv2.imread(setTile)
    img_gray = cv2.imread(setTile, 0)

    tilesImage = np.array([img_gray[x:x + 64, y:y + 64] for x in range(0, img_gray.shape[0], 64) for y in
                           range(0, img_gray.shape[1], 64)])