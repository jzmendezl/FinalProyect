import time
import numpy as np
import funtions as fn
import heuristic_search
import copy
url_game = "https://www.minijuegos.com/embed/diamond-rush"
chrome_pos = 1475, 1050
search_bar = 434, 70
focus_game = 700, 300
image_board = 1950, 500
image_save = 1992, 510
pos_name_image = 1940, 439
image_name = 'image'
path_image_start = 'image/image.png'
path_image_end = 'image/blocks/end_states/1.png'

# fn.open_nav(url_game)
# time.sleep(3)
# fn.get_img(image_board, image_save, image_name)
# time.sleep(2)


matrix = fn.get_mtx_gb(path_image_start)

s_start = matrix
s_end = fn.get_mtx_gb(path_image_end)

fn.print_mtx_gb(s_start)

print(s_start[0][2])


def get_ch_pos(matrix):
    pos = np.where(matrix == 'ch')
    x, y = pos[0].tolist()[0], pos[1].tolist()[0]
    return x, y


print('*************')


def get_elements(matrix, element):
    aux = []
    item = np.where(matrix == element)
    x, y = item[0].tolist(), item[1].tolist()

    for itemx, itemy in zip(x, y):
        aux.append([itemx, itemy])
    return aux


def get_movements(matrix):
    wall1 = np.where(matrix == 'w1')
    wall2 = np.where(matrix == 'w2')
    NF = np.where(matrix == 'NF')

    mov = len(wall1[0]) + len(wall2[0]) + len(NF[0])
    return 150 - mov


print('****')


def euclidian_dist(pos_ch, elem_list):
    dist = []
    for i in elem_list:
        dist.append(np.sqrt(np.sum(np.square(np.array(pos_ch) - np.array(i)))))
    return dist


def node_swap(node, pos_x, pos_y, operX, operY):
    node[pos_x][pos_y] = node[pos_x + operX][pos_y + operY]
    node[pos_x + operX][pos_y + operY] = 0
    return node


# Matrix, currentPosition, [objetivos], [distancias euclidianas], LIM_MOV
# diccionario: posicion

def getPath(Matrix, currentPosition, objetivos, distancias, LIM_MOV):
    # 1. obtener objetivo más cercano.
    nearest_objetive = objetivos.index(min(distancias))
    # 2. Ir al objetivo


def goTo(Matrix, currentPosition, objetivos, distancias, LIM_MOV, Result):

    row = currentPosition[0]
    col = currentPosition[1]

    print(Result)
    if Matrix[row][col] == "sT" and (len(objetivos) == 0):
        return Result
    if LIM_MOV == 0:
        return []

    # TODO Verificar si ya estoy en uno de los objetivos:
    if Matrix[row][col] == "d":
        toRemove = objetivos.index([row, col])
        objetivos.pop(toRemove)

    # Movimiento hacia arriba
    if isMovValid(Matrix[row - 1][col]) and Result[-1] != "down":
        # TODO calcular distancias
        print("en up")
        nextResult = copy.deepcopy(Result)
        nextResult.append("up")
        print(nextResult)
        print("---------")
        nextPosition = [row - 1, col]
        nextDistances = euclidian_dist(nextPosition, objetivos)
        Up = goTo(Matrix, nextPosition, objetivos,
                  nextDistances, LIM_MOV - 1, nextResult)
        return Up

    # Movimiento hacia abajo
    if isMovValid(Matrix[row + 1][col]) and Result[-1] != "up":
        nextPosition = [row + 1, col]
        nextDistances = euclidian_dist(nextPosition, objetivos)
        return [] + goTo(Matrix, nextPosition, objetivos,
                         nextDistances, LIM_MOV - 1, Result.append("down"))
    # Movimiento hacia derecha
    if isMovValid(Matrix[row][col + 1]) and Result[-1] != "left":
        nextPosition = [row, col + 1]
        nextDistances = euclidian_dist(nextPosition, objetivos)
        return goTo(Matrix, nextPosition, objetivos,
                    nextDistances, LIM_MOV - 1, Result.append("right"))

    # Movimiento hacia derecha
    if isMovValid(Matrix[row][col - 1]) and Result[-1] != "right":
        nextPosition = [row, col - 1]
        nextDistances = euclidian_dist(nextPosition, objetivos)
        return goTo(Matrix, nextPosition, objetivos,
                    nextDistances, LIM_MOV - 1, Result.append("left"))

    return ["No hay caminos"]


def isMovValid(Mov):
    return Mov != "NF" and Mov != "W1" and Mov != "W2"


[1, 2, 0.5, 2]
# aux = get_elements(matrix, 'd')
# print(aux)


def menor(lista):
    min = lista[0]
    for x in lista:
        if x < min:
            min = x
    return min


diams = get_elements(matrix, 'd')
keys = get_elements(matrix, 'k')
rocks = get_elements(matrix, 'r')
holes = get_elements(matrix, 'ho')
doorRs = get_elements(matrix, 'dR')
doorMs = get_elements(matrix, 'dM')
doorMs = get_elements(matrix, 'dM')
stairs = get_elements(matrix, 'sT')

print(diams)
print(keys)
print(rocks)
print(holes)
print(doorRs)
print(doorMs)
print(stairs[0])

print(get_ch_pos(matrix))
print(get_movements(matrix))

dist = euclidian_dist(get_ch_pos(matrix), diams)
print(dist)

print(type(goTo(
    matrix,
    [get_ch_pos(matrix)[0], get_ch_pos(matrix)[1]],
    diams,
    dist, get_movements(matrix),
    [""])))
# fn.press_and_release('right', pressed=5)
# fn.press_and_release('down', pressed=3)
# fn.press_and_release('left', pressed=5)
# fn.press_and_release('down', pressed=2)
# fn.press_and_release('right', pressed=1)
# fn.press_and_release('down', pressed=1)
# fn.press_and_release('right', pressed=4)
# fn.press_and_release('down', pressed=1)
