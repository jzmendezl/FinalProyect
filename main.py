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


def goTo(Matrix, currentPosition, objetivos, distancias, LIM_MOV, Result):

    row = copy.deepcopy(currentPosition[0])
    col = copy.deepcopy(currentPosition[1])
    Up = None
    Down = None
    Left = None
    Right = None

    if Matrix[row][col] == "sT" and objetivos[0] == [row, col]:
        return copy.deepcopy(Result)

    # TODO Verificar si ya estoy en uno de los objetivos:
    if Matrix[row][col] in ["d"] and ([row, col] in objetivos):
        toRemove = objetivos.index([row, col])
        objetivos.pop(toRemove)
        distancias.pop(toRemove)

        # Actualizar matrix
        Matrix[row][col] = "sP"

    nextMatrix = copy.deepcopy(Matrix)

    if (Matrix[row][col] != "sT" and len(objetivos) == 0):
        stairs = get_elements(matrix, 'sT')
        # Ir a la escalera
        objetivos.append(stairs[0])
        distancias.append(euclidian([row, col], stairs))
    try:
        distance = min(distancias)
        nearest_objetive = objetivos[distancias.index(min(distancias))]
    except:
        return "fail"

    # Movimiento hacia abajo
    nextPosition = [row + 1, col]
    if isMovValid(nextMatrix[row + 1][col], nextPosition, nearest_objetive, distance) and Result[-1] != "up":
        nextPosition = [row + 1, col]
        nextResult = copy.deepcopy(Result)
        nextResult.append("down")

        nextDistances = euclidian_dist(nextPosition, objetivos)
        Down = goTo(nextMatrix, nextPosition, objetivos,
                    nextDistances, LIM_MOV - 1, nextResult)

    # Movimiento hacia arriba
    nextPosition = [row - 1, col]
    if isMovValid(nextMatrix[row - 1, col], nextPosition, nearest_objetive, distance) and Result[-1] != "down":

        # TODO calcular distancias
        nextPosition = [row - 1, col]
        nextResult = copy.deepcopy(Result)
        nextResult.append("up")

        nextDistances = euclidian_dist(nextPosition, objetivos)
        Up = goTo(nextMatrix, nextPosition, objetivos,
                  nextDistances, LIM_MOV - 1, nextResult)

    # Movimiento hacia derecha
    nextPosition = [row, col + 1]
    if isMovValid(nextMatrix[row][col + 1], nextPosition, nearest_objetive, distance) and Result[-1] != "left":
        nextDistances = euclidian_dist(nextPosition, objetivos)
        nextResult = copy.deepcopy(Result)
        nextResult.append("right")
        Right = goTo(nextMatrix, nextPosition, objetivos,
                     nextDistances, LIM_MOV - 1, nextResult)

    # Movimiento hacia derecha
    nextPosition = [row, col - 1]
    if isMovValid(nextMatrix[row][col - 1], nextPosition, nearest_objetive, distance) and Result[-1] != "right":
        nextDistances = euclidian_dist(nextPosition, objetivos)
        nextResult = copy.deepcopy(Result)
        nextResult.append("left")
        Left = goTo(nextMatrix, nextPosition, objetivos,
                    nextDistances, LIM_MOV - 1, nextResult)

    Resultados = []
    if Up is not None:
        return Up
    elif Down is not None:
        return Down
    elif Left is not None:
        return Left
    elif Right is not None:
        return Right
    return ["No hay caminos"]


def isMovValid(Mov, next_position, nearest_objetive, distance):

    if (Mov not in ["NF", "w1", "w2"]) and (euclidian(next_position, nearest_objetive) <= distance):
        return True
    return False


def euclidian(origen, destino):
    return np.sqrt(np.sum(np.square(np.array(origen) - np.array(destino))))


[1, 2, 0.5, 2]
# aux = get_elements(matrix, 'd')
# print(aux)


diams = get_elements(matrix, 'd')
keys = get_elements(matrix, 'k')
rocks = get_elements(matrix, 'r')
holes = get_elements(matrix, 'ho')
doorRs = get_elements(matrix, 'dR')
doorMs = get_elements(matrix, 'dM')
doorMs = get_elements(matrix, 'dM')
stairs = get_elements(matrix, 'sT')

print(diams)
"""
print(keys)
print(rocks)
print(holes)
print(doorRs)
print(doorMs)
print(stairs[0])
"""
print(get_ch_pos(matrix))
print(get_movements(matrix))

dist = euclidian_dist(get_ch_pos(matrix), diams)
print(dist)

Resultado = goTo(
    matrix,
    [get_ch_pos(matrix)[0], get_ch_pos(matrix)[1]],
    diams,
    dist, get_movements(matrix),
    [""])
print("EL resultado")
print("Resultado")
Resultado.pop(0)
time.sleep(2)

for movement in Resultado:
    fn.press_and_release(movement, pressed=1)

# fn.press_and_release('right', pressed=5)
# fn.press_and_release('down', pressed=3)
# fn.press_and_release('left', pressed=5)
# fn.press_and_release('down', pressed=2)
# fn.press_and_release('right', pressed=1)
# fn.press_and_release('down', pressed=1)
# fn.press_and_release('right', pressed=4)
# fn.press_and_release('down', pressed=1)
