import time
import numpy as np
import funtions as fn
import heuristic_search
import copy
from more_itertools import locate
from object import Target

url_game = "https://www.minijuegos.com/embed/diamond-rush"
chrome_pos = 1475, 1050
search_bar = 434, 70
focus_game = 700, 300
image_board = 1950, 500
image_save = 1992, 510
pos_name_image = 1940, 439
image_name = 'image'
path_image_start = 'image/Niveles/3.png'
path_image_end = 'image/blocks/end_states/1.png'

# fn.open_nav(url_game)
# time.sleep(3)
# fn.get_img(image_board, image_save, image_name)
# time.sleep(2)


matrix = fn.get_mtx_gb(path_image_start)

s_start = matrix
s_end = fn.get_mtx_gb(path_image_end)

fn.print_mtx_gb(s_start)

# print(s_start[0][2])


def get_ch_pos(matrix):
    pos = np.where(matrix == 'ch')
    x, y = pos[0].tolist()[0], pos[1].tolist()[0]
    return x, y


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
        # dist.append(np.sqrt(np.sum(np.square(np.array(pos_ch) - np.array(i)))))
        dist.append(euclidian(pos_ch, i))
    return dist


def node_swap(node, pos_x, pos_y, operX, operY):
    node[pos_x][pos_y] = node[pos_x + operX][pos_y + operY]
    node[pos_x + operX][pos_y + operY] = 0
    return node

def goTo(Matrix, ch_pos, initial_target, targets, LM, Result):
# def goTo(Matrix, currentPosition, objetivos, distancias, LIM_MOV, Result):
    row = copy.deepcopy(ch_pos[0])
    col = copy.deepcopy(ch_pos[1])
    Up = None
    Down = None
    Left = None
    Right = None

    if Matrix[row][col] == "sT" \
            and targets[0].row == row \
            and targets[0].col == col \
            and len(targets) == 1:
        newResult = copy.deepcopy(Result)
        newResult.pop(0)
        return [copy.deepcopy(Matrix), [row, col], [], newResult]

    if initial_target.get_row() == row and initial_target.get_col() == col:
        for target in targets:
            if target.get_row() == initial_target.get_row() and target.get_col() == initial_target.get_col():
                targets.remove(target)
        newResult = copy.deepcopy(Result)
        newResult.pop(0)
        return [copy.deepcopy(Matrix),
                [row, col],
                update_targets(targets, [row, col]),
                newResult]

    for target in targets:
        if target.get_row() == row and target.get_col() == col:
            targets.remove(target)
            # Actualizar matrix
            Matrix[row][col] = "fS"
            break



    nextMatrix = copy.deepcopy(Matrix)
    # if nextMatrix[row][col] == 'sP':
    #     nextMatrix[row][col] = 'w1'

    if LM == 0:
        return [copy.deepcopy(Matrix), [row, col], targets, copy.deepcopy(Result)]

    # Movimiento hacia abajo
    nextPosition = [row + 1, col]
    if isMovValid(nextMatrix[row + 1][col],
                  nextPosition, [initial_target.row,
                                 initial_target.col],
                  initial_target.get_distance()) and Result[-1] != "up":
        nextResult = copy.deepcopy(Result)
        nextResult.append("down")

        initial_target.set_distance(euclidian(nextPosition, [initial_target.row, initial_target.col]))
        Down = goTo(nextMatrix, nextPosition, initial_target, targets, LM - 1, nextResult)

    # Movimiento hacia arriba
    nextPosition = [row - 1, col]
    if isMovValid(nextMatrix[row - 1][col],
                  nextPosition, [initial_target.get_row(),
                                 initial_target.get_col()],
                  initial_target.get_distance()) and Result[-1] != "down":
        nextResult = copy.deepcopy(Result)
        nextResult.append("up")

        initial_target.set_distance(euclidian(nextPosition, [initial_target.row, initial_target.col]))
        Up = goTo(nextMatrix, nextPosition, initial_target, targets, LM - 1, nextResult)

    # Movimiento hacia izquierda
    nextPosition = [row, col - 1]
    if isMovValid(nextMatrix[row][col - 1], nextPosition,
                  [initial_target.get_row(),
                   initial_target.get_col()],
                  initial_target.get_distance()) and Result[-1] != "right":
        nextResult = copy.deepcopy(Result)
        nextResult.append("left")
        Left = goTo(nextMatrix, nextPosition, initial_target, targets, LM - 1, nextResult)


# Movimiento hacia derecha
    nextPosition = [row, col + 1]
    if isMovValid(nextMatrix[row][col + 1], nextPosition,
                  [initial_target.get_row(),
                   initial_target.get_col()],
                  initial_target.get_distance()) and Result[-1] != "left":
        nextResult = copy.deepcopy(Result)
        nextResult.append("right")
        Right = goTo(nextMatrix, nextPosition, initial_target, targets, LM - 1, nextResult)



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
        # if (Mov not in ["NF", "w1", "w2"]) and ( <= distance):
        return True
    return False


# TODO: en caso de enpate en cuanto a minimo obtener todos los indices y coord de esos elementos y luego en una funcion
# que calcule los pasos y la devuelva y se retorna el de menos pasos y se escoge el indice de ese elemento el cual se
# pasa como atributo de la funcion goto y se actualiza cada vez que se obtenga un nuevo minimo.

# TODO: marcar como muro cuando pase por espina.

def get_board(image):
    return fn.get_mtx_gb(image)


def path_is_vaild(path, elem):
    if (np.where(path == elem)):
        return False
    return True


def desempate(matrix, elem):
    ls_elem = get_elements(matrix, elem)
    pos_diam = euclidian_dist(get_ch_pos(matrix), ls_elem)
    dup_elem = get_elem_dup(pos_diam)
    dup_index = list(locate(pos_diam, lambda x: x in dup_elem))
    coord_elem_dup = []

    for elem in dup_index:
        coord_elem_dup.append(ls_elem[elem])

    print('\n\n\n*********************************')
    print(ls_elem)
    print(pos_diam)
    print(dup_index)
    print(coord_elem_dup)


def get_elem_dup(ls):
    return [x for i, x in enumerate(ls) if i != ls.index(x)]


def euclidian(origen, destino):
    c = abs(origen[0] - destino[0]) + abs(origen[1] - destino[1])
    return float(c) + float(np.sqrt(np.sum(np.square(np.array(origen) - np.array(destino)))))


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

# print(diams)
"""
print(keys)
print(rocks)
print(holes)
print(doorRs)
print(doorMs)
print(stairs[0])
"""
# print(get_ch_pos(matrix))
# print(get_movements(matrix))
#
# dist = euclidian_dist(get_ch_pos(matrix), diams)
# print(dist)

# # Resultado = goTo(matrix, [get_ch_pos(matrix)[0], get_ch_pos(matrix)[1]], diams, dist, get_movements(matrix), [""])
# print("EL resultado")
# print(Resultado)
# Resultado.pop(0)
# time.sleep(2)

# for movement in Resultado:
#     fn.press_and_release(movement, pressed=1)


def walls():
    aux = []
    aux.append(get_elements(matrix, 'w1'))
    aux.append(get_elements(matrix, 'w2'))
    aux.append(get_elements(matrix, 'NF'))
    aux.append(get_elements(matrix, 'm'))
    return aux


# print(walls())

desempate(matrix, 'd')

print('***********************************')
# m2 = get_board('image/Niveles/2.png')
# fn.print_mtx_gb(m2)

# print(get_ch_pos(matrix))


def get_colum(matrix, column):
    return matrix[:, column]


def get_colum(matrix, column):
    return matrix[:, column]


def get_row(matrix, row):
    return matrix[row, :]


def get_target(matrix, ch_pos):
    elem_types = ['d', 'k', 'dR', 'dM', 'sT']
    target = []
    for elem_type in elem_types:
        target.extend(get_sub_target(
            get_elements(matrix, elem_type),
            elem_type,
            ch_pos))
    target.sort(key=lambda x: x.distance)
    return target


def get_sub_target(lista, elem_type, ch_pos):
    result = []
    for elem in lista:
        newTarget = Target(elem[0], elem[1], elem_type, euclidian(elem, ch_pos))
        result.append(newTarget)
    return result

def update_targets(targets, ch_pos):
    result = []
    for target in targets:
        target.set_distance(euclidian([target.get_row(), target.get_col()], ch_pos))
        result.append(target)

    result.sort(key=lambda x: x.distance)

    return result


print('***************---------**********************')

def solve(matrix, targets, acumlador, LM, ch_pos):
    Result = ['']
    for initial_target in targets:
        newMatrix = None
        newCh_pos = None
        newTargets = None
        newResult = None

        print('it',initial_target)
        response = goTo(matrix, ch_pos, initial_target, targets, LM, Result)
        print('reponse', response)
        if response is not None and len(response) == 4:
            newMatrix = response[0]
            # print(len(response))
            newCh_pos = response[1]
            newTargets = response[2]
            newResult = response[3]
        # # print('response', response)
        # # print('newMatrix', newMatrix)
        # print('newCh_pos', newCh_pos)
        # print('newTargets', newTargets)
        # print('newRes', newResult)
        # print('acm', acumlador)?
        if newTargets is not None and len(newTargets) == 0:
            return acumlador + newResult
        else:
            if newResult is not None:
                return solve(newMatrix, newTargets, acumlador + newResult, LM - 1, newCh_pos)

    return 'No path'

print(fn.print_mtx_gb(matrix))

path = solve(matrix,
             get_target(matrix, get_ch_pos(matrix)),
             [''],
             get_movements(matrix),
             get_ch_pos(matrix))

print('path', path)
time.sleep(2)
for movement in path:
    fn.press_and_release(movement, pressed=1)


