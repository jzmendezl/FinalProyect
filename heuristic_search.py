import numpy as np

from tree import Node
import copy


def heuristic_search(s_start, s_end):
    success = False
    edge_nodes = []
    visit_nodes = []

    start_node = Node(s_start)
    edge_nodes.append(start_node)
    compare = 0
    textError = ''

    while len(edge_nodes) != 0 and (not success):
        # nodo es el Nodo padre inicialmente y edge_nodes llegaria a ser una cola
        # porque saca el primero que estuvo en la cola y asi sucesivamente

        node = edge_nodes.pop(0)
        visit_nodes.append(node)
        print('Vuelta = ', compare)

        if node.get_data() == s_end:
            success = True
            print('Cantidad de comparaciones: ', compare)
            return node
        else:
            compare += 1
            aux = node.get_data()
            row = 0
            column = 0
            x = len(aux)
            y = len(aux[0])

            for i in range(x):
                for j in range(y):
                    if aux[i][j] == 'ch':
                        row = i
                        column = j
                        break

            children_nodes = []

            pos_row, pos_col = get_ch_pos(node)

            # ejemplo
            # if row == column and (row + column) == 2:
            #     mat_aux = node_swap(copy.deepcopy(aux), row, column, 0, -1)
            #     node_left = Node(mat_aux)
            #     node_left.set_cost(node_cost(s_end, mat_aux))

            # Nuestro::: si el ch esta en la posicion inicial y se puede mover en todas direcciones
            # --- Cambiar segun el tablero
            if row == pos_row and column == pos_col:
                # Mueve el charachter a la izquierda
                mat_aux = node_swap(copy.deepcopy(aux), row, column, 0, -1)
                node_left = Node(mat_aux)
                node_left.set_cost(node_cost(s_end, mat_aux))

                # Mueve el charachter arriba
                mat_aux = node_swap(copy.deepcopy(aux), row, column, -1, 0)
                node_up = Node(mat_aux)
                node_up.set_cost(node_cost(s_end, mat_aux))

                # Mueve el charachter a la derecha
                mat_aux = node_swap(copy.deepcopy(aux), row, column, 0, 1)
                node_rigth = Node(mat_aux)
                node_rigth.set_cost(node_cost(s_end, mat_aux))

                # Mueve el charachter abajo
                mat_aux = node_swap(copy.deepcopy(aux), row, column, 1, 0)
                node_down = Node(mat_aux)
                node_down.set_cost(node_cost(s_end, mat_aux))

                if not node_left.on_list(visit_nodes):
                    children_nodes.append(node_left)

                if not node_up.on_list(visit_nodes):
                    children_nodes.append(node_up)

                if not node_rigth.on_list(visit_nodes):
                    children_nodes.append(node_rigth)

                if not node_down.on_list(visit_nodes):
                    children_nodes.append(node_down)

                # Buscamos el nodo con mayor cantidad de elementos en commune
                chosen_node = search_chosen_node(children_nodes)
                edge_nodes.append(chosen_node)

                try:
                    node.set_children([chosen_node])
                except:
                    textError = 'Sin Nodos, No Solucionado'
                    break

            # si el ch esta en los laterales --- arriba y abajo
            elif row == 1 or column == 1:
                if row == 1:
                    node_up = node_swap(copy.deepcopy(aux), row, column, -1, 0)
                    node_down = node_swap(copy.deepcopy(aux), row, column, 1, 0)

                    if row > column:
                        node_center = node_swap(copy.deepcopy(aux), row, column, 0, 1)
                    else:
                        node_center = node_swap(copy.deepcopy(aux), row, column, 0, -1)

                    node_A = Node(node_up)
                    node_A.set_cost(node_cost(s_end, node_up))

                    node_B = Node(node_down)
                    node_B.set_cost(node_cost(s_end, node_down))

                    node_C = Node(node_center)
                    node_C.set_cost(node_cost(s_end, node_center))

                    if not node_A.on_list(visit_nodes):
                        children_nodes.append(node_A)

                    if not node_B.on_list(visit_nodes):
                        children_nodes.append(node_B)

                    if not node_C.on_list(visit_nodes):
                        children_nodes.append(node_C)

                else:
                    node_left = node_swap(copy.deepcopy(aux), row, column, 0, -1)
                    node_rigth = node_swap(copy.deepcopy(aux), row, column, 0, 1)

                    if column > row:
                        node_center = node_swap(copy.deepcopy(aux), row, column, 1, 0)
                    else:
                        node_center = node_swap(copy.deepcopy(aux), row, column, -1, 0)

                    node_L = Node(node_up)
                    node_L.set_cost(node_cost(s_end, node_up))

                    node_C = Node(node_down)
                    node_C.set_cost(node_cost(s_end, node_down))

                    node_R = Node(node_center)
                    node_R.set_cost(node_cost(s_end, node_center))

                    if not node_L.on_list(visit_nodes):
                        children_nodes.append(node_L)

                    if not node_C.on_list(visit_nodes):
                        children_nodes.append(node_C)

                    if not node_R.on_list(visit_nodes):
                        children_nodes.append(node_R)

                chosen_node = search_chosen_node(children_nodes)
                edge_nodes.append(chosen_node)

                try:
                    node.set_children([chosen_node])
                except:
                    textError = 'Sin Nodos, No Solucionado'
                    break
            # ch en las esquinas superiores
            else:
                if row <= column and (row + column) < len(aux):
                    node_down = node_swap(copy.deepcopy(aux), row, column, 1, 0)
                    if column <= row:
                        node_hoz = node_swap(copy.deepcopy(aux), row, column, 0, 1)
                    else:
                        node_hoz = node_swap(copy.deepcopy(aux), row, column, 0, -1)

                    node_U = Node(node_down)
                    node_U.set_cost(node_cost(s_end, node_down))

                    node_H = Node(node_hoz)
                    node_H.set_cost(node_cost(s_end, node_hoz))

                    if not node_U.on_list(visit_nodes):
                        children_nodes.append(node_U)

                    if not node_H.on_list(visit_nodes):
                        children_nodes.append(node_H)

                # ch en las esquinas inferiores
                else:
                    node_up = node_swap(copy.deepcopy(aux), row, column, -1, 0)

                    if column < row:
                        node_down = node_swap(copy.deepcopy(aux), row, column, 0, 1)
                    else:
                        node_down = node_swap(copy.deepcopy(aux), row, column, 0, -1)

                node_UU = Node(node_up)
                node_UU.set_cost(node_cost(s_end, node_up))

                node_UUU = Node(node_down)
                node_UUU.set_cost(node_cost(s_end, node_down))

                if not node_UU.on_list(visit_nodes):
                    children_nodes.append(node_UU)

                if not node_UUU.on_list(visit_nodes):
                    children_nodes.append(node_UUU)

            chosen_node = search_chosen_node(children_nodes)
            edge_nodes.append(chosen_node)

            try:
                node.set_children([chosen_node])
            except:
                textError = 'Sin Nodos, No Solucionado'
                break

    return textError


def node_swap(node, pos_x, pos_y, operX, operY):
    node[pos_x][pos_y] = node[pos_x + operX][pos_y + operY]
    node[pos_x + operX][pos_y + operY] = 0
    return node


def node_cost(parent_node, child_node):
    cost = 0
    for i in range(len(parent_node)):
        for j in range(len(parent_node[0])):
            if parent_node[i][j] == child_node[i][j]:
                cost += 1
    return cost


def search_chosen_node(children_node_list):
    cost = 0
    efficient_node = []
    for element in children_node_list:
        if element.get_cost() > cost:
            efficient_node = element
            cost = element.get_cost()
        elif element.get_cost() == cost:
            list1 = get_ch_pos(element.get_data())
            x1 = list1[0]
            y1 = list1[1]
            # verifica cuantos movimientos posibles tiene hacia todas las direcciones
            if (x1 + y1) >= 2 and x1 > 0 and y1 > 0 and x1 < 3 and y1 < 3:
                efficient_node = element

    return efficient_node


def get_ch_pos(matrix):
    pos = np.where(matrix == 'ch')
    x, y = pos[0].tolist()[0], pos[1].tolist()[0]
    return x, y
