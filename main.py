import time
import numpy as np
import funtions as fn
import heuristic_search

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

    for itemx in x:
        for itemy in y:
            aux.append([itemx, itemy])
    return aux

print('****')

aux = get_elements(matrix, 'd')
print(aux)
# x, y = get_elements(s_start, 'd')
# print(type(x), type(y))
# print(x[0], y[0])
print(get_ch_pos(matrix))



# fn.press_and_release('right', pressed=5)
# fn.press_and_release('down', pressed=3)
# fn.press_and_release('left', pressed=5)
# fn.press_and_release('down', pressed=2)
# fn.press_and_release('right', pressed=1)
# fn.press_and_release('down', pressed=1)
# fn.press_and_release('right', pressed=4)
# fn.press_and_release('down', pressed=1)
