from typing import Any


class Target:
    row = None
    col = None
    elem_type = None
    distance = None

    def __init__(self, row, col, elem_type, distance):
        self.row = row
        self.col = col
        self.elem_type = elem_type
        self.distance = distance

    def __repr__(self):
        return '[' + self.elem_type + ', ' + str(self.row) + ', ' + str(self.col) + ', ' + str(self.distance) + ']'

    def set_distance(self, distance):
        self.distance = distance

    def get_distance(self):
        return self.distance

    def set_row(self, row):
        self.row = row

    def get_row(self):
        return self.row

    def set_col(self, col):
        self.col = col

    def get_col(self):
        return self.col

    def set_elem_type(self, elem_type):
        self.elem_type = elem_type

    def get_elem_type(self):
        return self.elem_type