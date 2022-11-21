class Node:
    parent = None
    children = None
    data = None
    cost = None

    def __init__(self, data, children=None):
        self.children = None
        self.parent = None
        self.data = data
        self.set_children(children)

    def get_data(self):
        return self.data

    def set_data(self, data):
        self.data = data

    def get_cost(self):
        return self.cost

    def set_cost(self, cost):
        self.cost = cost

    def set_children(self, children):
        self.children = children
        if self.children is not None:
            for child in self.children:
                child.parent = self

    def get_children(self):
        return self.children

    def get_parent(self):
        return self.parent

    def set_parent(self, parent):
        self.parent = parent

    def equal(self, node):
        if self.get_data() == node.get_data():
            return True
        else:
            return False

    def on_list(self, node_list):
        on_list = False
        for n in node_list:
            if self.equal(n):
                on_list = True
        return on_list
