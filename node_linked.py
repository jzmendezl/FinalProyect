class Node:
    def __init__(self, data: str):
        self.data: str = data
        self.next: Node = None
        self.prev: Node = None