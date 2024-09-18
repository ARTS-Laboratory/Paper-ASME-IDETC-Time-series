
class Heap():
    """ """
    def __init__(self, init_size=1):
        capacity = self.set_capacity(init_size)
        head = None

    def max_heapify(self):
        raise NotImplementedError()

    def get_max(self):
        raise NotImplementedError()

    def insert(self, item):
        raise NotImplementedError()

    def remove(self, item):
        raise NotImplementedError()

    def swap(self, item_1, item_2):
        raise NotImplementedError()

    def remove_last_n(self, n):
        raise NotImplementedError()

    def remove_under_threshold(self, threshold):
        raise NotImplementedError()
