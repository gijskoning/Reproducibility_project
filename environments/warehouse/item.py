from numpy import ndarray


class Item():
    """
    A task on the factory floor
    """
    ID = 1  # to generate new task ids

    def __init__(self, item_id, pos:ndarray):
        """
        Initializes the item, places it at (0,0)
        """
        self._pos = pos
        self._id = item_id
        self._waiting_time = 0

    @property
    def get_id(self):
        """
        returns the task identifier
        """
        return self._id

    @property
    def get_position(self):
        """
        @return: (x,y) tuple with task position
        """
        return self._pos

    @property
    def get_waiting_time(self):
        """
        returns item's waiting time
        """
        return self._waiting_time

    def increase_waiting_time(self):
        """
        increases item's waiting time by one
        """
        self._waiting_time += 1
