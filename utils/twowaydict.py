class TwoWayDict(dict):
    """Two way 1 to 1 mapping"""
    def __init__(self, seq=None, **kwargs):
        if seq is None:
            super(TwoWayDict, self).__init__(**kwargs)
        else:
            super(TwoWayDict, self).__init__(seq, **kwargs)
            for k, v in seq.items(): dict.__setitem__(self, v, k)
            for k, v in kwargs.items(): dict.__setitem__(self, v, k)
    def __setitem__(self, key, value):
        # Remove any previous connections with these values
        if key in self:
            del self[key]
        if value in self:
            del self[value]
        dict.__setitem__(self, key, value)
        dict.__setitem__(self, value, key)
    def __delitem__(self, key):
        dict.__delitem__(self, self[key])
        dict.__delitem__(self, key)
    def __len__(self):
        """Returns the number of connections"""
        return dict.__len__(self) // 2
    def pop(self, k, v=None):
        re=self.__getitem__(k)
        self.__delitem__(k)
        return re
    def update(self, iterable, **kwargs):
        for k, v in iterable.items(): self.__setitem__(v, k)
