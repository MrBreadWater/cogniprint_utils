class LazyDict:
    """A dictionary built on demand from an iterator."""
    def __init__(self, iterator):
        self._dict = {}
        self._iterator = iterator

    def __getitem__(self, key):
        if key in self:
            return self._dict[key]
        else:
            raise KeyError(key)

    def __contains__(self, key):
        while key not in self._dict:
            try:
                k, v = next(self._iterator)
            except StopIteration:
                return False
            
            self._dict[k] = v
            print(self._dict)
           
            if k == key:
                    return True

    def get(self, key, default=None):
        try:
            return self.__getitem__(key)
        except KeyError:
            return default
