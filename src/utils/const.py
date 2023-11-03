class MetaConst(type):
    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        raise TypeError(f"Can't change the value `{value}` of a constant `{key}`")


class Const(object, metaclass=MetaConst):
    def __getattr__(self, item):
        return self[item]

    def __setattr__(self, key, value):
        raise TypeError(f"Can't change the value `{value}` of a constant `{key}`")
