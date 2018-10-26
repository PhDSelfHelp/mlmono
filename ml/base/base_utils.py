def all_subclasses(cls):
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in all_subclasses(c)])

def find_subclass_by_name(cls, subcls_name):
    for subcls in all_subclasses(cls):
        if subcls.__name__ == subcls_name:
            return subcls
    err_msg = 'Subclass of {} with name {} is not found!'
    err_msg = err_msg.format(cls.__name__, subcls_name)
    raise ValueError(err_msg)
