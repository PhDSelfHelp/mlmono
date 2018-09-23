def find_subclass_by_name(cls, subcls_name):
    for subcls in cls.__subclasses__():
        if subcls.__name__ == subcls_name:
            return subcls
    err_msg = 'Subclass of {} with name {} is not found!'
    err_msg.format(cls.__name__, subcls_name)
    raise ValueError(err_msg)
    