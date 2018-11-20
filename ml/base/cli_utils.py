_PARSING_PRIORITY = [int, float, str]

def override_global_config(global_config, custom_conf_args, can_create=False):
    for arg in custom_conf_args:
        config_path, val = arg.split('=')
        val = _config_val_parse(val, arg)
        config_path = config_path.split('__')
        global_config.override_field(config_path, val, can_create)
    return global_config

def _config_val_parse(val, arg_full_string):
    """ The default parsing priority is [int, float, str]
    """
    for var_type in _PARSING_PRIORITY:
        try:
            return var_type(val)
        except:
            pass
    err_msg = "The full arg string {} can't be parsed in any of the types of {}." 
    err_msg = err_msg.format(arg_full_string, _PARSING_PRIORITY)
    raise ValueError(err_msg)
