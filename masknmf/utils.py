import sys
import datetime

def display(msg):
    """
    Printing utility that logs time and flushes.
    """
    tag = '[' + datetime.datetime.today().strftime('%y-%m-%d %H:%M:%S') + ']: '
    sys.stdout.write(tag + msg + '\n')
    sys.stdout.flush()


def flatten(mapping):
    """
    Flattens a nested dictionary assuming that there are no key collisions.
    """
    items = []
    for key, val in mapping.items():
        if isinstance(val, dict):
            items.extend(flatten(val).items())
        else:
            items.append((key, val))
    return dict(items)

def load_config(filename, required={}, default={}):
    """
    Loads user-provided yaml file containing parameters key-value pairs.
    Omitted optional parameter keys are filled with default values.

    Args:
        filename : string
            Full path + name of config 'yaml' file.

    Returns:
        params : dict
            Parameter key-value pairs
    """

    # Read yaml file into dict
    display(f"Loading config: {filename}...")
    with open(filename, 'r') as stream:
        params = yaml.safe_load(stream)

    # Ensure that required arguments have been provided
    display('Checking required fields...')
    for group, keys in required.items():
        if group not in params.keys():
            raise ValueError(f'Missing group {group} with required fields.')
        for key in keys:
            if key not in params[group].keys():
                raise ValueError(f'Missing required field {group}:{key}')

    # Insert default values for missing optional fields
    display('Inserting defaults for missing optional arguments')
    for group, group_params in default.items():
        if group not in params.keys():
            display(f"Using all defaults in group '{group}={group_params}'")
            params[group] = group_params
        else:
            for key, val in group_params.items():
                if key not in params[group].keys():
                    display(f"Using default '{group}:{key}={val}'")
                    params[group][key] = val

    display("Config file successfully loaded.")
    return flatten(params)


