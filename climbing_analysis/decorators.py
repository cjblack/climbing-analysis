from functools import wraps

def log_call(label=None, type='run'):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            name = label or func.__qual__name__
            if type == 'run':
                func_type = 'RUNNING:'
            elif type == 'load':
                func_type = 'LOADING:'
            elif type=='plot':
                func_type = 'PLOTTING:'
            print(f'{func_type} {name}...')
            result = func(*args, **kwargs)
            print('FINISHED.')
            return result
        return wrapper
    return decorator