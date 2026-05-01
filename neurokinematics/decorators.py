""" 
Simple decorator function for neurokinematics/data/session.py
"""

from functools import wraps

# def log_call(label=None, type='run'):
#     def decorator(func):
#         @wraps(func)
#         def wrapper(*args, **kwargs):
#             name = label or func.__qual__name__
#             if type == 'run':
#                 func_type = 'Running:'
#             elif type == 'load':
#                 func_type = 'Loading:'
#             elif type == 'process':
#                 func_type = 'Processing.'
#             elif type=='plot':
#                 func_type = 'Ploting:'
#             print(f'{func_type} {name}...')
#             result = func(*args, **kwargs)
#             print('FINISHED.')
#             return result
#         return wrapper
#     return decorator

def log_call(label=None, type='run'):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            
            name = label or func.__qual__name

            labels = {
                'run': 'Running:',
                'load': 'Loading:',
                'process': 'Processing:',
                'plot': 'Plotting:'
            }
            print(f"{labels.get(type, 'Running:')} {name}...")

            result = func(*args, **kwargs)
            if isinstance(result, dict) and "status" in result:
                if result['status'] == 'exists':
                    print(f"STATUS: output already exists at {result['path']}. Use 'overwrite' mode to overwrite.")
            else:
                print('Finished.')
            return result
        return wrapper
    return decorator