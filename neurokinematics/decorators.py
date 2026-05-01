""" 
Simple decorator function for neurokinematics/data/session.py to provide output
"""

from functools import wraps

BOLD = "\033[1m"
DIM = "\033[2m"

YELLOW = "\033[93m"
BLUE = "\033[94m"
GREEN = "\033[92m"

RESET = "\033[0m"


def log_call(label=None, type='run'):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):

            name = label or func.__qualname__

            labels = {
                'run': 'RUNNING:',
                'load': 'LOADING:',
                'process': 'PROCESSING:',
                'plot': 'PLOTTING:'
            }
            print(f"\n{BOLD}{labels.get(type, 'RUNNING:')}{RESET} {name}...\n")

            result = func(*args, **kwargs)
            if isinstance(result, dict) and "exists" in result:
                if result['exists']:
                    print(f"{YELLOW}{BOLD}PROCESS SKIPPED{RESET}: \n    Output already exists at {GREEN}{result['path']}{RESET}.\n    Use {DIM}'overwrite'{RESET} mode to overwrite output.")
            else:
                print('Finished.')
            return result
        return wrapper
    return decorator