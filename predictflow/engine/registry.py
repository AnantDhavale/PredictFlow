# Global registry for actions
ACTIONS = {}

def register_action(func):
    """Decorator for registering workflow actions."""
    ACTIONS[func.__name__] = func
    return func
