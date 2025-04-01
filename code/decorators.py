from functools import wraps
import logging

# logger configuration
logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(lineno)d - %(message)s : ",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def handle_exception(func):
    """Decorator to handle class methods exceptions"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # error logging
            logger.error(f"Error in {func.__name__}: {e}")
            # return None or handle the error as needed
            print(f"An error has occurred in {func.__name__}: {e}")
            return None

    return wrapper
