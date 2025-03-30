import logging
import sys
import time

# Create logger
logger = logging.getLogger('ChordMini')
logger.setLevel(logging.INFO)

# Prevent propagation to root logger to avoid duplicate messages
logger.propagate = False

# Define custom formatter
class CustomFormatter(logging.Formatter):
    def format(self, record):
        timestamp = time.strftime('%m-%d %H:%M:%S.%3d', 
                                  time.localtime(record.created)) 
        level = record.levelname[0]  # Only first letter of level
        filename = record.filename
        lineno = record.lineno
        message = super().format(record)
        return f'I ChordMini {timestamp} {filename}:{lineno}] {message}'

# Create console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

# Add formatter to console handler
formatter = CustomFormatter()
console_handler.setFormatter(formatter)

# Add console handler to logger
logger.addHandler(console_handler)

# For compatibility with code using print-style logging
def info(message):
    """Log an info message"""
    logger.info(message)

def warning(message):
    """Log a warning message"""
    logger.warning(message)

def error(message):
    """Log an error message"""
    logger.error(message)

def debug(message):
    """Log a debug message"""
    logger.debug(message)

def logging_verbosity(verbose_level):
    """Set logging verbosity"""
    if verbose_level == 0:
        logger.setLevel(logging.WARNING)
    elif verbose_level == 1:
        logger.setLevel(logging.INFO)
    elif verbose_level >= 2:
        logger.setLevel(logging.DEBUG)

def is_debug():
    """
    Check if debug logging is enabled.
    This is a safer way to check logging level than directly accessing internal values.
    
    Returns:
        bool: True if debug logging is enabled, False otherwise
    """
    import logging
    try:
        # Try to get the logger instance and check its level
        root_logger = logging.getLogger()
        return root_logger.level <= logging.DEBUG
    except:
        # If anything fails, default to False
        return False
