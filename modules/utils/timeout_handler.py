"""
Timeout handler for distributed operations.

This module provides a context manager for handling timeouts in distributed operations.
"""

import signal
import time
from contextlib import contextmanager
from modules.utils.logger import warning, info, error

class TimeoutException(Exception):
    """Exception raised when a timeout occurs."""
    pass

@contextmanager
def timeout_handler(seconds=300, error_message="Operation timed out"):
    """
    Context manager for handling timeouts.
    
    Args:
        seconds: Timeout in seconds (default: 300)
        error_message: Error message to display when timeout occurs
        
    Raises:
        TimeoutException: When the operation times out
    """
    def _handle_timeout(signum, frame):
        raise TimeoutException(error_message)
    
    # Save the previous handler
    previous_handler = signal.getsignal(signal.SIGALRM)
    
    try:
        # Set the alarm
        signal.signal(signal.SIGALRM, _handle_timeout)
        signal.alarm(seconds)
        
        # Start time for logging
        start_time = time.time()
        
        # Yield control back to the caller
        yield
        
        # Operation completed successfully, log the time taken
        elapsed = time.time() - start_time
        info(f"Operation completed in {elapsed:.2f} seconds (timeout was {seconds} seconds)")
        
    finally:
        # Cancel the alarm and restore the previous handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
