import time

class Timer:
    """A simple timer for tracking elapsed time during training."""
    def __init__(self):
        self.start_time = None
        self.elapsed = 0.0

    def start(self):
        """Start the timer."""
        self.start_time = time.time()

    def stop(self):
        """Stop the timer and accumulate elapsed time."""
        if self.start_time is not None:
            self.elapsed += time.time() - self.start_time
            self.start_time = None
        return self.elapsed

    def reset(self):
        """Reset the timer."""
        self.start_time = None
        self.elapsed = 0.0

    def elapsed_time(self):
        """Return the current elapsed time without stopping the timer."""
        if self.start_time is not None:
            return self.elapsed + (time.time() - self.start_time)
        return self.elapsed

# Example usage for testing the Timer class.
if __name__ == '__main__':
    timer = Timer()
    timer.start()
    time.sleep(1.5)  # Simulate training work.
    print(f"Elapsed time: {timer.stop():.4f} seconds")
