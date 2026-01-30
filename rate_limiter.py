import time
from functools import wraps

class RateLimiter:
    def __init__(self, calls_per_second):
        self.interval = 1.0 / calls_per_second
        self.last_call_time = 0

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - self.last_call_time
            if elapsed < self.interval:
                time.sleep(self.interval - elapsed)
            self.last_call_time = time.time()
            return func(*args, **kwargs)
        return wrapper

# You can instantiate this with a specific rate, e.g., 1 call per second
# rate_limit_one_per_second = RateLimiter(1)
# rate_limit_five_per_minute = RateLimiter(5/60)
