import time

class PerformanceMonitor:
    def __init__(self):
        self._start = None

    def start(self):
        self._start = time.perf_counter()

    def stop(self):
        if self._start is None:
            raise RuntimeError("Timer has not been started.")
        elapsed = time.perf_counter() - self._start
        self._start = None
        return elapsed

    def measure(self, func, *args, **kwargs):
        self.start()
        result = func(*args, **kwargs)
        elapsed = self.stop()
        return result, elapsed

if __name__ == "__main__":
    pm = PerformanceMonitor()
    pm.start()
    time.sleep(0.2)
    print(f"Elapsed: {pm.stop():.4f} seconds")
