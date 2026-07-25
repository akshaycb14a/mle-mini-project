from collections import deque


class MovingAverageFilter:
    """
    Simple moving average filter.
    """

    def __init__(self, window_size=5):
        self.window_size = window_size
        self.values = deque(maxlen=window_size)

    def update(self, value):
        self.values.append(value)
        return sum(self.values) / len(self.values)

if __name__ == "__main__":

    filt = MovingAverageFilter()

    data = [4, 4.2, 4.1, 4.3, 4.0, 6.5, 4.2]

    for value in data:
        print(value, "->", round(filt.update(value), 2))