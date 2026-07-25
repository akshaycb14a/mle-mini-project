from collections import deque


class SlidingWindow:

    def __init__(self, size=30):

        self.temperature = deque(maxlen=size)

        self.vibration = deque(maxlen=size)

    def add(self, temperature, vibration):

        self.temperature.append(temperature)

        self.vibration.append(vibration)

    def ready(self):

        return len(self.temperature) == self.temperature.maxlen

if __name__ == "__main__":

    sw = SlidingWindow()

    for i in range(40):

        sw.add(i, i)

        print(
            i,
            sw.ready(),
            len(sw.temperature)
        )