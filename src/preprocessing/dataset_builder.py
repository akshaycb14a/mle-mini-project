import csv
from collections import deque
from pathlib import Path


class DatasetBuilder:

    def __init__(self, output_path):

        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.header_written = self.output_path.exists()

    def append(self, features, label):

        row = {
            **features,
            "label": label
        }

        with open(self.output_path, "a", newline="") as f:

            writer = csv.DictWriter(
                f,
                fieldnames=row.keys()
            )

            if not self.header_written:
                writer.writeheader()
                self.header_written = True

            writer.writerow(row)


class SlidingWindow:

    def __init__(self, size=30):

        self.temperature = deque(maxlen=size)
        self.vibration = deque(maxlen=size)
        self.door = deque(maxlen=size)
        self.status = deque(maxlen=size)

    def add(self, temperature, vibration, door_open, status):

        self.temperature.append(temperature)
        self.vibration.append(vibration)
        self.door.append(door_open)
        self.status.append(status)

    def ready(self):

        return len(self.temperature) == self.temperature.maxlen

if __name__ == "__main__":

    sw = SlidingWindow()

    for i in range(40):

        sw.add(i, i, False, "normal")

        print(
            i,
            sw.ready(),
            len(sw.temperature)
        )
