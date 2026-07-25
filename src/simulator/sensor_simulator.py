import random
import time
from datetime import datetime


class SensorSimulator:
    """
    Simulates refrigerated truck sensor readings.
    """

    def __init__(self, truck_id="TRUCK_001", mode="normal"):
        self.truck_id = truck_id
        self.mode = mode.lower()

    def generate_reading(self):

        if self.mode == "normal":
            temperature = round(random.uniform(2.0, 8.0), 2)
            vibration = round(random.uniform(0.4, 1.2), 2)
            door = random.random() < 0.05

        elif self.mode == "warning":
            temperature = round(random.uniform(8.0, 12.0), 2)
            vibration = round(random.uniform(1.2, 2.5), 2)
            door = random.random() < 0.20

        elif self.mode == "critical":
            temperature = round(random.uniform(12.0, 20.0), 2)
            vibration = round(random.uniform(2.5, 5.0), 2)
            door = random.random() < 0.60

        else:
            scenario = random.choice(
                ["normal", "warning", "critical"]
            )

            self.mode = scenario
            return self.generate_reading()

        return {
            "timestamp": datetime.now().isoformat(),
            "truck_id": self.truck_id,
            "temperature": temperature,
            "vibration": vibration,
            "door_open": door,
            "status": self.mode,
        }


def main():

    simulator = SensorSimulator(
        truck_id="TRUCK_001",
        mode="random"
    )

    while True:

        reading = simulator.generate_reading()

        print(reading)

        time.sleep(1)


if __name__ == "__main__":
    main()