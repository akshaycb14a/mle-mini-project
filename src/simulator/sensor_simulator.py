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
        self.mode_counter = 0
        self.mode_duration = 60

        if self.mode == "random":
            self.current_mode = random.choice(
                ["normal", "warning", "critical"]
            )
        else:
            self.current_mode = self.mode

    def update_mode(self):
        if self.mode != "random":
            self.current_mode = self.mode
            return

        if self.mode_counter >= self.mode_duration:
            self.current_mode = random.choice([
                "normal",
                "warning",
                "critical",
            ])
            self.mode_counter = 0

        self.mode_counter += 1

    def generate_reading(self):
        self.update_mode()
        mode = self.current_mode

        if mode == "normal":
            temperature = round(random.uniform(2.0, 8.0), 2)
            vibration = round(random.uniform(0.4, 1.2), 2)
            door = random.random() < 0.05

        elif mode == "warning":
            temperature = round(random.uniform(8.0, 12.0), 2)
            vibration = round(random.uniform(1.2, 2.5), 2)
            door = random.random() < 0.20

        elif mode == "critical":
            temperature = round(random.uniform(12.0, 20.0), 2)
            vibration = round(random.uniform(2.5, 5.0), 2)
            door = random.random() < 0.60

        else:
            raise ValueError(f"Unknown mode: {mode}")

        return {
            "timestamp": datetime.now().isoformat(),
            "truck_id": self.truck_id,
            "temperature": temperature,
            "vibration": vibration,
            "door_open": door,
            "status": mode,
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
