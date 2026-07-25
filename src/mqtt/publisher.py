import json
import time

import paho.mqtt.client as mqtt

from src.simulator.sensor_simulator import SensorSimulator


BROKER = "localhost"
PORT = 1883
TOPIC = "logiedge/truck001/sensors"

client = mqtt.Client(
    mqtt.CallbackAPIVersion.VERSION2
)

client.connect(BROKER, PORT)

simulator = SensorSimulator(
    truck_id="TRUCK_001",
    mode="random"
)

print("Publishing simulated sensor data...\n")

while True:

    reading = simulator.generate_reading()

    client.publish(
        TOPIC,
        json.dumps(reading)
    )

    print(reading)

    time.sleep(1)