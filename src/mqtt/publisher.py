import os
import json
import time

import paho.mqtt.client as mqtt

from src.simulator.sensor_simulator import SensorSimulator



BROKER = os.getenv("MQTT_BROKER", "localhost")
PORT = 1883
TOPIC = "logiedge/truck001/sensors"

client = mqtt.Client(
    mqtt.CallbackAPIVersion.VERSION2
)

client.connect(BROKER, PORT)
client.loop_start()

simulator = SensorSimulator(
    truck_id="TRUCK_001",
    mode="random"
)

print("Publishing simulated sensor data...\n")

while True:

    reading = simulator.generate_reading()

    info = client.publish(TOPIC, json.dumps(reading))
    info.wait_for_publish()

    print(reading)

    time.sleep(1)