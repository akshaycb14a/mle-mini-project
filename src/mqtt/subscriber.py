import json

import paho.mqtt.client as mqtt

BROKER = "localhost"
PORT = 1883
TOPIC = "logiedge/truck001/sensors"


def on_connect(client, userdata, flags, rc, properties=None):
    print("Connected")

    client.subscribe(TOPIC)


def on_message(client, userdata, msg):

    data = json.loads(msg.payload.decode())

    print("=" * 60)
    print(f"Truck       : {data['truck_id']}")
    print(f"Time        : {data['timestamp']}")
    print(f"Temperature : {data['temperature']} °C")
    print(f"Vibration   : {data['vibration']}")
    print(f"Door Open   : {data['door_open']}")
    print(f"Status      : {data['status']}")
    status = data["status"]

    if status == "critical":
        print("🚨 CRITICAL ALERT - Immediate action required!")

    elif status == "warning":
        print("⚠️ WARNING - Check refrigeration system.")

    else:
        print("✅ System operating normally.")


client = mqtt.Client(
    mqtt.CallbackAPIVersion.VERSION2
)

client.on_connect = on_connect

client.on_message = on_message

client.connect(BROKER, PORT)

client.loop_forever()