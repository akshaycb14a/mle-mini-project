import json

import paho.mqtt.client as mqtt

from src.preprocessing.pipeline import PreprocessingPipeline

BROKER = "localhost"
PORT = 1883
TOPIC = "logiedge/truck001/sensors"

pipeline = PreprocessingPipeline()

def on_connect(client, userdata, flags, rc, properties=None):
    print("Connected")

    client.subscribe(TOPIC)


def on_message(client, userdata, msg):

    data = json.loads(msg.payload.decode())

    print("=" * 60)
    print(data)

    # features = pipeline.process(data)
    pipeline.process(data)

    # if features is not None:
    #     prediction = edge_model.predict(features)


client = mqtt.Client(
    mqtt.CallbackAPIVersion.VERSION2
)

client.on_connect = on_connect

client.on_message = on_message

client.connect(BROKER, PORT)

client.loop_forever()