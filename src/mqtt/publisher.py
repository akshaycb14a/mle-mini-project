import sys

from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.simulator.sensor_simulator import build_parser, run_mqtt_stream


def main():
    run_mqtt_stream(build_parser().parse_args())


if __name__ == "__main__":
    main()
