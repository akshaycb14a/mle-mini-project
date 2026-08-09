# mle-mini-project

Mini project for the course Machine Learning on the Edge.

## Team

Team-39

| Member | BITS ID |
| --- | --- |
| Jayanthi Kanaka Nayana | 2024AC05998 |
| Akshay Pandu | 2024AC05981 |
| Tony Abraham Mammen | 2024AC05596 |
| A. C. Vikramathithan | 2024AD05147 |

## About this repo

This repository contains the code and generated outputs for the ML-on-Edge mini project.

We did not follow the template repository structure given in the problem statement. The project was built piece by piece while implementing the assignment requirements, and we kept the resulting structure instead of reorganizing everything at the end.

The web UI in this repository is only for demo and video presentation purposes. It helps show live predictions, sensor runs, and generated reports in one place. It is not a required part of the core assignment implementation and does not need to be included in the report.

## Project summary

This project simulates a cold-chain truck monitoring setup and runs edge inference on streaming sensor data. The main focus is to show an end-to-end ML-on-Edge workflow:

- sensor data generation over MQTT
- online preprocessing and feature extraction
- model training and TensorFlow Lite conversion
- live edge inference using saved normalization statistics
- drift monitoring using PSI
- deployment support using Docker and Ansible

The system works with temperature and vibration readings and classifies the stream into the project classes during runtime. The repository also includes benchmarking and deployment artefacts required for the assignment.

## What is in this repo

This repo includes:

- sensor simulation over MQTT
- preprocessing with moving average and sliding windows
- edge inference with TensorFlow Lite
- PSI drift monitoring
- model benchmark and quantization experiments
- Docker and Ansible deployment

## End-to-end flow

The project flow is:

1. Simulate truck sensor data and publish it to MQTT.
2. Apply 5-sample moving average filtering.
3. Build 30-second sliding windows with 10-second step.
4. Extract six features from each ready window:
   - temperature mean
   - temperature standard deviation
   - temperature rate of change
   - vibration RMS
   - vibration peak
   - vibration kurtosis
5. Normalize features using `training_stats.npy` generated from clean normal data.
6. Run TFLite inference on the edge.
7. Publish inference results back to MQTT and save logs locally.
8. Monitor confidence drift using PSI.

## Main components

- `src/simulator/` - sensor simulator with `none`, `temp_drift`, `vibration`, and `combined` anomaly modes
- `src/preprocessing/` - moving average, sliding window logic, and feature extraction
- `src/training/` - training, evaluation, training-stat generation, and normalization experiment
- `src/deployment/` - TFLite export, PTQ, pruning flow, and benchmarking
- `src/mqtt/` - live subscriber and MQTT inference publishing
- `src/monitoring/` - PSI drift monitoring
- `src/webapp/` - optional demo-only web interface
- `scripts/` - helper scripts for demo and local runs

## Key assignment outputs

The repository generates and uses the following outputs:

- `data/models/training_stats.npy` - normalization statistics from clean normal data
- `data/tflite/model_fp32.tflite` - FP32 deployment model
- `data/tflite/model_int8.tflite` - full INT8 PTQ model
- `data/tflite/model_pruned_int8.tflite` - pruning + full INT8 PTQ model
- `reports/normalization_experiment.json` and `reports/normalization_experiment.csv`
- `reports/benchmark_results.csv`
- `reports/pareto_chart.png`
- `reference_dist.json` - PSI reference confidence distribution

## Demo and run flow

For the demo, the project can be shown in this order:

1. Train and evaluate the model.
2. Generate the deployment models and benchmark them.
3. Start the live stack.
4. Run normal and anomaly sensor streams.
5. Show live inference and PSI behaviour.
6. Optionally show the web UI, Docker, and Ansible steps.

The shell scripts in `scripts/` are meant to reduce the number of manual commands during the demo.

## Main scripts

The main scripts are in `scripts/`:

```bash
./scripts/run_demo_slideshow.sh
./scripts/run_training_demo.sh
./scripts/run_web_app.sh
./scripts/start_live_stack.sh
./scripts/run_sensor_cycle.sh
./scripts/run_model_tasks.sh
./scripts/run_psi_demo.sh
./scripts/run_docker_cache_demo.sh
./scripts/run_ansible_demo.sh
./scripts/stop_live_stack.sh
```

Short purpose of each script:

- `run_training_demo.sh` - generate clean-data stats, train the model, and run evaluation
- `run_model_tasks.sh` - run normalization experiment, build TFLite models, and benchmark them
- `start_live_stack.sh` - start Mosquitto, inference subscriber, and PSI monitor
- `run_sensor.sh` - run the simulator with one anomaly mode
- `run_sensor_cycle.sh` - run all anomaly modes in sequence
- `run_psi_demo.sh` - run the PSI reference build and quick PSI demo
- `run_web_app.sh` - start the demo-only web UI
- `run_docker_cache_demo.sh` - show model-only Docker rebuild behaviour
- `run_ansible_demo.sh` - run the deployment playbook twice
- `stop_live_stack.sh` - stop the local live processes

## Environment and setup notes

To run the full project locally, the main requirements are:

- Python virtual environment with the packages from `requirements.txt`
- Docker Desktop or Docker Engine
- Mosquitto through Docker Compose in this repository
- Ansible for the deployment playbook

Some parts of the project are optional during normal use:

- The web UI is optional and only meant for presentation.
- Docker and Ansible are mainly for the deployment part of the assignment.
- If only the ML pipeline needs to be shown, training, model generation, live inference, and PSI are the important parts.

## Repository notes

- The project structure is the working structure that resulted from implementation, not a recreated template layout.
- The code was kept simple on purpose so the project remains easy to explain in a viva or screen demo.
- Generated reports, model files, and logs are stored inside the repository so the outputs are easy to inspect.

## Notes

- [Project overview](docs/PROJECT_OVERVIEW.md)
- [Demo commands](DEMO_COMMANDS.md)
- [Demo slideshow](docs/demo_slideshow.html)
