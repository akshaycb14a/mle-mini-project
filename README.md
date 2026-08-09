# mle-mini-project

Mini project for the course Machine Learning on the Edge.

## Team

Team-39

- Jayanthi Kanaka Nayana - BITS ID: 2024AC05998
- Akshay Pandu - BITS ID: 2024AC05981
- Tony Abraham Mammen - BITS ID: 2024AC05596
- A. C. Vikramathithan - BITS ID: 2024AD05147

## About this repo

This repository contains the code and generated outputs for the ML-on-Edge mini project.

We did not follow the template repository structure given in the problem statement. The project was built piece by piece while implementing the assignment requirements, and we kept the resulting structure instead of reorganizing everything at the end.

The web UI in this repository is only for demo and video presentation purposes. It helps show live predictions, sensor runs, and generated reports in one place. It is not a required part of the core assignment implementation and does not need to be included in the report.

## What is in this repo

This repo includes:

- sensor simulation over MQTT
- preprocessing with moving average and sliding windows
- edge inference with TensorFlow Lite
- PSI drift monitoring
- model benchmark and quantization experiments
- Docker and Ansible deployment

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

## Notes

- [Project overview](docs/PROJECT_OVERVIEW.md)
- [Demo commands](DEMO_COMMANDS.md)
- [Demo slideshow](docs/demo_slideshow.html)
