# mle-mini-project

Mini project for the course Machine Learning on the Edge.

## What is in this repo

This repo includes:

- sensor simulation over MQTT
- preprocessing with moving average and sliding windows
- edge inference with TensorFlow Lite
- PSI drift monitoring
- model benchmark and quantization experiments
- Docker and Ansible deployment

## Run

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
