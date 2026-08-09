# Repo Layout

## Notes

- Team: Team-39
- The repository was not reorganized to match the template structure from the problem statement.
- The current layout is the result of building the solution step by step and keeping the working structure.
- The web UI is only for demo use and screen recording. It is not required as part of the report.

## Files

- `src/simulator/sensor_simulator.py` - sensor readings and MQTT publish
- `src/preprocessing/pipeline.py` - 5-sample moving average and 30-second window
- `src/preprocessing/feature_extractor.py` - 6 window features
- `src/inference/predictor.py` - loads the TFLite model and `training_stats.npy`
- `src/mqtt/subscriber.py` - live inference and MQTT inference output
- `src/monitoring/drift_monitor.py` - PSI check and `reference_dist.json`
- `src/deployment/build_deployment_models.py` - exports the 3 TFLite models
- `src/deployment/benchmark_models.py` - benchmarks the models
- `src/training/normalization_experiment.py` - compares correct stats and shifted stats
- `ansible/logibridge_deploy.yml` - deployment playbook
- `Dockerfile` - inference container build
- `docker-compose.yml` - local broker and app setup

## Scripts

- `scripts/run_demo_slideshow.sh` - serve the simple HTML slideshow for the video
- `scripts/run_training_demo.sh` - generate stats, train, and evaluate the model
- `scripts/run_web_app.sh` - start the web demo
- `scripts/start_live_stack.sh` - start Mosquitto, inference, PSI monitor
- `scripts/stop_live_stack.sh` - stop the live stack
- `scripts/run_sensor.sh` - run one simulator mode
- `scripts/run_sensor_cycle.sh` - run all simulator modes
- `scripts/run_model_tasks.sh` - run normalization, export, benchmark
- `scripts/run_psi_demo.sh` - rebuild PSI reference and run the drift demo
- `scripts/run_docker_cache_demo.sh` - show Docker model-only rebuild
- `scripts/run_ansible_demo.sh` - run the Ansible playbook twice

## Outputs

- `data/models/training_stats.npy`
- `data/tflite/model_fp32.tflite`
- `data/tflite/model_int8.tflite`
- `data/tflite/model_pruned_int8.tflite`
- `reference_dist.json`
- `reports/benchmark_results.csv`
- `reports/pareto_chart.png`
- `reports/normalization_experiment.csv`
- `reports/normalization_experiment.json`
