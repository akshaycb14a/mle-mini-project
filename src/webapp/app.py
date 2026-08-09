import csv
import json
import os
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from flask import Flask, jsonify, redirect, render_template, request, send_from_directory, url_for


ROOT_DIR = Path(__file__).resolve().parents[2]
DB_PATH = ROOT_DIR / "logiedge.db"
REPORTS_DIR = ROOT_DIR / "reports"
LOGS_DIR = ROOT_DIR / "logs"
VENV_PYTHON = ROOT_DIR / ".venv" / "bin" / "python"
SENSOR_SCRIPT = ROOT_DIR / "src" / "simulator" / "sensor_simulator.py"
SENSOR_LOG = LOGS_DIR / "web_sensor.log"
BENCHMARK_CSV = REPORTS_DIR / "benchmark_results.csv"
NORMALIZATION_JSON = REPORTS_DIR / "normalization_experiment.json"
REFERENCE_JSON = ROOT_DIR / "reference_dist.json"

app = Flask(__name__)

sensor_process = None
sensor_run = {
    "anomaly": None,
    "count": None,
    "interval": None,
    "last_error": None,
}

active_sensor_log = SENSOR_LOG


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def row_to_dict(row):
    if row is None:
        return None
    return dict(row)


def load_latest_data():
    telemetry = None
    inference = None
    alerts = []

    if DB_PATH.exists():
        conn = get_db_connection()
        try:
            telemetry = row_to_dict(
                conn.execute(
                    "SELECT * FROM telemetry ORDER BY id DESC LIMIT 1"
                ).fetchone()
            )
            inference = row_to_dict(
                conn.execute(
                    "SELECT * FROM inference_results ORDER BY id DESC LIMIT 1"
                ).fetchone()
            )
            alerts = [
                dict(row)
                for row in conn.execute(
                    "SELECT * FROM alerts ORDER BY id DESC LIMIT 5"
                ).fetchall()
            ]
        finally:
            conn.close()

    return telemetry, inference, alerts


def read_log_tail(path, limit=20):
    if not Path(path).exists():
        return []
    lines = Path(path).read_text(errors="ignore").splitlines()
    return lines[-limit:]


def get_sensor_log_path():
    global active_sensor_log

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    preferred = SENSOR_LOG
    try:
        preferred.write_text("")
        active_sensor_log = preferred
        return preferred
    except OSError:
        fallback = Path(tempfile.gettempdir()) / "mle_web_sensor.log"
        fallback.write_text("")
        active_sensor_log = fallback
        return fallback


def load_benchmark_rows():
    if not BENCHMARK_CSV.exists():
        return []
    with BENCHMARK_CSV.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_normalization_result():
    if not NORMALIZATION_JSON.exists():
        return None
    return json.loads(NORMALIZATION_JSON.read_text())


def load_reference_summary():
    if not REFERENCE_JSON.exists():
        return None
    payload = json.loads(REFERENCE_JSON.read_text())
    return {
        "sample_count": payload.get("sample_count"),
        "distribution": payload.get("distribution"),
        "generated_at": payload.get("generated_at"),
    }


def latest_psi_line():
    lines = read_log_tail(LOGS_DIR / "psi_monitor.log", limit=50)
    for line in reversed(lines):
        if "Current PSI:" in line or "DRIFT ALERT" in line or "recovered below 0.10" in line:
            return line
    return "PSI output not available yet"


def current_sensor_status():
    global sensor_process

    if sensor_process is None:
        return {"running": False, "pid": None, **sensor_run}

    poll_code = sensor_process.poll()
    if poll_code is not None:
        sensor_process = None
        last_line = ""
        log_tail = read_log_tail(active_sensor_log, limit=20)
        if log_tail:
            last_line = log_tail[-1]
        if not sensor_run.get("last_error") and poll_code != 0:
            sensor_run["last_error"] = last_line or f"sensor process exited with code {poll_code}"
        return {
            "running": False,
            "pid": None,
            **sensor_run,
            "exit_code": poll_code,
        }

    return {"running": True, "pid": sensor_process.pid, **sensor_run}


def start_sensor_run(anomaly, count, interval):
    global sensor_process

    if current_sensor_status()["running"]:
        raise RuntimeError("A sensor run is already active")

    sensor_log_path = get_sensor_log_path()

    command = [
        str(VENV_PYTHON if VENV_PYTHON.exists() else Path(sys.executable)),
        str(SENSOR_SCRIPT),
        "--anomaly",
        anomaly,
        "--count",
        str(count),
        "--interval",
        str(interval),
    ]

    log_handle = sensor_log_path.open("a")
    sensor_process = subprocess.Popen(
        command,
        cwd=ROOT_DIR,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    sensor_run["anomaly"] = anomaly
    sensor_run["count"] = count
    sensor_run["interval"] = interval
    sensor_run["last_error"] = None

    time.sleep(0.4)
    exit_code = sensor_process.poll()
    if exit_code is not None:
        sensor_process = None
        log_tail = read_log_tail(sensor_log_path, limit=20)
        error_text = log_tail[-1] if log_tail else f"sensor process exited with code {exit_code}"
        sensor_run["last_error"] = error_text
        raise RuntimeError(error_text)


def stop_sensor_run():
    global sensor_process

    if sensor_process is None:
        return

    if sensor_process.poll() is None:
        sensor_process.send_signal(signal.SIGTERM)
    sensor_process = None


@app.route("/")
def index():
    return redirect(url_for("monitoring_page"))


@app.route("/monitoring")
def monitoring_page():
    return render_template("monitoring.html")


@app.route("/sensors")
def sensors_page():
    return render_template("sensors.html")


@app.route("/reports")
def reports_page():
    benchmark_rows = load_benchmark_rows()
    normalization_result = load_normalization_result()
    reference_summary = load_reference_summary()
    report_files = sorted(
        path.name for path in REPORTS_DIR.iterdir() if path.is_file()
    ) if REPORTS_DIR.exists() else []
    return render_template(
        "reports.html",
        benchmark_rows=benchmark_rows,
        normalization_result=normalization_result,
        reference_summary=reference_summary,
        report_files=report_files,
    )


@app.route("/api/monitoring")
def monitoring_api():
    telemetry, inference, alerts = load_latest_data()
    return jsonify(
        {
            "telemetry": telemetry,
            "prediction": inference,
            "alerts": alerts,
            "psi_line": latest_psi_line(),
            "inference_log_tail": read_log_tail(LOGS_DIR / "inference.log", limit=12),
        }
    )


@app.route("/api/sensors/status")
def sensor_status_api():
    status = current_sensor_status()
    status["log_tail"] = read_log_tail(active_sensor_log, limit=20)
    status["log_path"] = str(active_sensor_log)
    return jsonify(status)


@app.route("/api/sensors/run", methods=["POST"])
def sensor_run_api():
    try:
        payload = request.get_json(silent=True) or request.form
        anomaly = payload.get("anomaly", "none")
        count = int(payload.get("count", 60))
        interval = float(payload.get("interval", 0.2))
        start_sensor_run(anomaly, count, interval)
    except RuntimeError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 409
    except Exception as exc:
        sensor_run["last_error"] = str(exc)
        return jsonify({"ok": False, "error": str(exc)}), 500

    return jsonify({"ok": True, "status": current_sensor_status()})


@app.route("/api/sensors/stop", methods=["POST"])
def sensor_stop_api():
    stop_sensor_run()
    return jsonify({"ok": True, "status": current_sensor_status()})


@app.route("/reports/<path:filename>")
def report_asset(filename):
    return send_from_directory(REPORTS_DIR, filename)


def main():
    port = int(os.getenv("WEBAPP_PORT", "5001"))
    app.run(host="127.0.0.1", port=port, debug=False)


if __name__ == "__main__":
    main()
