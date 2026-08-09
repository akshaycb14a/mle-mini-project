SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS telemetry(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    device_id TEXT NOT NULL,
    temperature REAL,
    vibration REAL,
    anomaly_score REAL
);

CREATE TABLE IF NOT EXISTS inference_results(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    prediction TEXT,
    confidence REAL
);

CREATE TABLE IF NOT EXISTS alerts(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    severity TEXT,
    message TEXT
);

CREATE INDEX IF NOT EXISTS idx_telemetry_time ON telemetry(timestamp);
"""

def create_schema(conn):
    conn.executescript(SCHEMA_SQL)
    conn.commit()
