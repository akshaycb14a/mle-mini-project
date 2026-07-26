from dataclasses import dataclass

@dataclass
class Telemetry:
    device_id: str
    temperature: float
    humidity: float
    anomaly_score: float

@dataclass
class Prediction:
    label: str
    confidence: float

@dataclass
class Alert:
    alert_type: str
    message: str
    timestamp: str
