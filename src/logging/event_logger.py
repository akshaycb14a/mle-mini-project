from .logger import get_logger

logger = get_logger()

def log_telemetry(device_id, temperature, vibration):
    logger.info(
        "Telemetry | device=%s temp=%.2f vibration=%.2f",
        device_id,
        temperature,
        vibration,
    )

def log_inference(prediction, confidence):
    logger.info(
        "Inference | prediction=%s confidence=%.4f",
        prediction,
        confidence,
    )

def log_alert(severity, message):
    logger.warning(
        "Alert | severity=%s message=%s",
        severity,
        message,
    )

def log_error(error):
    logger.exception("Error | %s", error)
