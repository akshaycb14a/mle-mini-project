from .logger import get_logger

logger = get_logger()

def log_telemetry(device_id, temperature, humidity):
    logger.info(
        "Telemetry | device=%s temp=%.2f humidity=%.2f",
        device_id,
        temperature,
        humidity,
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
