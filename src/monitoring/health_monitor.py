import platform
import psutil
from datetime import datetime

class HealthMonitor:
    def get_health(self):
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "system": platform.system(),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage("/").percent,
            "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat(),
        }

if __name__ == "__main__":
    monitor = HealthMonitor()
    print(monitor.get_health())
