import psutil
import time
from codecarbon import OfflineEmissionsTracker


class GreenMetricsTracker:
    def __init__(self, country_iso_code="BGD"):  # Default: Bangladesh
        self.tracker = OfflineEmissionsTracker(
            country_iso_code=country_iso_code, log_level="error"
        )
        self.start_time = None
        self.start_net = None

    def start_tracking(self):
        self.start_time = time.time()
        self.start_net = psutil.net_io_counters()
        self.tracker.start()
        print("[GREEN METRICS] Tracking started...")

    def stop_tracking(self):
        emissions = self.tracker.stop()
        end_time = time.time()
        end_net = psutil.net_io_counters()

        duration = end_time - self.start_time
        bytes_sent = end_net.bytes_sent - self.start_net.bytes_sent
        bytes_recv = end_net.bytes_recv - self.start_net.bytes_recv
        total_bandwidth_mb = (bytes_sent + bytes_recv) / (1024 * 1024)

        metrics = {
            "duration_seconds": round(duration, 2),
            "cpu_usage_percent": psutil.cpu_percent(),
            "ram_usage_mb": round(
                psutil.virtual_memory().used / (1024 * 1024), 2
            ),
            "bandwidth_used_mb": round(total_bandwidth_mb, 4),
            "estimated_co2_kg": emissions,
            "power_kwh": self.tracker._total_energy.kWh,
        }

        print(
            "[GREEN METRICS] Tracking stopped. "
            f"CO2 emitted: {emissions:.6f} kg"
        )
        return metrics


if __name__ == "__main__":
    tracker = GreenMetricsTracker()
    tracker.start_tracking()
    time.sleep(2)  # Simulate workload
    metrics = tracker.stop_tracking()
    print(metrics)
