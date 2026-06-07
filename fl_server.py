"""
fl_server.py — Federated Learning server with DB logging and round broadcasting.
"""

import json
import asyncio
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models
import flwr as fl
from green_tracker import GreenMetricsTracker
from database import init_db, create_training_run, log_fl_round, finish_training_run

DEVICE = torch.device("cpu")
NUM_ROUNDS = 3
_fl_run_id: int = None
_round_broadcast_queue = None  # Set externally by api_backend if WebSocket needed


def set_broadcast_queue(q):
    global _round_broadcast_queue
    _round_broadcast_queue = q


def _broadcast(data: dict):
    """Post round update to WebSocket queue if available."""
    if _round_broadcast_queue is not None:
        try:
            _round_broadcast_queue.put_nowait(data)
        except Exception:
            pass


class GreenFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, run_id: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tracker = GreenMetricsTracker()
        self.run_id = run_id
        self.round_history = []

    def aggregate_fit(self, server_round, results, failures):
        print(f"\n── FL Round {server_round} Aggregation ──")
        self.tracker.start_tracking()

        aggregated_weights, metrics = super().aggregate_fit(
            server_round, results, failures
        )

        green = self.tracker.stop_tracking()

        # Collect per-node metrics from client results
        for client_proxy, fit_res in results:
            m = fit_res.metrics or {}
            node_id = m.get("node_id", "unknown")
            log_fl_round(
                run_id=self.run_id,
                round_num=server_round,
                node_id=node_id,
                local_loss=float(m.get("loss", 0.0)),
                local_acc=float(m.get("accuracy", 0.0)),
                co2_kg=float(m.get("co2_kg", 0.0)),
                bandwidth_mb=float(m.get("bandwidth_mb", 0.0)),
                num_samples=fit_res.num_examples,
            )

        # Log server-side aggregation metrics
        log_fl_round(
            run_id=self.run_id,
            round_num=server_round,
            node_id="server",
            local_loss=0.0,
            local_acc=0.0,
            co2_kg=float(green.get("estimated_co2_kg", 0.0)),
            bandwidth_mb=float(green.get("bandwidth_used_mb", 0.0)),
            num_samples=0,
        )

        round_data = {
            "type": "fl_round",
            "round": server_round,
            "total_rounds": NUM_ROUNDS,
            "clients": len(results),
            "failures": len(failures),
            "server_co2_kg": round(float(green.get("estimated_co2_kg", 0.0)), 6),
        }
        self.round_history.append(round_data)
        _broadcast(round_data)
        print(f"[SERVER] Round {server_round} complete. CO2: {green.get('estimated_co2_kg', 0):.6f} kg")

        return aggregated_weights, metrics

    def aggregate_evaluate(self, server_round, results, failures):
        aggregated = super().aggregate_evaluate(server_round, results, failures)

        # Compute average accuracy across nodes
        if results:
            total_acc = sum(r.metrics.get("accuracy", 0.0) for _, r in results)
            avg_acc = total_acc / len(results)
        else:
            avg_acc = 0.0

        _broadcast({
            "type": "fl_eval",
            "round": server_round,
            "global_accuracy": round(avg_acc * 100, 2),
        })
        return aggregated


def start_fl_server(num_rounds: int = NUM_ROUNDS, min_clients: int = 1):
    global _fl_run_id
    init_db()
    _fl_run_id = create_training_run("federated", notes=f"rounds={num_rounds}")

    strategy = GreenFedAvg(
        run_id=_fl_run_id,
        fraction_fit=1.0,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
    )

    print(f"Starting Federated Learning Server (rounds={num_rounds}, min_clients={min_clients})...")
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )
    finish_training_run(_fl_run_id, best_val_acc=0.0,
                        total_epochs=num_rounds, status="completed")
    print("[FL SERVER] Training complete.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--min-clients", type=int, default=1)
    args = parser.parse_args()
    start_fl_server(num_rounds=args.rounds, min_clients=args.min_clients)
