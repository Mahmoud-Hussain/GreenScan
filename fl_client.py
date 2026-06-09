"""
fl_client.py — Federated Learning client node with real dataset training.

Usage:
  python fl_client.py --node-id 0  (partition index 0, 1, or 2)
"""

import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import flwr as fl
from green_tracker import GreenMetricsTracker

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["clothing", "ewaste", "glass", "metal", "organic", "paper", "plastic", "recyclable"]
NUM_CLASSES = len(CLASSES)
NODE_NAMES = ["node_a", "node_b", "node_c"]


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model():
    model = models.mobilenet_v3_small(
        weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    )
    for param in model.features.parameters():
        param.requires_grad = False
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, NUM_CLASSES)
    return model.to(DEVICE)


# ── Flower Client ─────────────────────────────────────────────────────────────

class WasteClient(fl.client.NumPyClient):
    def __init__(self, node_id: int, data_loader, name: str = None):
        self.node_name = name if name else (NODE_NAMES[node_id] if node_id < len(NODE_NAMES) else f"node_{node_id}")
        self.model = build_model()
        self.loader = data_loader
        self.tracker = GreenMetricsTracker()
        self.criterion = nn.CrossEntropyLoss()
        print(f"[{self.node_name.upper()}] Initialized. Samples: {len(data_loader.dataset) if data_loader else 0}")

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.tracker.start_tracking()

        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=1e-3
        )
        self.model.train()

        total_loss, correct, total = 0.0, 0, 0

        if self.loader:
            for inputs, labels in self.loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += inputs.size(0)
        else:
            # No data — simulate with small delay
            time.sleep(2)
            total = 1

        avg_loss = total_loss / total if total > 0 else 0.5
        accuracy = correct / total if total > 0 else 0.0

        green = self.tracker.stop_tracking()
        print(f"[{self.node_name.upper()}] fit done — loss={avg_loss:.4f} acc={accuracy:.4f}")

        metrics = {
            "loss": float(avg_loss),
            "accuracy": float(accuracy),
            "co2_kg": float(green.get("estimated_co2_kg", 0.0)),
            "bandwidth_mb": float(green.get("bandwidth_used_mb", 0.0)),
            "node_id": self.node_name,
        }
        return self.get_parameters(config={}), total or 1, metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        if self.loader:
            with torch.no_grad():
                for inputs, labels in self.loader:
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    total_loss += loss.item() * inputs.size(0)
                    preds = outputs.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += inputs.size(0)
        else:
            total = 1
            total_loss = 0.5

        avg_loss = total_loss / total if total > 0 else 0.5
        accuracy = correct / total if total > 0 else 0.0
        return float(avg_loss), total, {"accuracy": float(accuracy)}


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GreenScan FL Client Node")
    parser.add_argument("--node-id", type=int, default=0,
                        help="Node partition index (0, 1, or 2)")
    parser.add_argument("--server", type=str, default="127.0.0.1:8080",
                        help="FL server address")
    parser.add_argument("--nodes", type=int, default=3,
                        help="Total number of FL nodes (for partitioning)")
    parser.add_argument("--name", type=str, default=None,
                        help="Custom display name for this client node")
    args = parser.parse_args()

    node_name = args.name if args.name else (NODE_NAMES[args.node_id] if args.node_id < len(NODE_NAMES) else f"node_{args.node_id}")
    print(f"Starting FL Edge Node [{node_name.upper()}]...")

    from dataset_manager import partition_for_fl
    partitions = partition_for_fl(num_nodes=args.nodes, batch_size=16)
    loader = partitions[args.node_id] if args.node_id < len(partitions) else None

    client = WasteClient(node_id=args.node_id, data_loader=loader, name=node_name)
    fl.client.start_client(
        server_address=args.server,
        client=client.to_client()
    )
