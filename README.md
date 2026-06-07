# GreenScan: Green Waste Federated Learning Platform

GreenScan is a lightweight, eco-friendly waste intelligence platform. It features a modern web interface for real-time waste classification powered by a PyTorch-based model (MobileNetV3), alongside a Federated Learning (FL) pipeline powered by Flower (`flwr`). The system integrates carbon footprint and hardware utilization tracking on both centralized and federated workloads.

---

## 📁 Project Architecture & Components

The codebase consists of the following key files:

### 1. `green_tracker.py` (Environmental Metrics Tracker)
Encapsulates resource monitoring and carbon footprint tracing using `codecarbon` and `psutil`.
- Logs duration, CPU utilization percentage, RAM usage (in MB), and network bandwidth (in MB).
- Tracks power consumption (in kWh) and carbon emissions (in kg of CO₂) during centralized and federated training runs.

### 2. `waste_model.py` (Real Inference Engine)
Implements a Python classifier based on a pre-trained **MobileNetV3-Small** architecture.
- Maps ImageNet's 1000 categories to five primary waste classes: **Organic**, **Plastic**, **Paper**, **Mixed Waste**, and **Concealed-Polybag** using a keyword voting system.
- Evaluates the top-10 predictions from the model to calculate normalized category confidence scores.

### 3. `api_backend.py` (FastAPI Server)
Acts as the central orchestration server.
- Serves the frontend web interface (`index.html`).
- Implements the `/predict` POST endpoint for real-time waste image classification using `waste_model.py`.
- Provides `/train/centralized` to simulate training while logging environmental metrics via `green_tracker.py`.
- Provides CORS capabilities and mounts folders for static file serving.

### 4. `index.html` (Web Frontend)
A premium dark-themed single-page application built with modern vanilla CSS and JavaScript.
- Offers interactive drag-and-drop or click-to-browse image uploading.
- Connects to the FastAPI backend to display predicted waste categories, confidence meters, inference times, and category breakdown charts.

### 5. `fl_server.py` (Federated Learning Server)
Initializes and starts the Flower server.
- Employs a custom strategy `GreenFedAvg` (extending Flower's `FedAvg`) to aggregate client weights.
- Logs aggregated carbon emissions and network statistics on the server side at the end of each round.

### 6. `fl_client.py` (Federated Learning Client / Edge Node)
Represents a decentralized node participating in the federated training cluster.
- Connects to the Flower server.
- Prepares local PyTorch MobileNetV3 models.
- Tracks and reports client-side carbon emissions to the server after each training epoch using `green_tracker.py`.

---

## 🚀 Getting Started

### 1. Download Datasets
The datasets used in this project can be downloaded from the following sources:
- **Kaggle**: [Waste Classification Data](https://www.kaggle.com/datasets/techsash/waste-classification-data)
- **Google Drive**: [Project Dataset Folder](https://drive.google.com/drive/u/0/folders/10flDS8WzA21_WUy_COY8cEBFmKYLYeuq?brid=YWdncwF_imC9fl9b-HCTtWo8iUi8)

Download the datasets and extract them into the appropriate dataset directories in the project root.

### 2. Install Dependencies
Ensure you have Python 3.13 installed. Install the required libraries:
```bash
pip install fastapi uvicorn pydantic python-multipart flwr torch torchvision codecarbon psutil pillow
```

### 2. Run the Real-Time Classifier (Web Application)
To start the web server and use the user interface:
```bash
python api_backend.py
```
Open your browser and navigate to:
👉 **[http://localhost:8000](http://localhost:8000)**

### 3. Run the Federated Learning Cluster
To run a 3-round training cycle, start the server first, then spin up the client node:

#### Start the FL Server:
```bash
python fl_server.py
```

#### Start the FL Client:
```bash
python fl_client.py
```

---

## 🌿 Waste Categories & Actions
* **🌱 Organic**: Biodegradable food scraps, leaves, and garden waste. (Action: Composting)
* **♻️ Plastic**: Bottles and synthetic polymer materials. (Action: Verify plastic number and recycle)
* **📄 Paper**: Cellulose-based dry material. (Action: Clean paper recycling)
* **🗑️ Mixed Waste**: Unsorted mixed garbage. (Action: Sort before disposal)
* **🛍️ Concealed-Polybag**: Hidden soft plastic packaging/bags. (Action: Check local guidelines for thin-film collection)
