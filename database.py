"""
database.py — SQLite database layer via SQLAlchemy ORM
All training runs, epoch metrics, FL rounds, predictions, model versions,
and green metrics are stored here for reporting and comparison.
"""

from datetime import datetime
from sqlalchemy import (
    create_engine, Column, Integer, Float, String,
    DateTime, ForeignKey, Text
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

Base = declarative_base()
ENGINE = create_engine(
    "sqlite:///greenscan.db",
    echo=False,
    connect_args={"check_same_thread": False, "timeout": 30}
)
SessionLocal = sessionmaker(bind=ENGINE)


# ── ORM Models ────────────────────────────────────────────────────────────────

class TrainingRun(Base):
    __tablename__ = "training_runs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    mode = Column(String, nullable=False)          # "centralized" | "federated"
    dataset_version = Column(String, default="v1")
    status = Column(String, default="running")     # running | completed | stopped
    started_at = Column(DateTime, default=datetime.utcnow)
    finished_at = Column(DateTime, nullable=True)
    best_val_acc = Column(Float, default=0.0)
    total_epochs = Column(Integer, default=0)
    notes = Column(Text, nullable=True)

    epoch_metrics = relationship("EpochMetric", back_populates="run", cascade="all, delete-orphan")
    fl_rounds = relationship("FLRound", back_populates="run", cascade="all, delete-orphan")
    green_metrics = relationship("GreenMetric", back_populates="run", cascade="all, delete-orphan")


class EpochMetric(Base):
    __tablename__ = "epoch_metrics"
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("training_runs.id"), nullable=False)
    epoch = Column(Integer, nullable=False)
    train_loss = Column(Float, default=0.0)
    val_loss = Column(Float, default=0.0)
    train_acc = Column(Float, default=0.0)
    val_acc = Column(Float, default=0.0)
    learning_rate = Column(Float, default=0.0)
    duration_seconds = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)

    run = relationship("TrainingRun", back_populates="epoch_metrics")


class FLRound(Base):
    __tablename__ = "fl_rounds"
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("training_runs.id"), nullable=False)
    round_num = Column(Integer, nullable=False)
    node_id = Column(String, nullable=False)       # "node_a" | "node_b" | "node_c" | "global"
    local_loss = Column(Float, default=0.0)
    local_acc = Column(Float, default=0.0)
    co2_kg = Column(Float, default=0.0)
    bandwidth_mb = Column(Float, default=0.0)
    num_samples = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

    run = relationship("TrainingRun", back_populates="fl_rounds")


class PredictionHistory(Base):
    __tablename__ = "prediction_history"
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    filename = Column(String, nullable=True)
    predicted_class = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    inference_ms = Column(Integer, default=0)
    model_version = Column(String, nullable=True)
    top5_json = Column(Text, nullable=True)


class ModelVersion(Base):
    __tablename__ = "model_versions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    version = Column(String, unique=True, nullable=False)   # "v1", "v2", …
    path = Column(String, nullable=False)
    train_acc = Column(Float, default=0.0)
    val_acc = Column(Float, default=0.0)
    test_acc = Column(Float, nullable=True)
    co2_kg = Column(Float, default=0.0)
    is_active = Column(Integer, default=0)                  # 1 = active
    created_at = Column(DateTime, default=datetime.utcnow)
    notes = Column(Text, nullable=True)


class GreenMetric(Base):
    __tablename__ = "green_metrics"
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("training_runs.id"), nullable=False)
    phase = Column(String, nullable=False)          # "epoch_N" | "fl_round_N" | "inference"
    duration_s = Column(Float, default=0.0)
    cpu_pct = Column(Float, default=0.0)
    ram_mb = Column(Float, default=0.0)
    bandwidth_mb = Column(Float, default=0.0)
    co2_kg = Column(Float, default=0.0)
    kwh = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)

    run = relationship("TrainingRun", back_populates="green_metrics")


# ── Helpers ───────────────────────────────────────────────────────────────────

def init_db():
    Base.metadata.create_all(ENGINE)
    print("[DB] Database initialized — greenscan.db")


def get_session():
    return SessionLocal()


def create_training_run(mode: str, notes: str = "") -> int:
    session = get_session()
    try:
        run = TrainingRun(mode=mode, notes=notes)
        session.add(run)
        session.commit()
        return run.id
    finally:
        session.close()


def log_epoch(run_id: int, epoch: int, train_loss: float, val_loss: float,
              train_acc: float, val_acc: float, lr: float, duration: float):
    session = get_session()
    try:
        session.add(EpochMetric(
            run_id=run_id, epoch=epoch,
            train_loss=train_loss, val_loss=val_loss,
            train_acc=train_acc, val_acc=val_acc,
            learning_rate=lr, duration_seconds=duration
        ))
        session.commit()
    finally:
        session.close()


def finish_training_run(run_id: int, best_val_acc: float, total_epochs: int,
                        status: str = "completed"):
    session = get_session()
    try:
        run = session.get(TrainingRun, run_id)
        if run:
            run.status = status
            run.best_val_acc = best_val_acc
            run.total_epochs = total_epochs
            run.finished_at = datetime.utcnow()
            session.commit()
    finally:
        session.close()


def log_fl_round(run_id: int, round_num: int, node_id: str,
                 local_loss: float, local_acc: float,
                 co2_kg: float, bandwidth_mb: float, num_samples: int):
    session = get_session()
    try:
        session.add(FLRound(
            run_id=run_id, round_num=round_num, node_id=node_id,
            local_loss=local_loss, local_acc=local_acc,
            co2_kg=co2_kg, bandwidth_mb=bandwidth_mb, num_samples=num_samples
        ))
        session.commit()
    finally:
        session.close()


def save_model_version(version: str, path: str, train_acc: float,
                       val_acc: float, co2_kg: float, notes: str = "") -> int:
    session = get_session()
    try:
        mv = ModelVersion(
            version=version, path=path,
            train_acc=train_acc, val_acc=val_acc,
            co2_kg=co2_kg, notes=notes
        )
        session.add(mv)
        session.commit()
        return mv.id
    finally:
        session.close()


def set_active_model(version: str):
    session = get_session()
    try:
        session.query(ModelVersion).update({"is_active": 0})
        mv = session.query(ModelVersion).filter_by(version=version).first()
        if mv:
            mv.is_active = 1
        session.commit()
    finally:
        session.close()


def get_all_model_versions() -> list:
    session = get_session()
    try:
        rows = session.query(ModelVersion).order_by(ModelVersion.created_at.desc()).all()
        return [
            {
                "version": r.version, "path": r.path,
                "train_acc": r.train_acc, "val_acc": r.val_acc,
                "test_acc": r.test_acc, "co2_kg": r.co2_kg,
                "is_active": bool(r.is_active),
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "notes": r.notes,
            }
            for r in rows
        ]
    finally:
        session.close()


def get_active_model_path() -> str | None:
    session = get_session()
    try:
        mv = session.query(ModelVersion).filter_by(is_active=1).first()
        return mv.path if mv else None
    finally:
        session.close()


def log_prediction(filename: str, predicted_class: str, confidence: float,
                   inference_ms: int, model_version: str = "imagenet",
                   top5_json: str = ""):
    session = get_session()
    try:
        session.add(PredictionHistory(
            filename=filename, predicted_class=predicted_class,
            confidence=confidence, inference_ms=inference_ms,
            model_version=model_version, top5_json=top5_json
        ))
        session.commit()
    finally:
        session.close()


def get_epoch_history(run_id: int) -> list:
    session = get_session()
    try:
        rows = session.query(EpochMetric).filter_by(run_id=run_id).order_by(
            EpochMetric.epoch).all()
        return [
            {
                "epoch": r.epoch, "train_loss": r.train_loss,
                "val_loss": r.val_loss, "train_acc": r.train_acc,
                "val_acc": r.val_acc, "learning_rate": r.learning_rate,
                "duration_seconds": r.duration_seconds,
            }
            for r in rows
        ]
    finally:
        session.close()


def get_all_runs() -> list:
    session = get_session()
    try:
        rows = session.query(TrainingRun).order_by(
            TrainingRun.started_at.desc()).all()
        return [
            {
                "id": r.id, "mode": r.mode, "status": r.status,
                "best_val_acc": r.best_val_acc, "total_epochs": r.total_epochs,
                "started_at": r.started_at.isoformat() if r.started_at else None,
                "finished_at": r.finished_at.isoformat() if r.finished_at else None,
            }
            for r in rows
        ]
    finally:
        session.close()


def get_fl_rounds_for_run(run_id: int) -> list:
    session = get_session()
    try:
        rows = session.query(FLRound).filter_by(run_id=run_id).order_by(
            FLRound.round_num, FLRound.node_id).all()
        return [
            {
                "round_num": r.round_num, "node_id": r.node_id,
                "local_loss": r.local_loss, "local_acc": r.local_acc,
                "co2_kg": r.co2_kg, "bandwidth_mb": r.bandwidth_mb,
                "num_samples": r.num_samples,
            }
            for r in rows
        ]
    finally:
        session.close()


if __name__ == "__main__":
    init_db()
