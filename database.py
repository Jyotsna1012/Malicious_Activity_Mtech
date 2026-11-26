"""SQLite persistence, salted password hashes, subscriptions, and quotas."""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, create_engine, func, select, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

BASE_DIR = Path(__file__).resolve().parent
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'incident_system.db'}")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
DEFAULT_ALERT_THRESHOLD = float(os.getenv("ALERT_THRESHOLD_PERCENT", "20"))

PLANS = {
    "Basic": {"monthly_scans": 5, "price": 0, "live_webcam": False, "custom_threshold": False},
    "Pro": {"monthly_scans": 50, "price": 9, "live_webcam": True, "custom_threshold": False},
    "Enterprise": {"monthly_scans": None, "price": 49, "live_webcam": True, "custom_threshold": True},
}


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(256))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    subscription: Mapped["Subscription"] = relationship(back_populates="user", uselist=False, cascade="all, delete-orphan")
    incidents: Mapped[list["IncidentLog"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class Subscription(Base):
    __tablename__ = "subscriptions"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), unique=True)
    plan: Mapped[str] = mapped_column(String(32), default="Basic")
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    alert_threshold: Mapped[float] = mapped_column(Float, default=DEFAULT_ALERT_THRESHOLD)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    user: Mapped[User] = relationship(back_populates="subscription")


class IncidentLog(Base):
    __tablename__ = "incident_logs"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    source_name: Mapped[str] = mapped_column(String(255))
    threat_percent: Mapped[float] = mapped_column(default=0.0)
    peak_confidence: Mapped[float] = mapped_column(default=0.0)
    incident_start: Mapped[str | None] = mapped_column(String(16), nullable=True)
    incident_end: Mapped[str | None] = mapped_column(String(16), nullable=True)
    incident_runs_json: Mapped[str] = mapped_column(Text, default="[]")
    action_labels_json: Mapped[str] = mapped_column(Text, default="{}")
    top_action_labels_json: Mapped[str] = mapped_column(Text, default="{}")
    summary: Mapped[str] = mapped_column(Text)
    snapshot_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    storyboard_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    alerted: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    user: Mapped[User] = relationship(back_populates="incidents")

    @property
    def incident_runs(self) -> list[dict]:
        return json.loads(self.incident_runs_json or "[]")

    @property
    def action_labels(self) -> dict:
        return json.loads(self.action_labels_json or "{}")

    @property
    def top_action_labels(self) -> dict:
        return json.loads(self.top_action_labels_json or "{}")


def init_db() -> None:
    """Create tables and add columns to installations created by older app versions."""
    Base.metadata.create_all(engine)
    if not DATABASE_URL.startswith("sqlite"):
        return
    migrations = {
        "subscriptions": {"alert_threshold": "FLOAT NOT NULL DEFAULT 20"},
        "incident_logs": {
            "incident_runs_json": "TEXT NOT NULL DEFAULT '[]'",
            "action_labels_json": "TEXT NOT NULL DEFAULT '{}'",
            "top_action_labels_json": "TEXT NOT NULL DEFAULT '{}'",
            "storyboard_path": "TEXT",
        },
    }
    with engine.begin() as connection:
        for table, columns in migrations.items():
            existing = {row[1] for row in connection.execute(text(f"PRAGMA table_info({table})"))}
            for column, definition in columns.items():
                if column not in existing:
                    connection.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {definition}"))


def _hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1)
    return f"{salt.hex()}:{digest.hex()}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        salt_hex, _ = encoded.split(":", 1)
        return hmac.compare_digest(_hash_password(password, bytes.fromhex(salt_hex)), encoded)
    except (ValueError, TypeError):
        return False


def create_user(email: str, password: str) -> tuple[User | None, str | None]:
    email = email.strip().lower()
    if "@" not in email or len(email) > 320:
        return None, "Enter a valid email address."
    if len(password) < 8:
        return None, "Use a password with at least 8 characters."
    with SessionLocal() as db:
        if db.scalar(select(User).where(User.email == email)):
            return None, "An account with that email already exists."
        user = User(email=email, password_hash=_hash_password(password))
        user.subscription = Subscription(plan="Basic", alert_threshold=DEFAULT_ALERT_THRESHOLD)
        db.add(user)
        db.commit()
        db.refresh(user)
        return user, None


def get_workspace_user() -> User:
    """Return a shared dashboard identity, creating one for a fresh install."""
    with SessionLocal() as db:
        user = db.scalar(select(User).order_by(User.id).limit(1))
        if user is not None:
            db.expunge(user)
            return user

        user = User(
            email="workspace@local.invalid",
            password_hash=_hash_password(secrets.token_urlsafe(32)),
        )
        user.subscription = Subscription(plan="Basic", alert_threshold=DEFAULT_ALERT_THRESHOLD)
        db.add(user)
        db.commit()
        db.refresh(user)
        db.expunge(user)
        return user


def authenticate(email: str, password: str) -> User | None:
    with SessionLocal() as db:
        user = db.scalar(select(User).where(User.email == email.strip().lower()))
        if user and verify_password(password, user.password_hash):
            db.expunge(user)
            return user
    return None


def set_plan(user_id: int, plan: str) -> None:
    if plan not in PLANS:
        raise ValueError("Unknown subscription plan.")
    with SessionLocal() as db:
        subscription = db.scalar(select(Subscription).where(Subscription.user_id == user_id))
        if subscription is None:
            db.add(Subscription(user_id=user_id, plan=plan, alert_threshold=DEFAULT_ALERT_THRESHOLD))
        else:
            subscription.plan, subscription.active = plan, True
        db.commit()


def get_usage(user_id: int) -> tuple[str, int, int | None]:
    now = datetime.now(timezone.utc)
    month_start = datetime(now.year, now.month, 1)
    with SessionLocal() as db:
        sub = db.scalar(select(Subscription).where(Subscription.user_id == user_id))
        plan = sub.plan if sub and sub.active else "Basic"
        used = db.scalar(select(func.count(IncidentLog.id)).where(
            IncidentLog.user_id == user_id, IncidentLog.created_at >= month_start
        )) or 0
    return plan, int(used), PLANS[plan]["monthly_scans"]


def assert_quota_available(user_id: int) -> None:
    plan, used, allowance = get_usage(user_id)
    if allowance is not None and used >= allowance:
        raise ValueError(f"Monthly scan quota reached for the {plan} plan ({allowance} scans).")


def get_alert_threshold(user_id: int) -> float:
    with SessionLocal() as db:
        sub = db.scalar(select(Subscription).where(Subscription.user_id == user_id))
        return float(sub.alert_threshold if sub else DEFAULT_ALERT_THRESHOLD)


def set_alert_threshold(user_id: int, threshold: float) -> None:
    if not 0 <= float(threshold) <= 100:
        raise ValueError("Alert threshold must be between 0 and 100 percent.")
    with SessionLocal() as db:
        sub = db.scalar(select(Subscription).where(Subscription.user_id == user_id))
        if sub is None:
            raise ValueError("No subscription was found for this account.")
        if not PLANS[sub.plan]["custom_threshold"]:
            raise ValueError("A custom alert threshold requires the Enterprise plan.")
        sub.alert_threshold = float(threshold)
        db.commit()


def save_incident(user_id: int, result: dict, source_name: str,
                  snapshot_path: str | None, alerted: bool) -> IncidentLog:
    with SessionLocal() as db:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        month_start = datetime(now.year, now.month, 1)
        subscription = db.scalar(select(Subscription).where(Subscription.user_id == user_id))
        plan = subscription.plan if subscription and subscription.active else "Basic"
        allowance = PLANS[plan]["monthly_scans"]
        used = db.scalar(select(func.count(IncidentLog.id)).where(
            IncidentLog.user_id == user_id, IncidentLog.created_at >= month_start
        )) or 0
        if allowance is not None and used >= allowance:
            raise ValueError(f"Monthly scan quota reached for the {plan} plan ({allowance} scans).")
        record = IncidentLog(
            user_id=user_id, source_name=source_name[:255],
            threat_percent=result["threat_percent"], peak_confidence=result["peak_confidence"],
            incident_start=result["incident_start"], incident_end=result["incident_end"],
            incident_runs_json=json.dumps(result.get("incident_runs", [])),
            action_labels_json=json.dumps(result.get("action_labels", {})),
            top_action_labels_json=json.dumps(result.get("top_action_labels", {})),
            summary=result["summary"], snapshot_path=snapshot_path,
            storyboard_path=result.get("storyboard_path") or snapshot_path, alerted=alerted,
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return record


def incident_history(user_id: int) -> list[IncidentLog]:
    with SessionLocal() as db:
        rows = list(db.scalars(select(IncidentLog).where(IncidentLog.user_id == user_id)
                               .order_by(IncidentLog.created_at.desc()).limit(100)))
        for row in rows:
            db.expunge(row)
        return rows
