"""SQLite persistence, password handling, plans, and usage limits."""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, create_engine, select, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

BASE_DIR = Path(__file__).resolve().parent
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'incident_system.db'}")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

PLANS = {
    "Basic": {"monthly_scans": 5, "price": 0},
    "Pro": {"monthly_scans": 100, "price": 9},
    "Enterprise": {"monthly_scans": 10000, "price": 49},
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
    active: Mapped[bool] = mapped_column(default=True)
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
    summary: Mapped[str] = mapped_column(Text)
    snapshot_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    alerted: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    user: Mapped[User] = relationship(back_populates="incidents")


def init_db() -> None:
    Base.metadata.create_all(engine)


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
        user.subscription = Subscription(plan="Basic")
        db.add(user)
        db.commit()
        db.refresh(user)
        return user, None


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
            db.add(Subscription(user_id=user_id, plan=plan))
        else:
            subscription.plan, subscription.active = plan, True
        db.commit()


def get_usage(user_id: int) -> tuple[str, int, int]:
    now = datetime.now(timezone.utc)
    month_start = datetime(now.year, now.month, 1, tzinfo=timezone.utc).replace(tzinfo=None)
    with SessionLocal() as db:
        sub = db.scalar(select(Subscription).where(Subscription.user_id == user_id))
        plan = sub.plan if sub and sub.active else "Basic"
        used = db.scalar(select(func.count(IncidentLog.id)).where(
            IncidentLog.user_id == user_id, IncidentLog.created_at >= month_start
        )) or 0
    return plan, int(used), PLANS[plan]["monthly_scans"]


def save_incident(user_id: int, result: dict, source_name: str, snapshot_path: str | None, alerted: bool) -> IncidentLog:
    with SessionLocal() as db:
        record = IncidentLog(
            user_id=user_id, source_name=source_name[:255],
            threat_percent=result["threat_percent"], peak_confidence=result["peak_confidence"],
            incident_start=result["incident_start"], incident_end=result["incident_end"],
            summary=result["summary"], snapshot_path=snapshot_path, alerted=alerted,
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return record


def incident_history(user_id: int) -> list[IncidentLog]:
    with SessionLocal() as db:
        rows = list(db.scalars(select(IncidentLog).where(IncidentLog.user_id == user_id).order_by(IncidentLog.created_at.desc()).limit(100)))
        for row in rows:
            db.expunge(row)
        return rows
