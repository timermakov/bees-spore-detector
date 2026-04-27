from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    projects: Mapped[list["Project"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
    )


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    user: Mapped["User"] = relationship(back_populates="projects")
    probes: Mapped[list["Probe"]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan",
    )
    model_weights: Mapped[list["ModelWeight"]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan",
    )


class Species(Base):
    __tablename__ = "species"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    latin_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    probes: Mapped[list["Probe"]] = relationship(back_populates="species")


class Probe(Base):
    __tablename__ = "probes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    species_id: Mapped[int] = mapped_column(ForeignKey("species.id", ondelete="RESTRICT"), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    treatment_type: Mapped[str] = mapped_column(String(100), nullable=False)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    project: Mapped["Project"] = relationship(back_populates="probes")
    species: Mapped["Species"] = relationship(back_populates="probes")
    samples: Mapped[list["Sample"]] = relationship(
        back_populates="probe",
        cascade="all, delete-orphan",
    )
    result: Mapped["ProbeResult | None"] = relationship(
        back_populates="probe",
        cascade="all, delete-orphan",
        uselist=False,
    )

    __table_args__ = (UniqueConstraint("project_id", "name", name="uq_probe_project_name"),)


class Sample(Base):
    __tablename__ = "samples"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    probe_id: Mapped[int] = mapped_column(ForeignKey("probes.id", ondelete="CASCADE"), nullable=False)
    replicate_label: Mapped[str] = mapped_column(String(100), nullable=False)
    collected_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    probe: Mapped["Probe"] = relationship(back_populates="samples")
    images: Mapped[list["MicroImage"]] = relationship(
        back_populates="sample",
        cascade="all, delete-orphan",
    )

    __table_args__ = (UniqueConstraint("probe_id", "replicate_label", name="uq_sample_probe_label"),)


class MicroImage(Base):
    __tablename__ = "micro_images"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    sample_id: Mapped[int] = mapped_column(ForeignKey("samples.id", ondelete="CASCADE"), nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    sample: Mapped["Sample"] = relationship(back_populates="images")


class ProbeResult(Base):
    __tablename__ = "probe_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    probe_id: Mapped[int] = mapped_column(ForeignKey("probes.id", ondelete="CASCADE"), unique=True, nullable=False)
    mean_titer: Mapped[float] = mapped_column(Float, nullable=False)
    std_titer: Mapped[float] = mapped_column(Float, nullable=False)
    n_measurements: Mapped[int] = mapped_column(Integer, nullable=False)
    p_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    method: Mapped[str] = mapped_column(String(100), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    probe: Mapped["Probe"] = relationship(back_populates="result")


class ModelWeight(Base):
    __tablename__ = "model_weights"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    project_id: Mapped[int | None] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    path: Mapped[str] = mapped_column(Text, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    project: Mapped["Project | None"] = relationship(back_populates="model_weights")
