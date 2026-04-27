from typing import Any, TypeVar

from sqlalchemy.orm import Session

from app import models

ModelType = TypeVar("ModelType")


def get_by_id(db: Session, model: type[ModelType], item_id: int) -> ModelType | None:
    return db.query(model).filter(model.id == item_id).first()


def list_all(db: Session, model: type[ModelType]) -> list[ModelType]:
    return db.query(model).order_by(model.id.desc()).all()


def create_item(db: Session, model: type[ModelType], payload: dict[str, Any]) -> ModelType:
    item = model(**payload)
    db.add(item)
    db.commit()
    db.refresh(item)
    return item


def update_item(db: Session, item: Any, payload: dict[str, Any]) -> Any:
    for key, value in payload.items():
        setattr(item, key, value)
    db.commit()
    db.refresh(item)
    return item


def delete_item(db: Session, item: Any) -> None:
    db.delete(item)
    db.commit()


def upsert_probe_result(
    db: Session,
    probe_id: int,
    mean_titer: float,
    std_titer: float,
    n_measurements: int,
    p_value: float | None,
    method: str,
) -> models.ProbeResult:
    existing = (
        db.query(models.ProbeResult).filter(models.ProbeResult.probe_id == probe_id).first()
    )
    if existing:
        existing.mean_titer = mean_titer
        existing.std_titer = std_titer
        existing.n_measurements = n_measurements
        existing.p_value = p_value
        existing.method = method
        db.commit()
        db.refresh(existing)
        return existing

    created = models.ProbeResult(
        probe_id=probe_id,
        mean_titer=mean_titer,
        std_titer=std_titer,
        n_measurements=n_measurements,
        p_value=p_value,
        method=method,
    )
    db.add(created)
    db.commit()
    db.refresh(created)
    return created
