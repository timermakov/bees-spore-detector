from sqlalchemy.orm import Session

from app import models
from app.db import Base, SessionLocal, engine


def seed(db: Session) -> None:
    # Safety net for fresh environments where migrations were skipped.
    Base.metadata.create_all(bind=engine)

    if not db.query(models.User).first():
        user = models.User(email="demo@bees.local", full_name="Demo Researcher")
        db.add(user)
        db.flush()
        project = models.Project(
            user_id=user.id,
            name="RNAi Nosema Pilot",
            description="Demo project for biological screening",
        )
        db.add(project)

    if not db.query(models.Species).first():
        species_records = [
            models.Species(name="Nosema bombycis", latin_name="Nosema bombycis"),
            models.Species(name="Nosema ceranae", latin_name="Nosema ceranae"),
        ]
        db.add_all(species_records)
    db.commit()


if __name__ == "__main__":
    session = SessionLocal()
    try:
        seed(session)
        print("Seed data inserted.")
    finally:
        session.close()
