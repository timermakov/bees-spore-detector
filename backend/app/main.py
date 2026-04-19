from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from app import analytics_service, crud, models, schemas
from app.config import get_settings
from app.db import Base, engine, get_db

settings = get_settings()
app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _not_found(name: str) -> HTTPException:
    return HTTPException(status_code=404, detail=f"{name} not found")


def _get_or_404(db: Session, model: type[Any], item_id: int, name: str) -> Any:
    item = crud.get_by_id(db, model, item_id)
    if not item:
        raise _not_found(name)
    return item


upload_dir = Path(settings.upload_dir)
upload_dir.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=upload_dir), name="uploads")


@app.on_event("startup")
def ensure_tables() -> None:
    # Keeps API usable in local/dev sessions even before manual migration.
    Base.metadata.create_all(bind=engine)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/users", response_model=list[schemas.UserRead])
def list_users(db: Session = Depends(get_db)):
    return crud.list_all(db, models.User)


@app.post("/users", response_model=schemas.UserRead)
def create_user(payload: schemas.UserCreate, db: Session = Depends(get_db)):
    return crud.create_item(db, models.User, payload.model_dump())


@app.get("/species", response_model=list[schemas.SpeciesRead])
def list_species(db: Session = Depends(get_db)):
    return crud.list_all(db, models.Species)


@app.post("/species", response_model=schemas.SpeciesRead)
def create_species(payload: schemas.SpeciesCreate, db: Session = Depends(get_db)):
    return crud.create_item(db, models.Species, payload.model_dump())


@app.put("/species/{species_id}", response_model=schemas.SpeciesRead)
def update_species(species_id: int, payload: schemas.SpeciesCreate, db: Session = Depends(get_db)):
    item = _get_or_404(db, models.Species, species_id, "Species")
    return crud.update_item(db, item, payload.model_dump())


@app.delete("/species/{species_id}")
def delete_species(species_id: int, db: Session = Depends(get_db)):
    item = _get_or_404(db, models.Species, species_id, "Species")
    crud.delete_item(db, item)
    return {"ok": True}


@app.get("/projects", response_model=list[schemas.ProjectRead])
def list_projects(db: Session = Depends(get_db)):
    return crud.list_all(db, models.Project)


@app.post("/projects", response_model=schemas.ProjectRead)
def create_project(payload: schemas.ProjectCreate, db: Session = Depends(get_db)):
    _get_or_404(db, models.User, payload.user_id, "User")
    return crud.create_item(db, models.Project, payload.model_dump())


@app.put("/projects/{project_id}", response_model=schemas.ProjectRead)
def update_project(project_id: int, payload: schemas.ProjectUpdate, db: Session = Depends(get_db)):
    item = _get_or_404(db, models.Project, project_id, "Project")
    data = payload.model_dump(exclude_none=True)
    if "user_id" in data:
        _get_or_404(db, models.User, data["user_id"], "User")
    return crud.update_item(db, item, data)


@app.delete("/projects/{project_id}")
def delete_project(project_id: int, db: Session = Depends(get_db)):
    item = _get_or_404(db, models.Project, project_id, "Project")
    crud.delete_item(db, item)
    return {"ok": True}


@app.get("/probes", response_model=list[schemas.ProbeRead])
def list_probes(db: Session = Depends(get_db)):
    return crud.list_all(db, models.Probe)


@app.post("/probes", response_model=schemas.ProbeRead)
def create_probe(payload: schemas.ProbeCreate, db: Session = Depends(get_db)):
    _get_or_404(db, models.Project, payload.project_id, "Project")
    _get_or_404(db, models.Species, payload.species_id, "Species")
    return crud.create_item(db, models.Probe, payload.model_dump())


@app.put("/probes/{probe_id}", response_model=schemas.ProbeRead)
def update_probe(probe_id: int, payload: schemas.ProbeUpdate, db: Session = Depends(get_db)):
    item = _get_or_404(db, models.Probe, probe_id, "Probe")
    data = payload.model_dump(exclude_none=True)
    if "project_id" in data:
        _get_or_404(db, models.Project, data["project_id"], "Project")
    if "species_id" in data:
        _get_or_404(db, models.Species, data["species_id"], "Species")
    return crud.update_item(db, item, data)


@app.delete("/probes/{probe_id}")
def delete_probe(probe_id: int, db: Session = Depends(get_db)):
    item = _get_or_404(db, models.Probe, probe_id, "Probe")
    crud.delete_item(db, item)
    return {"ok": True}


@app.get("/samples", response_model=list[schemas.SampleRead])
def list_samples(db: Session = Depends(get_db)):
    return crud.list_all(db, models.Sample)


@app.post("/samples", response_model=schemas.SampleRead)
def create_sample(payload: schemas.SampleCreate, db: Session = Depends(get_db)):
    _get_or_404(db, models.Probe, payload.probe_id, "Probe")
    return crud.create_item(db, models.Sample, payload.model_dump())


@app.put("/samples/{sample_id}", response_model=schemas.SampleRead)
def update_sample(sample_id: int, payload: schemas.SampleUpdate, db: Session = Depends(get_db)):
    item = _get_or_404(db, models.Sample, sample_id, "Sample")
    data = payload.model_dump(exclude_none=True)
    if "probe_id" in data:
        _get_or_404(db, models.Probe, data["probe_id"], "Probe")
    return crud.update_item(db, item, data)


@app.delete("/samples/{sample_id}")
def delete_sample(sample_id: int, db: Session = Depends(get_db)):
    item = _get_or_404(db, models.Sample, sample_id, "Sample")
    for image in item.images:
        try:
            Path(image.file_path).unlink(missing_ok=True)
        except OSError:
            pass
    crud.delete_item(db, item)
    return {"ok": True}


@app.get("/images", response_model=list[schemas.MicroImageRead])
def list_images(db: Session = Depends(get_db)):
    return crud.list_all(db, models.MicroImage)


@app.post("/samples/{sample_id}/images", response_model=list[schemas.MicroImageRead])
async def upload_images(
    sample_id: int,
    files: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
):
    _get_or_404(db, models.Sample, sample_id, "Sample")
    created: list[models.MicroImage] = []
    target_dir = upload_dir / str(sample_id)
    target_dir.mkdir(parents=True, exist_ok=True)

    for file in files:
        suffix = Path(file.filename).suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png"}:
            raise HTTPException(status_code=400, detail="Only JPG/PNG files are supported")
        generated_name = f"{uuid4().hex}_{Path(file.filename).name}"
        target_path = target_dir / generated_name
        with target_path.open("wb") as output:
            output.write(await file.read())
        image = crud.create_item(
            db,
            models.MicroImage,
            {
                "sample_id": sample_id,
                "filename": Path(file.filename).name,
                "file_path": str(target_path.resolve()),
            },
        )
        created.append(image)
    return created


@app.delete("/images/{image_id}")
def delete_image(image_id: int, db: Session = Depends(get_db)):
    image = _get_or_404(db, models.MicroImage, image_id, "Image")
    try:
        Path(image.file_path).unlink(missing_ok=True)
    except OSError:
        pass
    crud.delete_item(db, image)
    return {"ok": True}


@app.get("/probe-results", response_model=list[schemas.ProbeResultRead])
def list_probe_results(db: Session = Depends(get_db)):
    return crud.list_all(db, models.ProbeResult)


@app.post("/probe-results", response_model=schemas.ProbeResultRead)
def upsert_probe_result(payload: schemas.ProbeResultUpsert, db: Session = Depends(get_db)):
    _get_or_404(db, models.Probe, payload.probe_id, "Probe")
    return crud.upsert_probe_result(db=db, **payload.model_dump())


@app.post("/probes/{probe_id}/analyze", response_model=schemas.AnalyzeResponse)
def analyze_probe(probe_id: int, mode: str | None = None, db: Session = Depends(get_db)):
    try:
        result = analytics_service.analyze_probe(db, probe_id=probe_id, mode=mode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return schemas.AnalyzeResponse(
        probe_id=result.probe_id,
        mean_titer=result.mean_titer,
        std_titer=result.std_titer,
        n_measurements=result.n_measurements,
        p_value=result.p_value,
        method=result.method,
    )


@app.get("/model-weights", response_model=list[schemas.ModelWeightRead])
def list_model_weights(db: Session = Depends(get_db)):
    return crud.list_all(db, models.ModelWeight)


@app.post("/model-weights", response_model=schemas.ModelWeightRead)
def create_model_weight(payload: schemas.ModelWeightCreate, db: Session = Depends(get_db)):
    if payload.project_id:
        _get_or_404(db, models.Project, payload.project_id, "Project")
    return crud.create_item(db, models.ModelWeight, payload.model_dump())


@app.delete("/model-weights/{weight_id}")
def delete_model_weight(weight_id: int, db: Session = Depends(get_db)):
    item = _get_or_404(db, models.ModelWeight, weight_id, "ModelWeight")
    crud.delete_item(db, item)
    return {"ok": True}


def init_db() -> None:
    Base.metadata.create_all(bind=engine)  # pragma: no cover
