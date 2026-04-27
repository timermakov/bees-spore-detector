from datetime import datetime

from pydantic import BaseModel


class UserBase(BaseModel):
    email: str
    full_name: str


class UserCreate(UserBase):
    pass


class UserRead(UserBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True


class SpeciesBase(BaseModel):
    name: str
    latin_name: str | None = None


class SpeciesCreate(SpeciesBase):
    pass


class SpeciesRead(SpeciesBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True


class ProjectBase(BaseModel):
    user_id: int
    name: str
    description: str | None = None


class ProjectCreate(ProjectBase):
    pass


class ProjectUpdate(BaseModel):
    user_id: int | None = None
    name: str | None = None
    description: str | None = None


class ProjectRead(ProjectBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True


class ProbeBase(BaseModel):
    project_id: int
    species_id: int
    name: str
    treatment_type: str
    notes: str | None = None


class ProbeCreate(ProbeBase):
    pass


class ProbeUpdate(BaseModel):
    project_id: int | None = None
    species_id: int | None = None
    name: str | None = None
    treatment_type: str | None = None
    notes: str | None = None


class ProbeRead(ProbeBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True


class SampleBase(BaseModel):
    probe_id: int
    replicate_label: str
    collected_at: datetime | None = None


class SampleCreate(SampleBase):
    pass


class SampleUpdate(BaseModel):
    probe_id: int | None = None
    replicate_label: str | None = None
    collected_at: datetime | None = None


class SampleRead(SampleBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True


class MicroImageRead(BaseModel):
    id: int
    sample_id: int
    filename: str
    file_path: str
    uploaded_at: datetime

    class Config:
        from_attributes = True


class ProbeResultRead(BaseModel):
    id: int
    probe_id: int
    mean_titer: float
    std_titer: float
    n_measurements: int
    p_value: float | None
    method: str
    updated_at: datetime

    class Config:
        from_attributes = True


class ProbeResultUpsert(BaseModel):
    probe_id: int
    mean_titer: float
    std_titer: float
    n_measurements: int
    p_value: float | None = None
    method: str = "manual"


class ModelWeightBase(BaseModel):
    project_id: int | None = None
    name: str
    path: str
    is_active: bool = False


class ModelWeightCreate(ModelWeightBase):
    pass


class ModelWeightRead(ModelWeightBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True


class AnalyzeResponse(BaseModel):
    probe_id: int
    mean_titer: float
    std_titer: float
    n_measurements: int
    p_value: float | None
    method: str
