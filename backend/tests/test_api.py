from io import BytesIO


def _create_user(client):
    return client.post("/users", json={"email": "u@x.com", "full_name": "User"})


def _create_species(client):
    return client.post("/species", json={"name": "Nosema bombycis", "latin_name": "Nosema bombycis"})


def _create_project(client, user_id: int):
    return client.post(
        "/projects",
        json={"user_id": user_id, "name": "P1", "description": "D"},
    )


def _create_probe(client, project_id: int, species_id: int):
    return client.post(
        "/probes",
        json={
            "project_id": project_id,
            "species_id": species_id,
            "name": "delta",
            "treatment_type": "dsRNA",
            "notes": "test",
        },
    )


def _create_sample(client, probe_id: int):
    return client.post("/samples", json={"probe_id": probe_id, "replicate_label": "1"})


def test_crud_flow(client):
    user = _create_user(client)
    assert user.status_code == 200
    user_id = user.json()["id"]

    species = _create_species(client)
    assert species.status_code == 200
    species_id = species.json()["id"]

    project = _create_project(client, user_id)
    assert project.status_code == 200
    project_id = project.json()["id"]

    project_updated = client.put(
        f"/projects/{project_id}",
        json={"name": "P2", "description": "updated"},
    )
    assert project_updated.status_code == 200
    assert project_updated.json()["name"] == "P2"

    probe = _create_probe(client, project_id, species_id)
    assert probe.status_code == 200
    probe_id = probe.json()["id"]

    sample = _create_sample(client, probe_id)
    assert sample.status_code == 200
    sample_id = sample.json()["id"]

    image_resp = client.post(
        f"/samples/{sample_id}/images",
        files=[("files", ("test.jpg", BytesIO(b"fake"), "image/jpeg"))],
    )
    assert image_resp.status_code == 200
    image_id = image_resp.json()[0]["id"]

    delete_image = client.delete(f"/images/{image_id}")
    assert delete_image.status_code == 200

    delete_sample = client.delete(f"/samples/{sample_id}")
    assert delete_sample.status_code == 200
    delete_probe = client.delete(f"/probes/{probe_id}")
    assert delete_probe.status_code == 200
    delete_project = client.delete(f"/projects/{project_id}")
    assert delete_project.status_code == 200


def test_probe_results_upsert(client):
    user_id = _create_user(client).json()["id"]
    species_id = _create_species(client).json()["id"]
    project_id = _create_project(client, user_id).json()["id"]
    probe_id = _create_probe(client, project_id, species_id).json()["id"]

    payload = {
        "probe_id": probe_id,
        "mean_titer": 10.1,
        "std_titer": 1.2,
        "n_measurements": 3,
        "p_value": 0.05,
        "method": "manual",
    }
    r1 = client.post("/probe-results", json=payload)
    assert r1.status_code == 200

    payload["mean_titer"] = 20.2
    r2 = client.post("/probe-results", json=payload)
    assert r2.status_code == 200
    assert r2.json()["mean_titer"] == 20.2


def test_analyze_endpoint(client):
    user_id = _create_user(client).json()["id"]
    species_id = _create_species(client).json()["id"]
    project_id = _create_project(client, user_id).json()["id"]
    probe_id = _create_probe(client, project_id, species_id).json()["id"]
    sample_id = _create_sample(client, probe_id).json()["id"]

    client.post(
        f"/samples/{sample_id}/images",
        files=[("files", ("fake.jpg", BytesIO(b"not-image"), "image/jpeg"))],
    )
    analyze = client.post(f"/probes/{probe_id}/analyze?mode=opencv")
    assert analyze.status_code == 200
    assert analyze.json()["probe_id"] == probe_id
