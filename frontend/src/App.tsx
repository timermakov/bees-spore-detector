import { FormEvent, useEffect, useMemo, useState } from "react";
import Plot from "react-plotly.js";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

type User = { id: number; email: string; full_name: string };
type Species = { id: number; name: string; latin_name?: string | null };
type Project = { id: number; user_id: number; name: string; description?: string | null };
type Probe = {
  id: number;
  project_id: number;
  species_id: number;
  name: string;
  treatment_type: string;
  notes?: string | null;
};
type Sample = { id: number; probe_id: number; replicate_label: string };
type ImageRec = { id: number; sample_id: number; filename: string; file_path: string };
type ProbeResult = {
  id: number;
  probe_id: number;
  mean_titer: number;
  std_titer: number;
  n_measurements: number;
  p_value: number | null;
  method: string;
};

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed: ${response.status}`);
  }
  return response.json();
}

export default function App() {
  const [status, setStatus] = useState<string>("");
  const [users, setUsers] = useState<User[]>([]);
  const [species, setSpecies] = useState<Species[]>([]);
  const [projects, setProjects] = useState<Project[]>([]);
  const [probes, setProbes] = useState<Probe[]>([]);
  const [samples, setSamples] = useState<Sample[]>([]);
  const [images, setImages] = useState<ImageRec[]>([]);
  const [results, setResults] = useState<ProbeResult[]>([]);
  const [uploadSampleId, setUploadSampleId] = useState<number | "">("");
  const [uploadFiles, setUploadFiles] = useState<FileList | null>(null);

  const [newUser, setNewUser] = useState({ email: "demo@bees.local", full_name: "Demo Researcher" });
  const [newSpecies, setNewSpecies] = useState({ name: "", latin_name: "" });
  const [newProject, setNewProject] = useState({ user_id: "", name: "", description: "" });
  const [newProbe, setNewProbe] = useState({ project_id: "", species_id: "", name: "", treatment_type: "control", notes: "" });
  const [newSample, setNewSample] = useState({ probe_id: "", replicate_label: "1" });

  const refreshAll = async () => {
    setStatus("Loading...");
    try {
      const [u, s, p, pr, sm, im, rs] = await Promise.all([
        api<User[]>("/users"),
        api<Species[]>("/species"),
        api<Project[]>("/projects"),
        api<Probe[]>("/probes"),
        api<Sample[]>("/samples"),
        api<ImageRec[]>("/images"),
        api<ProbeResult[]>("/probe-results")
      ]);
      setUsers(u);
      setSpecies(s);
      setProjects(p);
      setProbes(pr);
      setSamples(sm);
      setImages(im);
      setResults(rs);
      setStatus("");
    } catch (error) {
      setStatus((error as Error).message);
    }
  };

  useEffect(() => {
    void refreshAll();
  }, []);

  const speciesMap = useMemo(() => new Map(species.map((s) => [s.id, s.name])), [species]);
  const probeMap = useMemo(() => new Map(probes.map((p) => [p.id, p])), [probes]);

  const submitUser = async (event: FormEvent) => {
    event.preventDefault();
    await api<User>("/users", { method: "POST", body: JSON.stringify(newUser) });
    await refreshAll();
  };

  const submitSpecies = async (event: FormEvent) => {
    event.preventDefault();
    await api<Species>("/species", { method: "POST", body: JSON.stringify(newSpecies) });
    setNewSpecies({ name: "", latin_name: "" });
    await refreshAll();
  };

  const submitProject = async (event: FormEvent) => {
    event.preventDefault();
    await api<Project>("/projects", {
      method: "POST",
      body: JSON.stringify({ ...newProject, user_id: Number(newProject.user_id) })
    });
    setNewProject({ user_id: "", name: "", description: "" });
    await refreshAll();
  };

  const submitProbe = async (event: FormEvent) => {
    event.preventDefault();
    await api<Probe>("/probes", {
      method: "POST",
      body: JSON.stringify({
        ...newProbe,
        project_id: Number(newProbe.project_id),
        species_id: Number(newProbe.species_id)
      })
    });
    setNewProbe({ project_id: "", species_id: "", name: "", treatment_type: "control", notes: "" });
    await refreshAll();
  };

  const submitSample = async (event: FormEvent) => {
    event.preventDefault();
    await api<Sample>("/samples", {
      method: "POST",
      body: JSON.stringify({ ...newSample, probe_id: Number(newSample.probe_id) })
    });
    setNewSample({ probe_id: "", replicate_label: "1" });
    await refreshAll();
  };

  const uploadImages = async (event: FormEvent) => {
    event.preventDefault();
    if (!uploadSampleId || !uploadFiles || uploadFiles.length === 0) {
      return;
    }
    const data = new FormData();
    Array.from(uploadFiles).forEach((file) => data.append("files", file));
    const response = await fetch(`${API_BASE}/samples/${uploadSampleId}/images`, {
      method: "POST",
      body: data
    });
    if (!response.ok) {
      throw new Error(await response.text());
    }
    setUploadFiles(null);
    await refreshAll();
  };

  const remove = async (path: string) => {
    await api<{ ok: boolean }>(path, { method: "DELETE" });
    await refreshAll();
  };

  const editProbe = async (probe: Probe) => {
    const newName = prompt("Новое название пробы", probe.name);
    if (!newName) return;
    const newTreatment = prompt("Новый тип обработки", probe.treatment_type);
    if (!newTreatment) return;
    await api<Probe>(`/probes/${probe.id}`, {
      method: "PUT",
      body: JSON.stringify({ name: newName, treatment_type: newTreatment })
    });
    await refreshAll();
  };

  const runAnalysis = async (probeId: number) => {
    await api(`/probes/${probeId}/analyze?mode=yolo`, { method: "POST" });
    await refreshAll();
  };

  return (
    <div className="container">
      <h1>Bee BioData Platform</h1>
      {status ? <p className="status">{status}</p> : null}

      <div className="row">
        <section className="card">
          <h2>Users</h2>
          <form onSubmit={submitUser}>
            <input value={newUser.email} onChange={(e) => setNewUser({ ...newUser, email: e.target.value })} placeholder="email" />
            <input value={newUser.full_name} onChange={(e) => setNewUser({ ...newUser, full_name: e.target.value })} placeholder="full name" />
            <button type="submit">Add</button>
          </form>
          <table><thead><tr><th>ID</th><th>Email</th><th>Name</th></tr></thead><tbody>{users.map((u) => <tr key={u.id}><td>{u.id}</td><td>{u.email}</td><td>{u.full_name}</td></tr>)}</tbody></table>
        </section>

        <section className="card">
          <h2>Species</h2>
          <form onSubmit={submitSpecies}>
            <input value={newSpecies.name} onChange={(e) => setNewSpecies({ ...newSpecies, name: e.target.value })} placeholder="species" />
            <input value={newSpecies.latin_name} onChange={(e) => setNewSpecies({ ...newSpecies, latin_name: e.target.value })} placeholder="latin name" />
            <button type="submit">Add</button>
          </form>
          <table>
            <thead><tr><th>ID</th><th>Name</th><th>Latin</th><th /></tr></thead>
            <tbody>{species.map((s) => <tr key={s.id}><td>{s.id}</td><td>{s.name}</td><td>{s.latin_name}</td><td><button className="danger" onClick={() => void remove(`/species/${s.id}`)}>Delete</button></td></tr>)}</tbody>
          </table>
        </section>
      </div>

      <section className="card">
        <h2>Projects</h2>
        <form onSubmit={submitProject}>
          <select value={newProject.user_id} onChange={(e) => setNewProject({ ...newProject, user_id: e.target.value })}>
            <option value="">User</option>
            {users.map((u) => <option key={u.id} value={u.id}>{u.full_name}</option>)}
          </select>
          <input value={newProject.name} onChange={(e) => setNewProject({ ...newProject, name: e.target.value })} placeholder="project name" />
          <input value={newProject.description} onChange={(e) => setNewProject({ ...newProject, description: e.target.value })} placeholder="description" />
          <button type="submit">Add</button>
        </form>
        <table>
          <thead><tr><th>ID</th><th>User</th><th>Name</th><th>Description</th><th /></tr></thead>
          <tbody>{projects.map((p) => <tr key={p.id}><td>{p.id}</td><td>{p.user_id}</td><td>{p.name}</td><td>{p.description}</td><td><button className="danger" onClick={() => void remove(`/projects/${p.id}`)}>Delete</button></td></tr>)}</tbody>
        </table>
      </section>

      <section className="card">
        <h2>Probes</h2>
        <form onSubmit={submitProbe}>
          <select value={newProbe.project_id} onChange={(e) => setNewProbe({ ...newProbe, project_id: e.target.value })}>
            <option value="">Project</option>
            {projects.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
          </select>
          <select value={newProbe.species_id} onChange={(e) => setNewProbe({ ...newProbe, species_id: e.target.value })}>
            <option value="">Species</option>
            {species.map((s) => <option key={s.id} value={s.id}>{s.name}</option>)}
          </select>
          <input value={newProbe.name} onChange={(e) => setNewProbe({ ...newProbe, name: e.target.value })} placeholder="probe name" />
          <input value={newProbe.treatment_type} onChange={(e) => setNewProbe({ ...newProbe, treatment_type: e.target.value })} placeholder="treatment" />
          <button type="submit">Add</button>
        </form>
        <table>
          <thead><tr><th>ID</th><th>Probe</th><th>Type</th><th>Species</th><th>Actions</th></tr></thead>
          <tbody>
            {probes.map((p) => (
              <tr key={p.id}>
                <td>{p.id}</td>
                <td>{p.name}</td>
                <td>{p.treatment_type}</td>
                <td>{speciesMap.get(p.species_id) ?? p.species_id}</td>
                <td>
                  <button onClick={() => void editProbe(p)}>Edit</button>{" "}
                  <button onClick={() => void runAnalysis(p.id)}>Analyze</button>{" "}
                  <button className="danger" onClick={() => void remove(`/probes/${p.id}`)}>Delete</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <section className="card">
        <h2>Samples</h2>
        <form onSubmit={submitSample}>
          <select value={newSample.probe_id} onChange={(e) => setNewSample({ ...newSample, probe_id: e.target.value })}>
            <option value="">Probe</option>
            {probes.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
          </select>
          <input value={newSample.replicate_label} onChange={(e) => setNewSample({ ...newSample, replicate_label: e.target.value })} placeholder="replicate" />
          <button type="submit">Add</button>
        </form>
        <table>
          <thead><tr><th>ID</th><th>Probe</th><th>Replicate</th><th /></tr></thead>
          <tbody>{samples.map((s) => <tr key={s.id}><td>{s.id}</td><td>{probeMap.get(s.probe_id)?.name ?? s.probe_id}</td><td>{s.replicate_label}</td><td><button className="danger" onClick={() => void remove(`/samples/${s.id}`)}>Delete</button></td></tr>)}</tbody>
        </table>
      </section>

      <section className="card">
        <h2>Image Upload</h2>
        <form onSubmit={(e) => void uploadImages(e)}>
          <select value={uploadSampleId} onChange={(e) => setUploadSampleId(Number(e.target.value))}>
            <option value="">Sample</option>
            {samples.map((s) => <option key={s.id} value={s.id}>{`${s.id}: ${probeMap.get(s.probe_id)?.name ?? "probe"} / ${s.replicate_label}`}</option>)}
          </select>
          <input type="file" multiple onChange={(e) => setUploadFiles(e.target.files)} />
          <button type="submit">Upload JPG/PNG</button>
        </form>
        <table>
          <thead><tr><th>ID</th><th>Sample</th><th>File</th><th /></tr></thead>
          <tbody>{images.map((img) => <tr key={img.id}><td>{img.id}</td><td>{img.sample_id}</td><td>{img.filename}</td><td><button className="danger" onClick={() => void remove(`/images/${img.id}`)}>Delete</button></td></tr>)}</tbody>
        </table>
      </section>

      <section className="card">
        <h2>Results Dashboard</h2>
        <table>
          <thead><tr><th>Species</th><th>Probe</th><th>Mean</th><th>Std</th><th>n</th><th>P-value</th><th>Method</th></tr></thead>
          <tbody>
            {results.map((r) => {
              const probe = probeMap.get(r.probe_id);
              return (
                <tr key={r.id}>
                  <td>{speciesMap.get(probe?.species_id || 0) ?? "-"}</td>
                  <td>{probe?.name ?? r.probe_id}</td>
                  <td>{r.mean_titer.toFixed(3)}</td>
                  <td>{r.std_titer.toFixed(3)}</td>
                  <td>{r.n_measurements}</td>
                  <td>{r.p_value?.toFixed(6) ?? "N/A"}</td>
                  <td>{r.method}</td>
                </tr>
              );
            })}
          </tbody>
        </table>

        <Plot
          data={[
            {
              type: "bar",
              x: results.map((r) => probeMap.get(r.probe_id)?.name ?? String(r.probe_id)),
              y: results.map((r) => r.mean_titer),
              error_y: { type: "data", array: results.map((r) => r.std_titer), visible: true }
            }
          ]}
          layout={{
            width: 1000,
            height: 360,
            title: { text: "Mean titer per probe" },
            xaxis: { title: { text: "Probe" } },
            yaxis: { title: { text: "Mean titer (million spores/ml)" } }
          }}
        />
      </section>
    </div>
  );
}
