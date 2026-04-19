import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useEffect, useMemo, useState } from "react";
import Plot from "react-plotly.js";
const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
async function api(path, init) {
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
    const [status, setStatus] = useState("");
    const [users, setUsers] = useState([]);
    const [species, setSpecies] = useState([]);
    const [projects, setProjects] = useState([]);
    const [probes, setProbes] = useState([]);
    const [samples, setSamples] = useState([]);
    const [images, setImages] = useState([]);
    const [results, setResults] = useState([]);
    const [uploadSampleId, setUploadSampleId] = useState("");
    const [uploadFiles, setUploadFiles] = useState(null);
    const [newUser, setNewUser] = useState({ email: "demo@bees.local", full_name: "Demo Researcher" });
    const [newSpecies, setNewSpecies] = useState({ name: "", latin_name: "" });
    const [newProject, setNewProject] = useState({ user_id: "", name: "", description: "" });
    const [newProbe, setNewProbe] = useState({ project_id: "", species_id: "", name: "", treatment_type: "control", notes: "" });
    const [newSample, setNewSample] = useState({ probe_id: "", replicate_label: "1" });
    const refreshAll = async () => {
        setStatus("Loading...");
        try {
            const [u, s, p, pr, sm, im, rs] = await Promise.all([
                api("/users"),
                api("/species"),
                api("/projects"),
                api("/probes"),
                api("/samples"),
                api("/images"),
                api("/probe-results")
            ]);
            setUsers(u);
            setSpecies(s);
            setProjects(p);
            setProbes(pr);
            setSamples(sm);
            setImages(im);
            setResults(rs);
            setStatus("");
        }
        catch (error) {
            setStatus(error.message);
        }
    };
    useEffect(() => {
        void refreshAll();
    }, []);
    const speciesMap = useMemo(() => new Map(species.map((s) => [s.id, s.name])), [species]);
    const probeMap = useMemo(() => new Map(probes.map((p) => [p.id, p])), [probes]);
    const submitUser = async (event) => {
        event.preventDefault();
        await api("/users", { method: "POST", body: JSON.stringify(newUser) });
        await refreshAll();
    };
    const submitSpecies = async (event) => {
        event.preventDefault();
        await api("/species", { method: "POST", body: JSON.stringify(newSpecies) });
        setNewSpecies({ name: "", latin_name: "" });
        await refreshAll();
    };
    const submitProject = async (event) => {
        event.preventDefault();
        await api("/projects", {
            method: "POST",
            body: JSON.stringify({ ...newProject, user_id: Number(newProject.user_id) })
        });
        setNewProject({ user_id: "", name: "", description: "" });
        await refreshAll();
    };
    const submitProbe = async (event) => {
        event.preventDefault();
        await api("/probes", {
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
    const submitSample = async (event) => {
        event.preventDefault();
        await api("/samples", {
            method: "POST",
            body: JSON.stringify({ ...newSample, probe_id: Number(newSample.probe_id) })
        });
        setNewSample({ probe_id: "", replicate_label: "1" });
        await refreshAll();
    };
    const uploadImages = async (event) => {
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
    const remove = async (path) => {
        await api(path, { method: "DELETE" });
        await refreshAll();
    };
    const editProbe = async (probe) => {
        const newName = prompt("Новое название пробы", probe.name);
        if (!newName)
            return;
        const newTreatment = prompt("Новый тип обработки", probe.treatment_type);
        if (!newTreatment)
            return;
        await api(`/probes/${probe.id}`, {
            method: "PUT",
            body: JSON.stringify({ name: newName, treatment_type: newTreatment })
        });
        await refreshAll();
    };
    const runAnalysis = async (probeId) => {
        await api(`/probes/${probeId}/analyze?mode=yolo`, { method: "POST" });
        await refreshAll();
    };
    return (_jsxs("div", { className: "container", children: [_jsx("h1", { children: "Bee BioData Platform" }), status ? _jsx("p", { className: "status", children: status }) : null, _jsxs("div", { className: "row", children: [_jsxs("section", { className: "card", children: [_jsx("h2", { children: "Users" }), _jsxs("form", { onSubmit: submitUser, children: [_jsx("input", { value: newUser.email, onChange: (e) => setNewUser({ ...newUser, email: e.target.value }), placeholder: "email" }), _jsx("input", { value: newUser.full_name, onChange: (e) => setNewUser({ ...newUser, full_name: e.target.value }), placeholder: "full name" }), _jsx("button", { type: "submit", children: "Add" })] }), _jsxs("table", { children: [_jsx("thead", { children: _jsxs("tr", { children: [_jsx("th", { children: "ID" }), _jsx("th", { children: "Email" }), _jsx("th", { children: "Name" })] }) }), _jsx("tbody", { children: users.map((u) => _jsxs("tr", { children: [_jsx("td", { children: u.id }), _jsx("td", { children: u.email }), _jsx("td", { children: u.full_name })] }, u.id)) })] })] }), _jsxs("section", { className: "card", children: [_jsx("h2", { children: "Species" }), _jsxs("form", { onSubmit: submitSpecies, children: [_jsx("input", { value: newSpecies.name, onChange: (e) => setNewSpecies({ ...newSpecies, name: e.target.value }), placeholder: "species" }), _jsx("input", { value: newSpecies.latin_name, onChange: (e) => setNewSpecies({ ...newSpecies, latin_name: e.target.value }), placeholder: "latin name" }), _jsx("button", { type: "submit", children: "Add" })] }), _jsxs("table", { children: [_jsx("thead", { children: _jsxs("tr", { children: [_jsx("th", { children: "ID" }), _jsx("th", { children: "Name" }), _jsx("th", { children: "Latin" }), _jsx("th", {})] }) }), _jsx("tbody", { children: species.map((s) => _jsxs("tr", { children: [_jsx("td", { children: s.id }), _jsx("td", { children: s.name }), _jsx("td", { children: s.latin_name }), _jsx("td", { children: _jsx("button", { className: "danger", onClick: () => void remove(`/species/${s.id}`), children: "Delete" }) })] }, s.id)) })] })] })] }), _jsxs("section", { className: "card", children: [_jsx("h2", { children: "Projects" }), _jsxs("form", { onSubmit: submitProject, children: [_jsxs("select", { value: newProject.user_id, onChange: (e) => setNewProject({ ...newProject, user_id: e.target.value }), children: [_jsx("option", { value: "", children: "User" }), users.map((u) => _jsx("option", { value: u.id, children: u.full_name }, u.id))] }), _jsx("input", { value: newProject.name, onChange: (e) => setNewProject({ ...newProject, name: e.target.value }), placeholder: "project name" }), _jsx("input", { value: newProject.description, onChange: (e) => setNewProject({ ...newProject, description: e.target.value }), placeholder: "description" }), _jsx("button", { type: "submit", children: "Add" })] }), _jsxs("table", { children: [_jsx("thead", { children: _jsxs("tr", { children: [_jsx("th", { children: "ID" }), _jsx("th", { children: "User" }), _jsx("th", { children: "Name" }), _jsx("th", { children: "Description" }), _jsx("th", {})] }) }), _jsx("tbody", { children: projects.map((p) => _jsxs("tr", { children: [_jsx("td", { children: p.id }), _jsx("td", { children: p.user_id }), _jsx("td", { children: p.name }), _jsx("td", { children: p.description }), _jsx("td", { children: _jsx("button", { className: "danger", onClick: () => void remove(`/projects/${p.id}`), children: "Delete" }) })] }, p.id)) })] })] }), _jsxs("section", { className: "card", children: [_jsx("h2", { children: "Probes" }), _jsxs("form", { onSubmit: submitProbe, children: [_jsxs("select", { value: newProbe.project_id, onChange: (e) => setNewProbe({ ...newProbe, project_id: e.target.value }), children: [_jsx("option", { value: "", children: "Project" }), projects.map((p) => _jsx("option", { value: p.id, children: p.name }, p.id))] }), _jsxs("select", { value: newProbe.species_id, onChange: (e) => setNewProbe({ ...newProbe, species_id: e.target.value }), children: [_jsx("option", { value: "", children: "Species" }), species.map((s) => _jsx("option", { value: s.id, children: s.name }, s.id))] }), _jsx("input", { value: newProbe.name, onChange: (e) => setNewProbe({ ...newProbe, name: e.target.value }), placeholder: "probe name" }), _jsx("input", { value: newProbe.treatment_type, onChange: (e) => setNewProbe({ ...newProbe, treatment_type: e.target.value }), placeholder: "treatment" }), _jsx("button", { type: "submit", children: "Add" })] }), _jsxs("table", { children: [_jsx("thead", { children: _jsxs("tr", { children: [_jsx("th", { children: "ID" }), _jsx("th", { children: "Probe" }), _jsx("th", { children: "Type" }), _jsx("th", { children: "Species" }), _jsx("th", { children: "Actions" })] }) }), _jsx("tbody", { children: probes.map((p) => (_jsxs("tr", { children: [_jsx("td", { children: p.id }), _jsx("td", { children: p.name }), _jsx("td", { children: p.treatment_type }), _jsx("td", { children: speciesMap.get(p.species_id) ?? p.species_id }), _jsxs("td", { children: [_jsx("button", { onClick: () => void editProbe(p), children: "Edit" }), " ", _jsx("button", { onClick: () => void runAnalysis(p.id), children: "Analyze" }), " ", _jsx("button", { className: "danger", onClick: () => void remove(`/probes/${p.id}`), children: "Delete" })] })] }, p.id))) })] })] }), _jsxs("section", { className: "card", children: [_jsx("h2", { children: "Samples" }), _jsxs("form", { onSubmit: submitSample, children: [_jsxs("select", { value: newSample.probe_id, onChange: (e) => setNewSample({ ...newSample, probe_id: e.target.value }), children: [_jsx("option", { value: "", children: "Probe" }), probes.map((p) => _jsx("option", { value: p.id, children: p.name }, p.id))] }), _jsx("input", { value: newSample.replicate_label, onChange: (e) => setNewSample({ ...newSample, replicate_label: e.target.value }), placeholder: "replicate" }), _jsx("button", { type: "submit", children: "Add" })] }), _jsxs("table", { children: [_jsx("thead", { children: _jsxs("tr", { children: [_jsx("th", { children: "ID" }), _jsx("th", { children: "Probe" }), _jsx("th", { children: "Replicate" }), _jsx("th", {})] }) }), _jsx("tbody", { children: samples.map((s) => _jsxs("tr", { children: [_jsx("td", { children: s.id }), _jsx("td", { children: probeMap.get(s.probe_id)?.name ?? s.probe_id }), _jsx("td", { children: s.replicate_label }), _jsx("td", { children: _jsx("button", { className: "danger", onClick: () => void remove(`/samples/${s.id}`), children: "Delete" }) })] }, s.id)) })] })] }), _jsxs("section", { className: "card", children: [_jsx("h2", { children: "Image Upload" }), _jsxs("form", { onSubmit: (e) => void uploadImages(e), children: [_jsxs("select", { value: uploadSampleId, onChange: (e) => setUploadSampleId(Number(e.target.value)), children: [_jsx("option", { value: "", children: "Sample" }), samples.map((s) => _jsx("option", { value: s.id, children: `${s.id}: ${probeMap.get(s.probe_id)?.name ?? "probe"} / ${s.replicate_label}` }, s.id))] }), _jsx("input", { type: "file", multiple: true, onChange: (e) => setUploadFiles(e.target.files) }), _jsx("button", { type: "submit", children: "Upload JPG/PNG" })] }), _jsxs("table", { children: [_jsx("thead", { children: _jsxs("tr", { children: [_jsx("th", { children: "ID" }), _jsx("th", { children: "Sample" }), _jsx("th", { children: "File" }), _jsx("th", {})] }) }), _jsx("tbody", { children: images.map((img) => _jsxs("tr", { children: [_jsx("td", { children: img.id }), _jsx("td", { children: img.sample_id }), _jsx("td", { children: img.filename }), _jsx("td", { children: _jsx("button", { className: "danger", onClick: () => void remove(`/images/${img.id}`), children: "Delete" }) })] }, img.id)) })] })] }), _jsxs("section", { className: "card", children: [_jsx("h2", { children: "Results Dashboard" }), _jsxs("table", { children: [_jsx("thead", { children: _jsxs("tr", { children: [_jsx("th", { children: "Species" }), _jsx("th", { children: "Probe" }), _jsx("th", { children: "Mean" }), _jsx("th", { children: "Std" }), _jsx("th", { children: "n" }), _jsx("th", { children: "P-value" }), _jsx("th", { children: "Method" })] }) }), _jsx("tbody", { children: results.map((r) => {
                                    const probe = probeMap.get(r.probe_id);
                                    return (_jsxs("tr", { children: [_jsx("td", { children: speciesMap.get(probe?.species_id || 0) ?? "-" }), _jsx("td", { children: probe?.name ?? r.probe_id }), _jsx("td", { children: r.mean_titer.toFixed(3) }), _jsx("td", { children: r.std_titer.toFixed(3) }), _jsx("td", { children: r.n_measurements }), _jsx("td", { children: r.p_value?.toFixed(6) ?? "N/A" }), _jsx("td", { children: r.method })] }, r.id));
                                }) })] }), _jsx(Plot, { data: [
                            {
                                type: "bar",
                                x: results.map((r) => probeMap.get(r.probe_id)?.name ?? String(r.probe_id)),
                                y: results.map((r) => r.mean_titer),
                                error_y: { type: "data", array: results.map((r) => r.std_titer), visible: true }
                            }
                        ], layout: {
                            width: 1000,
                            height: 360,
                            title: { text: "Mean titer per probe" },
                            xaxis: { title: { text: "Probe" } },
                            yaxis: { title: { text: "Mean titer (million spores/ml)" } }
                        } })] })] }));
}
