import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  apiBaseUrl,
  createRelightJob,
  getRelightJob,
  getRelightResult,
  resolveApiUrl,
} from "./api/sunitApi.js";
import "./index.css";

const MAX_SIZE_BYTES = 15 * 1024 * 1024;
const MAX_SIDE = 4096;
const PROCESSING_SIDE = 1536;
const MIN_SIDE = 128;

const presets = [
  { id: "left", name: "Left", vector: [-0.55, -0.12, 0.82], point: [27, 50] },
  { id: "right", name: "Right", vector: [0.55, -0.12, 0.82], point: [73, 50] },
  { id: "top-left", name: "Top Left", vector: [-0.45, -0.48, 0.76], point: [31, 29] },
  { id: "top-right", name: "Top Right", vector: [0.45, -0.48, 0.76], point: [69, 29] },
  { id: "soft-front", name: "Soft Front", vector: [0.12, -0.08, 1.0], point: [57, 48] },
  { id: "dramatic-side", name: "Dramatic Side", vector: [0.82, 0.18, 0.54], point: [83, 61] },
];

const stylePresets = [
  {
    id: "natural",
    name: "Natural",
    values: {
      personStrength: 0.65,
      ambient: 0.38,
      highlight: 0.08,
      boundaryRelight: 0.30,
      shadowRelight: 0.45,
    },
  },
  {
    id: "soft_portrait",
    name: "Soft Portrait",
    values: {
      personStrength: 0.55,
      ambient: 0.44,
      highlight: 0.06,
      boundaryRelight: 0.25,
      shadowRelight: 0.38,
    },
  },
  {
    id: "dramatic",
    name: "Dramatic",
    values: {
      personStrength: 0.85,
      ambient: 0.28,
      highlight: 0.16,
      boundaryRelight: 0.35,
      shadowRelight: 0.55,
    },
  },
];

const STAGE_LABELS = {
  queued: "Waiting for the GPU worker",
  validating: "Checking your image",
  preparing_input: "Preparing the portrait",
  estimating_normals: "Understanding surface geometry",
  generating_matte: "Separating subject and background",
  relighting: "Applying the new light",
  saving_output: "Saving your result",
  completed: "Relighting complete",
};

const STAGE_ORDER = Object.keys(STAGE_LABELS);

const pipeline = [
  ["Image preprocessing", "Validate, resize, and normalize the portrait before inference."],
  ["DSINE normal estimation", "Predict a detailed surface normal map from the single image."],
  ["BiRefNet soft matte", "Separate the portrait with soft hair and clothing boundaries."],
  ["V8 ratio relighting", "Apply geometry-aware shading ratios from the target light vector."],
  ["Exact background restoration", "Composite the original background back into the final frame."],
  ["Final output", "Save a downloadable PNG and optional diagnostic maps."],
];

const steps = STAGE_ORDER.map((stage) => STAGE_LABELS[stage]);

function formatBytes(bytes) {
  if (!bytes) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const exponent = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / 1024 ** exponent).toFixed(exponent ? 1 : 0)} ${units[exponent]}`;
}

function normalizeVector(x, y, z = 0.74) {
  const length = Math.max(Math.hypot(x, y, z), 0.0001);
  return [x / length, y / length, z / length];
}

async function readImageDimensions(file) {
  const objectUrl = URL.createObjectURL(file);
  try {
    return await new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = () => resolve({ width: image.naturalWidth, height: image.naturalHeight });
      image.onerror = () => reject(new Error("Could not read image dimensions."));
      image.src = objectUrl;
    });
  } finally {
    URL.revokeObjectURL(objectUrl);
  }
}

async function canvasEncode(file, maxSide, forceReencode) {
  const objectUrl = URL.createObjectURL(file);
  try {
    const image = await new Promise((resolve, reject) => {
      const element = new Image();
      element.onload = () => resolve(element);
      element.onerror = () => reject(new Error("Could not prepare image for upload."));
      element.src = objectUrl;
    });

    const scale = Math.min(1, maxSide / Math.max(image.naturalWidth, image.naturalHeight));
    if (scale === 1 && !forceReencode) return file;

    const canvas = document.createElement("canvas");
    canvas.width = Math.round(image.naturalWidth * scale);
    canvas.height = Math.round(image.naturalHeight * scale);
    const context = canvas.getContext("2d", { alpha: false });
    context.drawImage(image, 0, 0, canvas.width, canvas.height);

    const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.92));
    if (!blob) throw new Error("Could not encode resized image.");
    return new File([blob], file.name.replace(/\.[^.]+$/, "") + "_sunit.jpg", {
      type: "image/jpeg",
      lastModified: Date.now(),
    });
  } finally {
    URL.revokeObjectURL(objectUrl);
  }
}

function inferFailureKind(message) {
  const text = (message || "").toLowerCase();
  if (text.includes("subject") || text.includes("portrait")) return "No clear portrait subject";
  if (text.includes("large") || text.includes("15 mb")) return "Image is too large";
  if (text.includes("jpg") || text.includes("webp") || text.includes("image type")) return "Invalid image type";
  if (text.includes("dsine") || text.includes("normal")) return "Could not estimate normals";
  if (text.includes("timeout")) return "API timeout";
  if (text.includes("unavailable") || text.includes("network") || text.includes("fetch")) return "Relighting service unavailable";
  return "Relight failed";
}

function App() {
  const fileInputRef = useRef(null);
  const abortRef = useRef(null);
  const pollRef = useRef(null);
  const timerRef = useRef(null);
  const lastStageRef = useRef("");

  const [file, setFile] = useState(null);
  const [uploadFile, setUploadFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [fileInfo, setFileInfo] = useState(null);
  const [validation, setValidation] = useState(null);
  const [selectedPreset, setSelectedPreset] = useState(presets[4]);
  const [stylePreset, setStylePresetState] = useState(stylePresets[0]);
  const [newLight, setNewLight] = useState(presets[4].vector);
  const [padPoint, setPadPoint] = useState(presets[4].point);
  const [settings, setSettings] = useState({
    ...stylePresets[0].values,
    advanced: false,
    autoResize: true,
    removeExif: true,
    showDebug: false,
  });
  const [job, setJob] = useState({
    status: "idle",
    id: "",
    stage: "queued",
    queueState: "Ready",
    elapsed: 0,
    error: "",
    technical: "",
    logs: [],
  });
  const [result, setResult] = useState(null);
  const [debugImages, setDebugImages] = useState({});
  const [compare, setCompare] = useState(50);
  const [progress, setProgress] = useState(0);

  const canGenerate = file && !validation && job.status !== "processing";
  const statusLabel = job.status === "idle" ? "Idle" : job.status[0].toUpperCase() + job.status.slice(1);
  const elapsedLabel = useMemo(() => {
    const minutes = String(Math.floor(job.elapsed / 60)).padStart(2, "0");
    const seconds = String(job.elapsed % 60).padStart(2, "0");
    return `${minutes}:${seconds}`;
  }, [job.elapsed]);

  const addLog = useCallback((message) => {
    setJob((current) => ({
      ...current,
      logs: [...current.logs.slice(-8), { time: new Date().toLocaleTimeString(), message }],
    }));
  }, []);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      abortRef.current?.abort();
      clearTimeout(pollRef.current);
      clearInterval(timerRef.current);
    };
  }, [previewUrl]);

  useEffect(() => {
    if (job.status !== "processing") return;
    timerRef.current = setInterval(() => {
      setJob((current) => ({ ...current, elapsed: current.elapsed + 1 }));
    }, 1000);
    return () => clearInterval(timerRef.current);
  }, [job.status]);

  const updateSetting = (key, value) => {
    setSettings((current) => ({ ...current, [key]: value }));
  };

  const setPreset = (preset) => {
    setSelectedPreset(preset);
    setNewLight(preset.vector);
    setPadPoint(preset.point);
  };

  const setStylePreset = (preset) => {
    setStylePresetState(preset);
    setSettings((current) => ({
      ...current,
      ...preset.values,
    }));
  };

  const validateAndSetFile = async (candidate) => {
    abortRef.current?.abort();
    clearTimeout(pollRef.current);
    lastStageRef.current = "";
    setResult(null);
    setDebugImages({});
    setProgress(0);
    setJob((current) => ({
      ...current,
      status: "idle",
      stage: "queued",
      error: "",
      technical: "",
      queueState: "Ready",
    }));

    if (!candidate) {
      setValidation({ title: "Upload an image to begin", detail: "Choose a JPG or PNG portrait." });
      return;
    }

    const allowed = ["image/jpeg", "image/png", "image/webp"];
    if (!allowed.includes(candidate.type)) {
      setFile(null);
      setUploadFile(null);
      setFileInfo({ name: candidate.name, size: candidate.size, type: candidate.type || "Unknown" });
      setValidation({
        title: "Unsupported file format",
        detail: "Please upload a JPG, PNG, or WEBP image.",
        action: "choose",
      });
      return;
    }

    let dimensions;
    try {
      dimensions = await readImageDimensions(candidate);
    } catch (error) {
      setValidation({ title: "Could not read image", detail: error.message });
      return;
    }

    if (Math.min(dimensions.width, dimensions.height) < MIN_SIDE) {
      setFile(null);
      setUploadFile(null);
      setFileInfo({ name: candidate.name, size: candidate.size, ...dimensions, type: candidate.type });
      setValidation({
        title: "Image dimensions are too small",
        detail: `Each side must be at least ${MIN_SIDE}px.`,
        action: "choose",
      });
      return;
    }

    if (candidate.size > MAX_SIZE_BYTES) {
      setFile(null);
      setUploadFile(null);
      setFileInfo({ name: candidate.name, size: candidate.size, ...dimensions, type: candidate.type });
      setValidation({
        title: "Image file is too large",
        detail: `Recommended max file size is ${formatBytes(MAX_SIZE_BYTES)}.`,
        action: "choose",
      });
      return;
    }

    const tooLarge = Math.max(dimensions.width, dimensions.height) > MAX_SIDE;
    if (tooLarge && !settings.autoResize) {
      setFile(null);
      setUploadFile(null);
      setFileInfo({ name: candidate.name, size: candidate.size, ...dimensions, type: candidate.type });
      setValidation({
        title: "Image dimensions are too large",
        detail: `Recommended max side is ${MAX_SIDE}px. Your image is ${dimensions.width} x ${dimensions.height}.`,
        action: "resize",
      });
      return;
    }

    try {
      const prepared = await canvasEncode(
        candidate,
        PROCESSING_SIDE,
        settings.removeExif || tooLarge,
      );
      const preparedDimensions = await readImageDimensions(prepared);
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      setFile(candidate);
      setUploadFile(prepared);
      setPreviewUrl(URL.createObjectURL(prepared));
      setFileInfo({
        name: candidate.name,
        uploadName: prepared.name,
        size: candidate.size,
        uploadSize: prepared.size,
        type: candidate.type,
        resized: prepared.name !== candidate.name || prepared.size !== candidate.size,
        ...preparedDimensions,
      });
      setValidation(null);
      addLog("Image loaded");
    } catch (error) {
      setValidation({ title: "Could not prepare image", detail: error.message });
    }
  };

  const onDrop = (event) => {
    event.preventDefault();
    validateAndSetFile(event.dataTransfer.files?.[0]);
  };

  const handlePadClick = (event) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const xPct = ((event.clientX - rect.left) / rect.width) * 100;
    const yPct = ((event.clientY - rect.top) / rect.height) * 100;
    const x = (xPct - 50) / 50;
    const y = (yPct - 50) / 50;
    setSelectedPreset({ id: "custom", name: "Custom", vector: normalizeVector(x, y), point: [xPct, yPct] });
    setNewLight(normalizeVector(x, y));
    setPadPoint([Math.max(8, Math.min(92, xPct)), Math.max(8, Math.min(92, yPct))]);
  };

  const finishFromPayload = (payload, fallbackJobId) => {
    const jobId = payload?.job_id || fallbackJobId;
    const outputUrl = resolveApiUrl(payload?.output_url);

    if (!outputUrl) {
      throw new Error("The API completed but did not return a usable output image.");
    }

    setResult(outputUrl);
    setDebugImages({
      normal: resolveApiUrl(`/outputs/${jobId}/intermediate/normal.png`),
      mask: resolveApiUrl(`/outputs/${jobId}/intermediate/person_alpha.png`),
      shading: resolveApiUrl(`/outputs/${jobId}/debug/relighted_debug/new_total.png`),
    });
    setProgress(100);
    setJob((current) => ({
      ...current,
      id: jobId || current.id,
      status: "completed",
      stage: "completed",
      queueState: "Completed",
      error: "",
      technical: "",
    }));
    addLog("Relight completed");
  };

  const pollJob = (jobId, controller) => {
    const poll = async () => {
      try {
        const data = await getRelightJob(jobId, controller.signal);
        const stage = data.stage || data.status;
        const stageLabel = STAGE_LABELS[stage] || "Processing portrait";

        setProgress(Number(data.progress || 0));
        setJob((current) => ({
          ...current,
          id: jobId,
          stage,
          queueState: stageLabel,
        }));

        if (stage !== lastStageRef.current) {
          lastStageRef.current = stage;
          addLog(stageLabel);
        }

        if (data.status === "finished") {
          const resultPayload = await getRelightResult(jobId, controller.signal);
          finishFromPayload(resultPayload, jobId);
          return;
        }

        if (["failed", "stopped", "canceled", "cancelled"].includes(data.status)) {
          throw new Error(
            data.error?.message || "The worker failed while processing this job.",
          );
        }

        pollRef.current = setTimeout(poll, 1500);
      } catch (error) {
        if (controller.signal.aborted) return;
        clearTimeout(pollRef.current);
        setJob((current) => ({
          ...current,
          status: "failed",
          queueState: "Failed",
          error: inferFailureKind(error.message),
          technical: error.message,
        }));
        addLog("Job failed");
      }
    };

    pollRef.current = setTimeout(poll, 500);
  };

  const generateRelight = async () => {
    if (!canGenerate) return;
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setResult(null);
    setDebugImages({});
    setProgress(0);
    lastStageRef.current = "queued";
    setJob({
      status: "processing",
      id: "",
      stage: "queued",
      queueState: "Uploading portrait",
      elapsed: 0,
      error: "",
      technical: "",
      logs: [{ time: new Date().toLocaleTimeString(), message: "Job received" }],
    });

    let timeout;
    try {
      timeout = setTimeout(() => controller.abort("timeout"), 120000);
      const payload = await createRelightJob(
        uploadFile || file,
        {
          newLight,
          personStrength: settings.personStrength,
          ambient: settings.ambient,
          highlight: settings.highlight,
          boundaryRelight: settings.boundaryRelight,
          shadowRelight: settings.shadowRelight,
          saveDebug: settings.showDebug,
          preset: stylePreset.id,
        },
        controller.signal,
      );
      clearTimeout(timeout);

      const jobId = payload.job_id;
      setJob((current) => ({
        ...current,
        id: jobId,
        stage: payload.stage,
        queueState: STAGE_LABELS[payload.stage] || "Queued",
      }));
      addLog("Upload complete");
      pollJob(jobId, controller);
    } catch (error) {
      clearTimeout(timeout);
      const cancelled = controller.signal.aborted;
      setJob((current) => ({
        ...current,
        status: cancelled ? "cancelled" : "failed",
        queueState: cancelled ? "Cancelled" : "Failed",
        error: cancelled ? "Request cancelled" : inferFailureKind(error.message),
        technical: cancelled ? "The frontend request was cancelled." : error.message,
      }));
      addLog(cancelled ? "Request cancelled" : "Request failed");
    }
  };

  const cancelJob = () => {
    abortRef.current?.abort();
    clearTimeout(pollRef.current);
    setJob((current) => ({
      ...current,
      status: "cancelled",
      queueState: "Cancelled",
      error: "Request cancelled",
      technical: "The frontend stopped waiting for this job. The backend worker may still complete it.",
    }));
  };

  const resetAll = () => {
    abortRef.current?.abort();
    clearTimeout(pollRef.current);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setFile(null);
    setUploadFile(null);
    setPreviewUrl("");
    setFileInfo(null);
    setValidation(null);
    setResult(null);
    setDebugImages({});
    setProgress(0);
    lastStageRef.current = "";
    if (fileInputRef.current) fileInputRef.current.value = "";
    setJob({ status: "idle", id: "", stage: "queued", queueState: "Ready", elapsed: 0, error: "", technical: "", logs: [] });
  };

  const downloadLogs = () => {
    const logs = {
      job,
      fileInfo,
      selectedPreset: selectedPreset.name,
      settings: { ...settings, newLight },
      validation,
      result,
      apiBase: apiBaseUrl,
      createdAt: new Date().toISOString(),
    };
    const blob = new Blob([JSON.stringify(logs, null, 2)], { type: "application/json" });
    const href = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = href;
    anchor.download = `sunit-debug-${job.id || "local"}.json`;
    anchor.click();
    URL.revokeObjectURL(href);
  };

  const activeStep = Math.max(0, STAGE_ORDER.indexOf(job.stage));

  return (
    <main className="app-shell">
      <div className="sun-beam" />
      <nav className="top-nav">
        <div className="brand">
          <span className="brand-mark" />
          <div>
            <strong>Sunit</strong>
            <span>AI Relighting Studio</span>
          </div>
        </div>
        <div className="nav-links">
          <a href="#demo">Demo</a>
          <a href="#pipeline">Pipeline</a>
          <a href="#debug">Debug</a>
          <a href="#about">About</a>
        </div>
        <div className={`status-pill ${job.status}`}>
          <span />
          {statusLabel}
        </div>
      </nav>

      <section className="hero" id="about">
        <div>
          <p className="eyebrow">Single-image portrait relighting</p>
          <h1>Relight portraits from a single image</h1>
          <p>
            Sunit estimates geometry, lighting, and reflectance, then applies region-aware
            relighting while preserving the identity and background of your portrait.
          </p>
        </div>
        <div className="hero-metrics">
          <span>Model <strong>Sunit v1.0</strong></span>
          <span>Quality <strong>High</strong></span>
          <span>Output <strong>PNG</strong></span>
        </div>
      </section>

      <section className="studio-grid" id="demo">
        <aside className="panel controls-panel">
          <SectionTitle number="1" title="Upload Image" />
          <div
            className="drop-zone"
            onDragOver={(event) => event.preventDefault()}
            onDrop={onDrop}
            onClick={() => fileInputRef.current?.click()}
            role="button"
            tabIndex={0}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="image/png,image/jpeg,image/webp"
              onChange={(event) => validateAndSetFile(event.target.files?.[0])}
            />
            <div className="upload-icon">+</div>
            <strong>Drag & drop or click to upload</strong>
            <span>JPG, PNG, or WEBP · max {formatBytes(MAX_SIZE_BYTES)}</span>
          </div>

          {fileInfo && (
            <div className={`file-card ${validation ? "invalid" : ""}`}>
              {previewUrl ? <img src={previewUrl} alt="Uploaded preview" /> : <div className="file-placeholder">IMG</div>}
              <div>
                <strong>{fileInfo.name}</strong>
                <span>{formatBytes(fileInfo.size)} · {fileInfo.width ? `${fileInfo.width} x ${fileInfo.height}` : fileInfo.type}</span>
                <em>{validation ? validation.title : fileInfo.resized ? "Prepared and resized" : "Upload ready"}</em>
              </div>
              <button type="button" onClick={resetAll}>Remove</button>
            </div>
          )}

          {validation && (
            <ErrorNotice
              title={validation.title}
              detail={validation.detail}
              primaryLabel={validation.action === "resize" ? "Enable auto-resize" : "Choose different file"}
              onPrimary={() => {
                if (validation.action === "resize") updateSetting("autoResize", true);
                fileInputRef.current?.click();
              }}
            />
          )}

          <div className="requirements">
            <div><span>Format</span><strong>JPG, PNG, WEBP</strong></div>
            <div><span>Max side</span><strong>{MAX_SIDE}px</strong></div>
            <div><span>Max file size</span><strong>15 MB</strong></div>
          </div>

          <SectionTitle number="2" title="Direction Preset" extra={selectedPreset.name} />
          <div className="preset-grid">
            {presets.map((preset) => (
              <button
                key={preset.id}
                type="button"
                className={selectedPreset.id === preset.id ? "active" : ""}
                onClick={() => setPreset(preset)}
              >
                <span style={{ "--x": `${preset.point[0]}%`, "--y": `${preset.point[1]}%` }} />
                {preset.name}
              </button>
            ))}
          </div>

          <SectionTitle number="3" title="Light Direction" />
          <div className="light-pad-wrap">
            <button className="light-pad" type="button" onClick={handlePadClick} aria-label="Choose custom light direction">
              <span className="axis horizontal" />
              <span className="axis vertical" />
              <span className="light-dot" style={{ left: `${padPoint[0]}%`, top: `${padPoint[1]}%` }} />
            </button>
            <div className="vector-readout">
              <span>x {newLight[0].toFixed(2)}</span>
              <span>y {newLight[1].toFixed(2)}</span>
              <span>z {newLight[2].toFixed(2)}</span>
            </div>
          </div>

          <SectionTitle number="4" title="Relight Style" extra={stylePreset.name} />
          <div className="style-grid">
            {stylePresets.map((preset) => (
              <button
                key={preset.id}
                type="button"
                className={stylePreset.id === preset.id ? "active" : ""}
                onClick={() => setStylePreset(preset)}
              >
                {preset.name}
              </button>
            ))}
          </div>

          <SectionTitle number="5" title="Relight Controls" />
          <Slider label="Person Strength" value={settings.personStrength} onChange={(value) => updateSetting("personStrength", value)} />
          <Slider label="Ambient Light" value={settings.ambient} onChange={(value) => updateSetting("ambient", value)} min={0.05} max={0.8} />
          <Slider label="Highlight Boost" value={settings.highlight} onChange={(value) => updateSetting("highlight", value)} max={0.3} />

          <SectionTitle number="6" title="Advanced" />
          <Toggle label="Advanced Mode" checked={settings.advanced} onChange={(value) => updateSetting("advanced", value)} />
          <Toggle label="Auto-resize large images" checked={settings.autoResize} onChange={(value) => updateSetting("autoResize", value)} />
          <Toggle label="Remove EXIF data" checked={settings.removeExif} onChange={(value) => updateSetting("removeExif", value)} />
          <Toggle label="Show debug outputs" checked={settings.showDebug} onChange={(value) => updateSetting("showDebug", value)} />

          {settings.advanced && (
            <div className="advanced-controls">
              <Slider label="Boundary Relight" value={settings.boundaryRelight} onChange={(value) => updateSetting("boundaryRelight", value)} max={0.7} />
              <Slider label="Shadow Relight" value={settings.shadowRelight} onChange={(value) => updateSetting("shadowRelight", value)} />
            </div>
          )}
        </aside>

        <section className="panel preview-panel">
          <SectionTitle number="7" title="Preview" extra={result ? "Drag to compare" : "Canvas"} />
          <div className={`preview-canvas ${!previewUrl ? "empty" : ""}`}>
            {!previewUrl && (
              <div className="empty-state">
                <span className="orbital" />
                <h2>Your relighting canvas is ready</h2>
                <p>Upload a clear portrait to generate warm, controllable relighting.</p>
              </div>
            )}
            {previewUrl && !result && (
              <div className="single-preview">
                <img src={previewUrl} alt="Original portrait" />
                <div className="faux-light" />
              </div>
            )}
            {previewUrl && result && (
              <div className="comparison" style={{ "--compare": compare }}>
                <img src={previewUrl} alt="Original portrait" />
                <div className="after" style={{ clipPath: `inset(0 0 0 ${compare}%)` }}>
                  <img src={result} alt="Relit portrait" />
                </div>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={compare}
                  onChange={(event) => setCompare(Number(event.target.value))}
                  aria-label="Before after comparison"
                />
                <span className="compare-label original">Original</span>
                <span className="compare-label relit">Relit</span>
              </div>
            )}

            {job.status === "processing" && (
              <div className="processing-overlay">
                <div className="processing-card">
                  <div className="processing-badge">AI</div>
                  <h2>Processing Relight</h2>
                  <p>{job.queueState}</p>
                  <div className="progress-track" aria-label={`Relighting progress ${progress}%`}>
                    <span style={{ width: `${progress}%` }} />
                  </div>
                  <strong className="progress-value">{progress}%</strong>
                  <div className="stepper">
                    {steps.map((step, index) => (
                      <div key={step} className={index < activeStep ? "done" : index === activeStep ? "current" : ""}>
                        <span>{index < activeStep ? "✓" : index + 1}</span>
                        <strong>{step}</strong>
                        <em>{index < activeStep ? "Completed" : index === activeStep ? "In progress" : "Pending"}</em>
                      </div>
                    ))}
                  </div>
                  <button type="button" className="ghost full" onClick={cancelJob}>Cancel Job</button>
                </div>
              </div>
            )}
          </div>

          <div className="actions">
            <button className="primary" type="button" disabled={!canGenerate} onClick={generateRelight}>Generate Relight</button>
            <button className="secondary" type="button" onClick={resetAll}>Reset</button>
            <a className={`download ${!result ? "disabled" : ""}`} href={result || "#"} download={`sunit-${job.id || "result"}.png`}>Download Result</a>
          </div>
          <p className="privacy-line">Images are sent only to your configured Sunit backend: {apiBaseUrl}</p>
        </section>

        <aside className="panel info-panel">
          <div className="job-card">
            <div className="job-head">
              <h3>Job Status</h3>
              <span className={`status-pill compact ${job.status}`}><span />{statusLabel}</span>
            </div>
            <InfoRow label="Job ID" value={job.id || "Not started"} />
            <InfoRow label="State" value={job.queueState} />
            <InfoRow label="Elapsed" value={elapsedLabel} />
            <InfoRow label="Direction" value={selectedPreset.name} />
            <InfoRow label="Style" value={stylePreset.name} />
            <button type="button" className="secondary full" disabled={!file || job.status === "processing"} onClick={generateRelight}>Retry</button>
          </div>

          {job.error && (
            <ErrorNotice
              title={job.error}
              detail={job.technical}
              primaryLabel="Retry"
              onPrimary={generateRelight}
            />
          )}

          <section id="pipeline" className="pipeline">
            <h3>Pipeline Overview</h3>
            {pipeline.map(([title, detail], index) => (
              <div className="pipeline-card" key={title}>
                <span>{index + 1}</span>
                <div><strong>{title}</strong><p>{detail}</p></div>
              </div>
            ))}
          </section>

          <section className="activity-log">
            <h3>Activity Log</h3>
            {job.logs.length === 0 ? <p>No activity yet.</p> : job.logs.map((entry, index) => (
              <div key={`${entry.time}-${index}`}><span>{entry.time}</span><strong>{entry.message}</strong></div>
            ))}
          </section>

          <section id="debug" className="debug-section">
            <h3>Debug Outputs</h3>
            <div className="debug-grid">
              <DebugTile title="Normal Map" src={settings.showDebug ? debugImages.normal : ""} />
              <DebugTile title="Person Mask" src={settings.showDebug ? debugImages.mask : ""} />
              <DebugTile title="Shading Map" src={settings.showDebug ? debugImages.shading : ""} />
            </div>
          </section>

          <section className="recovery">
            <h3>Recovery Options</h3>
            <button type="button" onClick={generateRelight} disabled={!file || job.status === "processing"}>Retry last operation</button>
            <button type="button" onClick={resetAll}>Reset inputs</button>
            <button type="button" onClick={() => updateSetting("advanced", true)}>Open advanced mode</button>
            <button type="button" onClick={downloadLogs}>Download logs</button>
          </section>
        </aside>
      </section>
    </main>
  );
}

function SectionTitle({ number, title, extra }) {
  return (
    <div className="section-title">
      <span>{number}</span>
      <strong>{title}</strong>
      {extra && <em>{extra}</em>}
    </div>
  );
}

function Slider({ label, value, onChange, min = 0, max = 1 }) {
  return (
    <label className="slider-row">
      <span>{label}</span>
      <input type="range" min={min} max={max} step="0.01" value={value} onChange={(event) => onChange(Number(event.target.value))} />
      <output>{value.toFixed(2)}</output>
    </label>
  );
}

function Toggle({ label, checked, onChange }) {
  return (
    <label className="toggle-row">
      <span>{label}</span>
      <input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} />
      <em />
    </label>
  );
}

function InfoRow({ label, value }) {
  return (
    <div className="info-row">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function ErrorNotice({ title, detail, primaryLabel, onPrimary }) {
  return (
    <div className="error-notice">
      <strong>{title}</strong>
      {detail && <p>{detail}</p>}
      <button type="button" onClick={onPrimary}>{primaryLabel}</button>
    </div>
  );
}

function DebugTile({ title, src }) {
  return (
    <div className="debug-tile">
      {src ? <img src={src} alt={title} onError={(event) => { event.currentTarget.style.display = "none"; }} /> : <span />}
      <strong>{title}</strong>
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);
