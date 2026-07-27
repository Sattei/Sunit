export const apiBaseUrl = (
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000"
).replace(/\/$/, "");

export class SunitApiError extends Error {
  constructor(message, status = 0, payload = null) {
    super(message);
    this.name = "SunitApiError";
    this.status = status;
    this.payload = payload;
  }
}

export function resolveApiUrl(path) {
  if (!path) return "";
  return new URL(path, `${apiBaseUrl}/`).toString();
}

async function readJson(response) {
  try {
    return await response.json();
  } catch {
    return {};
  }
}

function errorMessage(payload, fallback) {
  if (typeof payload?.detail === "string") return payload.detail;
  if (typeof payload?.error === "string") return payload.error;
  if (typeof payload?.error?.message === "string") return payload.error.message;
  return fallback;
}

async function requestJson(path, options = {}) {
  let response;

  try {
    response = await fetch(resolveApiUrl(path), options);
  } catch (error) {
    if (error.name === "AbortError") throw error;
    throw new SunitApiError(
      "The Sunit API is unavailable. Check that the local services are running.",
    );
  }

  const payload = await readJson(response);

  if (!response.ok) {
    throw new SunitApiError(
      errorMessage(payload, `The API returned ${response.status}.`),
      response.status,
      payload,
    );
  }

  return payload;
}

export async function createRelightJob(file, settings, signal) {
  const body = new FormData();
  body.append("image", file);
  body.append("new_x", settings.newLight[0].toFixed(4));
  body.append("new_y", settings.newLight[1].toFixed(4));
  body.append("new_z", settings.newLight[2].toFixed(4));
  body.append("person_strength", String(settings.personStrength));
  body.append("background_strength", "0");
  body.append("ambient", String(settings.ambient));
  body.append("highlight", String(settings.highlight));
  body.append("boundary_relight", String(settings.boundaryRelight));
  body.append("shadow_relight", String(settings.shadowRelight));
  body.append("save_debug", String(settings.saveDebug));
  body.append("preset", settings.preset);

  return requestJson("/api/relight-auto", {
    method: "POST",
    body,
    signal,
  });
}

export async function getRelightJob(jobId, signal) {
  return requestJson(`/api/jobs/${encodeURIComponent(jobId)}`, { signal });
}

export async function getRelightResult(jobId, signal) {
  return requestJson(
    `/api/jobs/${encodeURIComponent(jobId)}/result`,
    { signal },
  );
}
