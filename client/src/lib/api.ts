// Everything the UI shows comes from the FastAPI server (real model output).
// If the server is down the UI says so; it never falls back to made-up data.

export const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type Lookalike = { lat: number; lng: number; score: number };
export type TimelinePoint = { year: number; changedHa: number };

export type SiteSummary = {
  id: string;
  name: string;
  region: string;
  lat: number;
  lng: number;
  halfDeg: number;
  fromClass: string | null;
  toClass: string | null;
  changedHa: number;
  confidence: number;
  started: number | null;
  lookalikes: Lookalike[];
};

export type ModelResult = {
  label: string;
  timeline: TimelinePoint[];
  masks: Record<string, string>;
  confidence: number;
  started: number | null;
};

export type SiteDetail = {
  id: string;
  name: string;
  region: string;
  lat: number;
  lng: number;
  halfDeg: number;
  bbox: [number, number, number, number]; // minX, minY, maxX, maxY
  years: number[];
  baseYear: number;
  primaryModel: string;
  fromClass: string | null;
  toClass: string | null;
  models: Record<string, ModelResult>;
  lookalikes: Lookalike[];
  source: string;
  canRetune: boolean;
};

type Stat = { mean: number; std: number };
export type ChangeMetrics = {
  seeds: number;
  params: number;
  train_seconds: number;
  f1: Stat;
  iou: Stat;
  precision: Stat;
  recall: Stat;
};
export type Metrics = {
  change: Record<string, ChangeMetrics>;
  landCover: { seeds: number; params: number; test_acc: Stat } | null;
};
export type OscdExample = {
  file: string;
  f1: number;
  iou: number;
  precision: number;
  recall: number;
};

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${API}${path}`);
  if (!res.ok) throw new Error(`${path}: ${res.status}`);
  return res.json() as Promise<T>;
}

export const getSites = () => get<SiteSummary[]>("/api/sites");
export const getSite = (id: string, thr?: number) =>
  get<SiteDetail>(`/api/sites/${id}${thr === undefined ? "" : `?thr=${thr}`}`);
export const getMetrics = () => get<Metrics>("/api/metrics");
export const getOscd = () => get<OscdExample[]>("/api/oscd");
export type Alert = {
  siteId: string;
  kind: string;
  newScenes: number;
  latestDate: string | null;
  message: string;
  createdAt: string;
};
export const getAlerts = () => get<Alert[]>("/api/alerts");

export const maskUrl = (
  id: string,
  model: string,
  year: number,
  thr?: number,
) =>
  `${API}/api/sites/${id}/mask/${model}/${year}.png${thr === undefined ? "" : `?thr=${thr}`}`;
export const oscdUrl = (file: string) => `${API}/api/oscd/${file}`;

/** Real Sentinel-2 cloudless yearly mosaic (EOX) for the same footprint, as a visual backdrop. */
export function imageUrl(bbox: number[], year: number, size = 640) {
  const q = new URLSearchParams({
    service: "WMS",
    version: "1.1.1",
    request: "GetMap",
    layers: `s2cloudless-${year}`,
    styles: "",
    format: "image/jpeg",
    srs: "EPSG:4326",
    bbox: bbox.join(","),
    width: String(size),
    height: String(size),
  });
  return `https://tiles.maps.eox.at/wms?${q.toString()}`;
}
