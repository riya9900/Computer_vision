// Demo sites. Imagery is REAL (EOX Sentinel-2 cloudless yearly mosaics).
// Change statistics, look-alike points and alerts are PLACEHOLDERS until the API
// (server/) serves real model output.

export type Lookalike = { lat: number; lng: number; score: number };

export type Site = {
  id: string;
  name: string;
  region: string;
  lat: number;
  lng: number;
  halfDeg: number; // half-width of the analysed footprint, degrees
  fromClass: string;
  toClass: string;
  confidence: number;
  alertAge: string;
  /** changed hectares per year, cumulative since the first year */
  timeline: { year: number; changedHa: number }[];
  lookalikes: Lookalike[];
};

export const YEARS = [2018, 2019, 2020, 2021, 2022, 2023] as const;

export const SITES: Site[] = [
  {
    id: "nusantara",
    name: "Nusantara capital zone",
    region: "East Kalimantan, Indonesia",
    lat: -0.95,
    lng: 116.7,
    halfDeg: 0.08,
    fromClass: "Forest",
    toClass: "Bare ground / roads",
    confidence: 0.82,
    alertAge: "3 days ago",
    timeline: [
      { year: 2018, changedHa: 0 },
      { year: 2019, changedHa: 140 },
      { year: 2020, changedHa: 260 },
      { year: 2021, changedHa: 610 },
      { year: 2022, changedHa: 1480 },
      { year: 2023, changedHa: 2350 },
    ],
    lookalikes: [
      { lat: -0.62, lng: 116.95, score: 0.91 },
      { lat: -1.21, lng: 116.52, score: 0.88 },
      { lat: -0.88, lng: 117.18, score: 0.84 },
      { lat: -1.4, lng: 116.9, score: 0.8 },
    ],
  },
  {
    id: "atacama",
    name: "Salar de Atacama lithium ponds",
    region: "Antofagasta, Chile",
    lat: -23.7,
    lng: -68.27,
    halfDeg: 0.07,
    fromClass: "Salt flat",
    toClass: "Industrial (brine ponds)",
    confidence: 0.77,
    alertAge: "9 days ago",
    timeline: [
      { year: 2018, changedHa: 0 },
      { year: 2019, changedHa: 30 },
      { year: 2020, changedHa: 75 },
      { year: 2021, changedHa: 120 },
      { year: 2022, changedHa: 210 },
      { year: 2023, changedHa: 330 },
    ],
    lookalikes: [
      { lat: -23.52, lng: -68.4, score: 0.86 },
      { lat: -23.95, lng: -68.15, score: 0.83 },
      { lat: -24.1, lng: -68.35, score: 0.79 },
    ],
  },
];

export function footprint(site: Site) {
  const { lat, lng, halfDeg: h } = site;
  return { minX: lng - h, minY: lat - h, maxX: lng + h, maxY: lat + h };
}

/** Real Sentinel-2 cloudless mosaic for a footprint and year (free, public WMS). */
export function imageUrl(site: Site, year: number, size = 640) {
  const { minX, minY, maxX, maxY } = footprint(site);
  const q = new URLSearchParams({
    service: "WMS",
    version: "1.1.1",
    request: "GetMap",
    layers: `s2cloudless-${year}`,
    styles: "",
    format: "image/jpeg",
    srs: "EPSG:4326",
    bbox: `${minX},${minY},${maxX},${maxY}`,
    width: String(size),
    height: String(size),
  });
  return `https://tiles.maps.eox.at/wms?${q.toString()}`;
}
