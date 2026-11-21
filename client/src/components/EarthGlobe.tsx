"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import Globe, { type GlobeMethods } from "react-globe.gl";
import type { SiteSummary } from "@/lib/api";

const TEX = "https://unpkg.com/three-globe@2.41.12/example/img";

type Props = {
  sites: SiteSummary[];
  selected: SiteSummary | null;
  onSelect: (site: SiteSummary | null) => void;
};

export default function EarthGlobe({ sites, selected, onSelect }: Props) {
  const [webgl, setWebgl] = useState<boolean | null>(null);
  const globe = useRef<GlobeMethods | undefined>(undefined);
  const box = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ w: 800, h: 600 });

  useEffect(() => {
    const el = box.current;
    if (!el) return;
    const ro = new ResizeObserver(() =>
      setSize({ w: el.clientWidth, h: el.clientHeight }),
    );
    ro.observe(el);
    setSize({ w: el.clientWidth, h: el.clientHeight });
    return () => ro.disconnect();
  }, []);

  // Slow idle spin until a site is chosen.
  useEffect(() => {
    const controls = globe.current?.controls();
    if (!controls) return;
    controls.autoRotate = !selected;
    controls.autoRotateSpeed = 0.45;
  }, [selected]);

  // Fly to the selected site (or back out).
  useEffect(() => {
    const g = globe.current;
    if (!g) return;
    if (selected) {
      g.pointOfView(
        { lat: selected.lat, lng: selected.lng, altitude: 0.75 },
        1600,
      );
    } else {
      g.pointOfView({ altitude: 2.4 }, 1200);
    }
  }, [selected]);

  // Some browsers/GPUs cannot create a WebGL context; say so instead of crashing the page.
  useEffect(() => {
    try {
      const c = document.createElement("canvas");
      setWebgl(!!(c.getContext("webgl2") || c.getContext("webgl")));
    } catch {
      setWebgl(false);
    }
  }, []);

  const lookalikes = useMemo(() => selected?.lookalikes ?? [], [selected]);

  const polygons = useMemo(() => {
    if (!selected) return [];
    const h = selected.halfDeg;
    const { lat, lng } = selected;
    return [
      {
        type: "Polygon" as const,
        coordinates: [
          [
            [lng - h, lat - h],
            [lng + h, lat - h],
            [lng + h, lat + h],
            [lng - h, lat + h],
            [lng - h, lat - h],
          ],
        ],
      },
    ];
  }, [selected]);

  if (webgl === false) {
    return (
      <div className="absolute inset-0 grid place-items-center p-6">
        <div className="max-w-sm rounded-xl border border-white/10 bg-black/60 p-4 text-sm text-white/80">
          <p className="font-medium text-white">The 3D globe needs WebGL</p>
          <p className="mt-1 text-white/60">
            Your browser could not start it. Turn on hardware acceleration in
            the browser settings (System), restart the browser, or try another
            one. You can still pick a site from the list and use everything
            else.
          </p>
        </div>
      </div>
    );
  }

  if (webgl === null) return <div ref={box} className="absolute inset-0" />;

  return (
    <div ref={box} className="absolute inset-0">
      <Globe
        ref={globe}
        width={size.w}
        height={size.h}
        globeImageUrl={`${TEX}/earth-blue-marble.jpg`}
        bumpImageUrl={`${TEX}/earth-topology.png`}
        backgroundImageUrl={`${TEX}/night-sky.png`}
        atmosphereColor="#7dd3fc"
        atmosphereAltitude={0.18}
        onGlobeReady={() => {
          globe.current?.pointOfView({ lat: 5, lng: 60, altitude: 2.4 }, 0);
        }}
        onGlobeClick={() => onSelect(null)}
        // watched sites: pulsing rings + labelled pins
        ringsData={sites}
        ringLat="lat"
        ringLng="lng"
        ringColor={() => (t: number) => `rgba(251,146,60,${1 - t})`}
        ringMaxRadius={selected ? 0.6 : 3.2}
        ringPropagationSpeed={2}
        ringRepeatPeriod={1400}
        labelsData={sites}
        labelLat="lat"
        labelLng="lng"
        labelText="name"
        labelSize={selected ? 0.18 : 0.9}
        labelDotRadius={selected ? 0.12 : 0.5}
        labelColor={() => "#fed7aa"}
        labelAltitude={0.01}
        onLabelClick={(d) => onSelect(d as SiteSummary)}
        // analysed footprint
        polygonsData={polygons}
        polygonGeoJsonGeometry={(d) => d as never}
        polygonCapColor={() => "rgba(251,146,60,0.18)"}
        polygonSideColor={() => "rgba(251,146,60,0.0)"}
        polygonStrokeColor={() => "#fb923c"}
        polygonAltitude={0.004}
        // look-alike places found by Clay feature similarity
        pointsData={lookalikes}
        pointLat="lat"
        pointLng="lng"
        pointColor={() => "#2dd4bf"}
        pointAltitude={(d) => 0.02 + (d as { score: number }).score * 0.05}
        pointRadius={0.1}
        pointLabel={(d) =>
          `Look-alike · similarity ${((d as { score: number }).score * 100).toFixed(0)}%`
        }
      />
    </div>
  );
}
