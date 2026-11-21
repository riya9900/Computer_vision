"use client";

import dynamic from "next/dynamic";
import Link from "next/link";
import { useEffect, useState } from "react";
import {
  type Alert,
  API,
  getAlerts,
  getSites,
  type SiteSummary,
} from "@/lib/api";
import SitePanel from "./SitePanel";

// three.js touches window/WebGL, so the globe is client-only.
const EarthGlobe = dynamic(() => import("./EarthGlobe"), {
  ssr: false,
  loading: () => (
    <div className="absolute inset-0 grid place-items-center text-sm text-white/50">
      Loading globe…
    </div>
  ),
});

export default function OrbitApp() {
  const [sites, setSites] = useState<SiteSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<SiteSummary | null>(null);

  const [alerts, setAlerts] = useState<Alert[]>([]);

  useEffect(() => {
    getSites()
      .then(setSites)
      .catch((e: Error) => setError(e.message));
    // Alerts come from the weekly Temporal worker; none yet is a normal state.
    getAlerts()
      .then(setAlerts)
      .catch(() => setAlerts([]));
  }, []);

  const latestAlert = new Map<string, Alert>();
  for (const a of alerts)
    if (!latestAlert.has(a.siteId)) latestAlert.set(a.siteId, a);

  const totalHa = (sites ?? []).reduce((a, s) => a + s.changedHa, 0);

  return (
    <div className="flex h-dvh flex-col bg-[#05070d] md:flex-row">
      <section className="relative min-h-[50dvh] flex-1">
        <EarthGlobe
          sites={sites ?? []}
          selected={selected}
          onSelect={setSelected}
        />

        <div className="pointer-events-none absolute top-0 left-0 flex w-full flex-col gap-3 p-4 md:w-80">
          <div className="pointer-events-auto rounded-xl border border-white/10 bg-black/55 p-4 backdrop-blur">
            <div className="flex items-center justify-between">
              <h1 className="text-base font-semibold tracking-wide text-white">
                OrbitWatch
              </h1>
              <Link
                href="/results"
                className="rounded-full bg-orange-400/20 px-2.5 py-0.5 font-mono text-[11px] text-orange-300 hover:bg-orange-400/30"
              >
                Model results →
              </Link>
            </div>
            <p className="mt-1 text-xs text-white/60">
              Land change, watched from space. Pick a site to see what our model
              found and where look-alikes are.
            </p>

            {error && (
              <p className="mt-3 rounded-md bg-red-500/10 p-2 text-xs text-red-300">
                Can't reach the API at {API}. Start it with{" "}
                <code className="font-mono">
                  uv run uvicorn app.main:app --reload
                </code>{" "}
                in <code className="font-mono">server/</code>.
              </p>
            )}
            {!sites && !error && (
              <p className="mt-3 text-xs text-white/50">Loading sites…</p>
            )}
            {sites && sites.length === 0 && (
              <p className="mt-3 text-xs text-white/60">
                The API has no results yet. Put the Colab{" "}
                <code className="font-mono">demo</code> folder in{" "}
                <code className="font-mono">server/data/</code>.
              </p>
            )}

            <ul className="mt-3 flex flex-col gap-1.5">
              {(sites ?? []).map((s) => (
                <li key={s.id}>
                  <button
                    type="button"
                    onClick={() => setSelected(s)}
                    aria-pressed={selected?.id === s.id}
                    className={`w-full rounded-lg border px-3 py-2 text-left transition-colors ${
                      selected?.id === s.id
                        ? "border-orange-400/60 bg-orange-400/10"
                        : "border-white/10 hover:bg-white/5"
                    }`}
                  >
                    <span className="block text-sm text-white">{s.name}</span>
                    <span className="block text-[11px] text-orange-300">
                      {s.changedHa.toLocaleString()} ha changed ·{" "}
                      {s.region.split(",").pop()?.trim()}
                    </span>
                    {latestAlert.get(s.id) && (
                      <span
                        className="mt-1 block text-[11px] text-teal-300"
                        title={latestAlert.get(s.id)?.message}
                      >
                        ● {latestAlert.get(s.id)?.newScenes} new scenes
                        {latestAlert.get(s.id)?.latestDate
                          ? ` · latest ${latestAlert.get(s.id)?.latestDate}`
                          : ""}
                      </span>
                    )}
                  </button>
                </li>
              ))}
            </ul>
            {sites && sites.length > 0 && (
              <p className="mt-3 border-t border-white/10 pt-2 font-mono text-[11px] text-white/50">
                {sites.length} {sites.length === 1 ? "site" : "sites"} ·{" "}
                {totalHa.toLocaleString()} ha flagged
              </p>
            )}
          </div>
        </div>

        {selected && (
          <div className="pointer-events-none absolute bottom-3 left-4 flex items-center gap-2 rounded-lg bg-black/55 px-3 py-1.5 text-xs text-white/70 backdrop-blur">
            <span className="inline-block h-2 w-2 rounded-full bg-teal-400" />
            Look-alike places (Clay feature similarity)
          </div>
        )}
      </section>

      {selected && (
        <div className="max-h-[50dvh] w-full shrink-0 border-t border-white/10 bg-[#0b0f19] md:max-h-none md:w-[26rem] md:border-t-0 md:border-l">
          <SitePanel
            key={selected.id}
            site={selected}
            onClose={() => setSelected(null)}
          />
        </div>
      )}
    </div>
  );
}
