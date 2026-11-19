"use client";

import dynamic from "next/dynamic";
import { useState } from "react";
import { SITES, type Site } from "@/lib/sites";
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
  const [selected, setSelected] = useState<Site | null>(null);

  return (
    <div className="flex h-dvh flex-col bg-[#05070d] md:flex-row">
      <section className="relative min-h-[50dvh] flex-1">
        <EarthGlobe selected={selected} onSelect={setSelected} />

        <div className="pointer-events-none absolute top-0 left-0 flex w-full flex-col gap-3 p-4 md:w-80">
          <div className="pointer-events-auto rounded-xl border border-white/10 bg-black/55 p-4 backdrop-blur">
            <div className="flex items-center justify-between">
              <h1 className="text-base font-semibold tracking-wide text-white">
                OrbitWatch
              </h1>
              <span className="rounded-full bg-orange-400/20 px-2 py-0.5 font-mono text-[10px] text-orange-300">
                DEMO
              </span>
            </div>
            <p className="mt-1 text-xs text-white/60">
              Land change, watched from space. Pick a site to see what changed
              and where look-alikes are.
            </p>
            <ul className="mt-3 flex flex-col gap-1.5">
              {SITES.map((s) => (
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
                      Alert · {s.alertAge}
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          </div>
        </div>

        {selected && (
          <div className="pointer-events-none absolute bottom-3 left-4 flex items-center gap-2 rounded-lg bg-black/55 px-3 py-1.5 text-xs text-white/70 backdrop-blur">
            <span className="inline-block h-2 w-2 rounded-full bg-teal-400" />
            Look-alike places (placeholder)
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
