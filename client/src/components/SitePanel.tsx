"use client";

import { useState } from "react";
import { type Site, YEARS } from "@/lib/sites";
import BeforeAfter from "./BeforeAfter";
import Timeline from "./Timeline";

export default function SitePanel({
  site,
  onClose,
}: {
  site: Site;
  onClose: () => void;
}) {
  const before = YEARS[0];
  const [year, setYear] = useState<number>(YEARS[YEARS.length - 1]);
  const [report, setReport] = useState<string | null>(null);

  const changed = site.timeline.find((t) => t.year === year)?.changedHa ?? 0;
  const firstChange = site.timeline.find((t) => t.changedHa > 0)?.year;

  function writeReport() {
    setReport(
      `${site.name} (${site.region}): between ${before} and ${year}, about ${changed.toLocaleString()} ha ` +
        `changed from ${site.fromClass.toLowerCase()} to ${site.toClass.toLowerCase()}, ` +
        `first visible around ${firstChange}. Model confidence ${(site.confidence * 100).toFixed(0)}%. ` +
        `${site.lookalikes.length} similar sites found nearby. ` +
        "This reports that land changed; it does not say whether the change is legal. A person should verify.",
    );
  }

  return (
    <aside className="flex h-full flex-col gap-4 overflow-y-auto p-4 md:p-5 [&>*]:shrink-0">
      <header className="flex items-start justify-between gap-3">
        <div>
          <p className="font-mono text-[11px] tracking-widest text-orange-300 uppercase">
            Change detected · {site.alertAge}
          </p>
          <h2 className="text-lg leading-tight font-semibold text-white">
            {site.name}
          </h2>
          <p className="text-sm text-white/60">{site.region}</p>
        </div>
        <button
          type="button"
          onClick={onClose}
          aria-label="Close site"
          className="rounded-md border border-white/15 px-2 py-1 text-sm text-white/70 hover:bg-white/10"
        >
          ✕
        </button>
      </header>

      <BeforeAfter site={site} before={before} after={year} />

      <div>
        <h3 className="mb-2 text-xs font-medium tracking-wide text-white/50 uppercase">
          Changed area per year (tap a bar)
        </h3>
        <Timeline site={site} year={year} onYear={setYear} />
      </div>

      <dl className="grid grid-cols-3 gap-2 text-sm">
        <Stat label="Changed" value={`${changed.toLocaleString()} ha`} />
        <Stat label="Started" value={firstChange ? `~${firstChange}` : "—"} />
        <Stat
          label="Confidence"
          value={`${(site.confidence * 100).toFixed(0)}%`}
        />
      </dl>
      <p className="rounded-md bg-white/5 px-3 py-2 text-sm text-white/80">
        <span className="text-white/50">From → to: </span>
        {site.fromClass} →{" "}
        <span className="text-orange-300">{site.toClass}</span>
      </p>

      <button
        type="button"
        onClick={writeReport}
        className="rounded-md bg-orange-400 px-3 py-2 text-sm font-medium text-black hover:bg-orange-300"
      >
        Ask assistant to write a report
      </button>
      {report && (
        <p className="rounded-md border border-white/10 bg-white/5 p-3 text-sm leading-relaxed text-white/85">
          {report}
        </p>
      )}

      <p className="mt-auto text-[11px] leading-relaxed text-white/40">
        Imagery: Sentinel-2 cloudless yearly mosaics (EOX), real. Change
        numbers, look-alikes and the report are placeholders until the API is
        connected. Satellite photos can show large changes only.
      </p>
    </aside>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md bg-white/5 px-3 py-2">
      <dt className="text-[11px] text-white/50">{label}</dt>
      <dd className="font-mono text-white">{value}</dd>
    </div>
  );
}
