"use client";

import { useEffect, useState } from "react";
import { getSite, maskUrl, type SiteDetail, type SiteSummary } from "@/lib/api";
import BeforeAfter from "./BeforeAfter";
import Timeline from "./Timeline";

const MODEL_NOTE: Record<string, string> = {
  siamese_unet: "trained from scratch on OSCD",
  clay_conv: "frozen Clay + our head",
};

export default function SitePanel({
  site,
  onClose,
}: {
  site: SiteSummary;
  onClose: () => void;
}) {
  const [detail, setDetail] = useState<SiteDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [model, setModel] = useState<string | null>(null);
  const [year, setYear] = useState<number | null>(null);
  const [showMask, setShowMask] = useState(true);
  const [strict, setStrict] = useState<number | null>(null); // null = pipeline default
  const [committed, setCommitted] = useState<number | null>(null);
  const [report, setReport] = useState<string | null>(null);

  // Debounce the strictness slider so we don't call the API on every pixel of drag.
  useEffect(() => {
    const t = setTimeout(() => setCommitted(strict), 250);
    return () => clearTimeout(t);
  }, [strict]);

  useEffect(() => {
    let live = true;
    getSite(site.id, committed ?? undefined)
      .then((d) => {
        if (!live) return;
        setDetail(d);
        setModel((m) => m ?? d.primaryModel);
        setYear((y) => y ?? d.years[d.years.length - 1]);
      })
      .catch((e: Error) => live && setError(e.message));
    return () => {
      live = false;
    };
  }, [site.id, committed]);

  if (error) {
    return (
      <p className="p-5 text-sm text-red-300">
        Could not load this site from the API ({error}).
      </p>
    );
  }
  if (!detail || !model || year === null) {
    return <p className="p-5 text-sm text-white/50">Loading real results…</p>;
  }

  const result = detail.models[model];
  const isPrimary = model === detail.primaryModel;
  const changed = result.timeline.find((t) => t.year === year)?.changedHa ?? 0;
  const thr =
    strict === null || !isPrimary ? undefined : (committed ?? undefined);
  const overlay =
    year === detail.baseYear ? undefined : maskUrl(detail.id, model, year, thr);

  function writeReport() {
    const d = detail as SiteDetail;
    const fromTo =
      d.fromClass && d.toClass
        ? `, mostly ${d.fromClass.toLowerCase()} turning into ${d.toClass.toLowerCase()}`
        : "";
    setReport(
      `${d.name} (${d.region}): between ${d.baseYear} and ${year}, the ${result.label} marked about ` +
        `${changed.toLocaleString()} ha as changed${fromTo}. Mean model confidence ` +
        `${(result.confidence * 100).toFixed(0)}%. ${d.lookalikes.length} look-alike places were found nearby. ` +
        "This reports that land changed; it does not say whether the change is legal. A person should verify.",
    );
  }

  return (
    <aside className="flex h-full flex-col gap-4 overflow-y-auto p-4 md:p-5 [&>*]:shrink-0">
      <header className="flex items-start justify-between gap-3">
        <div>
          <p className="font-mono text-[11px] tracking-widest text-orange-300 uppercase">
            Change detected · {detail.baseYear} → {year}
          </p>
          <h2 className="text-lg leading-tight font-semibold text-white">
            {detail.name}
          </h2>
          <p className="text-sm text-white/60">{detail.region}</p>
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

      <div className="flex gap-1 rounded-lg bg-white/5 p-1 text-xs">
        {Object.keys(detail.models).map((m) => (
          <button
            key={m}
            type="button"
            onClick={() => setModel(m)}
            aria-pressed={m === model}
            className={`flex-1 rounded-md px-2 py-1.5 text-left transition-colors ${
              m === model
                ? "bg-orange-400 text-black"
                : "text-white/70 hover:bg-white/10"
            }`}
          >
            <span className="block font-medium">
              {m === detail.primaryModel ? "Our Siamese U-Net" : "Clay + head"}
            </span>
            <span className="block opacity-70">{MODEL_NOTE[m]}</span>
          </button>
        ))}
      </div>

      <BeforeAfter
        bbox={detail.bbox}
        before={detail.baseYear}
        after={year}
        overlay={showMask ? overlay : undefined}
        name={detail.name}
      />

      <div className="flex items-center justify-between gap-3 text-sm">
        <label className="flex items-center gap-2 text-white/80">
          <input
            type="checkbox"
            checked={showMask}
            onChange={(e) => setShowMask(e.target.checked)}
            className="accent-orange-400"
          />
          Show model-detected change
        </label>
        <span className="flex items-center gap-1.5 text-[11px] text-white/50">
          <span className="inline-block h-2.5 w-2.5 rounded-sm bg-orange-400/80" />
          changed
        </span>
      </div>

      {isPrimary && (
        <div>
          <div className="mb-1 flex justify-between text-xs text-white/50">
            <span>Strictness (re-computed live from the model)</span>
            <span className="font-mono text-white/70">
              {(strict ?? 0.75).toFixed(2)}
            </span>
          </div>
          <input
            type="range"
            min={0.3}
            max={0.95}
            step={0.05}
            value={strict ?? 0.75}
            onChange={(e) => setStrict(Number(e.target.value))}
            aria-label="Detection strictness"
            className="w-full accent-orange-400"
          />
        </div>
      )}

      <div>
        <h3 className="mb-2 text-xs font-medium tracking-wide text-white/50 uppercase">
          Changed area (hectares)
        </h3>
        <Timeline timeline={result.timeline} year={year} onYear={setYear} />
      </div>

      <dl className="grid grid-cols-3 gap-2 text-sm">
        <Stat label="Changed" value={`${changed.toLocaleString()} ha`} />
        <Stat label="Look-alikes" value={String(detail.lookalikes.length)} />
        <Stat
          label="Confidence"
          value={`${(result.confidence * 100).toFixed(0)}%`}
        />
      </dl>
      {detail.fromClass && detail.toClass && (
        <p className="rounded-md bg-white/5 px-3 py-2 text-sm text-white/80">
          <span className="text-white/50">From → to: </span>
          {detail.fromClass} →{" "}
          <span className="text-orange-300">{detail.toClass}</span>
        </p>
      )}

      <button
        type="button"
        onClick={writeReport}
        className="rounded-md bg-orange-400 px-3 py-2 text-sm font-medium text-black hover:bg-orange-300"
      >
        Write a report from these results
      </button>
      {report && (
        <p className="rounded-md border border-white/10 bg-white/5 p-3 text-sm leading-relaxed text-white/85">
          {report}
        </p>
      )}

      <p className="mt-auto text-[11px] leading-relaxed text-white/40">
        Model output on real Sentinel-2 (L2A) yearly composites. Background
        photos: EOX Sentinel-2 cloudless. Season and haze differences between
        years can cause false alarms; strictness is a demo setting. Satellite
        photos show large changes only. Reports are templated from the numbers
        above.
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
