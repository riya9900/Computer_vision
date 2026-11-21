"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import {
  API,
  getMetrics,
  getOscd,
  type Metrics,
  type OscdExample,
  oscdUrl,
} from "@/lib/api";

const NAMES: Record<string, { label: string; ours: boolean; note: string }> = {
  siamese_unet: {
    label: "Siamese U-Net",
    ours: true,
    note: "our model, trained from scratch",
  },
  clay_conv: {
    label: "Frozen Clay + conv head",
    ours: true,
    note: "foundation model, our trained head",
  },
  clay_linear: {
    label: "Frozen Clay + linear probe",
    ours: false,
    note: "baseline",
  },
};

const pm = (s: { mean: number; std: number }, d = 3) =>
  `${s.mean.toFixed(d)} ± ${s.std.toFixed(d)}`;

export default function Results() {
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [gallery, setGallery] = useState<OscdExample[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getMetrics()
      .then(setMetrics)
      .catch((e: Error) => setError(e.message));
    getOscd()
      .then(setGallery)
      .catch(() => setGallery([]));
  }, []);

  const rows = metrics
    ? Object.entries(metrics.change).sort((a, b) => b[1].f1.mean - a[1].f1.mean)
    : [];

  return (
    <main className="mx-auto max-w-5xl px-4 py-8 text-white md:px-8">
      <Link href="/" className="text-sm text-orange-300 hover:underline">
        ← Back to the globe
      </Link>
      <h1 className="mt-3 text-2xl font-semibold">Model results</h1>
      <p className="mt-1 max-w-2xl text-sm text-white/60">
        What we trained, on which data, and how well it works on images the
        models never saw. Numbers are mean ± std over 3 random seeds, measured
        on the held-out OSCD test pairs (change detection) and a held-out
        EuroSAT split (land cover).
      </p>

      {error && (
        <p className="mt-6 rounded-md bg-red-500/10 p-3 text-sm text-red-300">
          Can't load metrics from {API} ({error}).
        </p>
      )}

      {metrics && (
        <>
          <h2 className="mt-8 text-lg font-medium">
            Change detection (OSCD test, 10 image pairs)
          </h2>
          <div className="mt-3 overflow-x-auto rounded-xl border border-white/10">
            <table className="w-full min-w-[640px] text-left text-sm">
              <thead className="bg-white/5 text-xs text-white/50 uppercase">
                <tr>
                  <th className="px-3 py-2">Model</th>
                  <th className="px-3 py-2">F1</th>
                  <th className="px-3 py-2">IoU</th>
                  <th className="px-3 py-2">Precision</th>
                  <th className="px-3 py-2">Recall</th>
                  <th className="px-3 py-2">Trainable params</th>
                  <th className="px-3 py-2">Train time per run (GPU)</th>
                </tr>
              </thead>
              <tbody>
                {rows.map(([key, m]) => {
                  const n = NAMES[key] ?? { label: key, ours: false, note: "" };
                  return (
                    <tr key={key} className="border-t border-white/10">
                      <td className="px-3 py-2">
                        <span className="font-medium">{n.label}</span>
                        <span className="block text-[11px] text-white/45">
                          {n.note}
                        </span>
                      </td>
                      <td className="px-3 py-2">
                        <div className="font-mono">{pm(m.f1)}</div>
                        <div className="mt-1 h-1.5 w-28 rounded bg-white/10">
                          <div
                            className={`h-1.5 rounded ${n.ours ? "bg-orange-400" : "bg-white/40"}`}
                            style={{ width: `${m.f1.mean * 100}%` }}
                          />
                        </div>
                      </td>
                      <td className="px-3 py-2 font-mono">{pm(m.iou)}</td>
                      <td className="px-3 py-2 font-mono">{pm(m.precision)}</td>
                      <td className="px-3 py-2 font-mono">{pm(m.recall)}</td>
                      <td className="px-3 py-2 font-mono">
                        {m.params.toLocaleString("en-US")}
                      </td>
                      <td className="px-3 py-2 font-mono">
                        {m.train_seconds.toFixed(0)} s
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <p className="mt-3 max-w-3xl text-xs leading-relaxed text-white/55">
            <b className="text-white/70">Training time.</b> The column above is
            the measured time of each final training run on a Colab GPU (the
            Siamese U-Net: about 2 minutes per seed, 3 seeds). Total Colab time
            spent by the team on training, including a first slower attempt, the
            three seeds of each model and caching the Clay features, was about{" "}
            <b className="text-white/70">40 minutes</b> (team-reported).
          </p>

          {metrics.landCover && (
            <div className="mt-6 rounded-xl border border-white/10 bg-white/5 p-4">
              <h2 className="text-lg font-medium">
                Land cover (EuroSAT, 10 classes)
              </h2>
              <p className="mt-1 text-sm text-white/70">
                Test accuracy{" "}
                <span className="font-mono text-orange-300">
                  {(metrics.landCover.test_acc.mean * 100).toFixed(1)}% ±{" "}
                  {(metrics.landCover.test_acc.std * 100).toFixed(1)}
                </span>{" "}
                with a {metrics.landCover.params.toLocaleString("en-US")}
                -parameter head on frozen Clay features. This is what says
                "forest → industrial" in the site reports.
              </p>
            </div>
          )}
        </>
      )}

      {gallery.length > 0 && (
        <>
          <h2 className="mt-10 text-lg font-medium">
            Our model against human labels
          </h2>
          <p className="mt-1 max-w-2xl text-sm text-white/60">
            Held-out OSCD test scenes. Left to right: before, after, the change
            people marked by hand, and what our Siamese U-Net predicted.
          </p>
          <div className="mt-4 grid gap-4">
            {gallery.map((g) => (
              <figure
                key={g.file}
                className="overflow-hidden rounded-xl border border-white/10 bg-white"
              >
                {/* biome-ignore lint/performance/noImgElement: served by our API */}
                <img
                  src={oscdUrl(g.file)}
                  alt={`OSCD test scene ${g.file}: before, after, ground truth, prediction`}
                  className="w-full"
                  loading="lazy"
                />
                <figcaption className="bg-[#0b0f19] px-3 py-1.5 font-mono text-xs text-white/70">
                  F1 {g.f1.toFixed(2)} · IoU {g.iou.toFixed(2)} · precision{" "}
                  {g.precision.toFixed(2)} · recall {g.recall.toFixed(2)}
                </figcaption>
              </figure>
            ))}
          </div>
        </>
      )}

      <h2 className="mt-10 text-lg font-medium">Data, training and limits</h2>
      <ul className="mt-2 list-disc space-y-1.5 pl-5 text-sm text-white/70">
        <li>
          <b>Datasets:</b> OSCD (24 Sentinel-2 pairs, 13 bands; 14 train / 10
          test) for change; EuroSAT (27,000 labelled patches) for land cover.
        </li>
        <li>
          <b>Siamese U-Net:</b> shared-weight encoder on both dates, absolute
          feature difference at each scale, Dice + weighted BCE loss, random
          crops with flips, rotations and before/after swaps.
        </li>
        <li>
          <b>Clay:</b> an open foundation model (frozen). It is run once and its
          features cached; only the small heads are trained.
        </li>
        <li>
          <b>Honest limits:</b> OSCD labels urban change only. Mines, lakes and
          reservoirs are shown as qualitative demos of the same models, not
          scored. Satellite photos are not sharp, so only large changes are
          visible. Season and haze between years cause false alarms. We report
          that land changed, not whether it is legal.
        </li>
      </ul>
    </main>
  );
}
