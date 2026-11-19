"use client";

import type { Site } from "@/lib/sites";

type Props = { site: Site; year: number; onYear: (y: number) => void };

export default function Timeline({ site, year, onYear }: Props) {
  const max = Math.max(...site.timeline.map((t) => t.changedHa), 1);
  return (
    <div>
      <div className="flex h-24 items-end gap-1.5">
        {site.timeline.map((t) => {
          const active = t.year === year;
          return (
            <button
              key={t.year}
              type="button"
              onClick={() => onYear(t.year)}
              aria-label={`Show ${t.year}: ${t.changedHa} hectares changed`}
              aria-pressed={active}
              className="group flex h-full flex-1 flex-col justify-end"
            >
              <div
                className={`w-full rounded-t transition-colors ${
                  active
                    ? "bg-orange-400"
                    : "bg-orange-400/30 group-hover:bg-orange-400/60"
                }`}
                style={{ height: `${Math.max((t.changedHa / max) * 100, 3)}%` }}
              />
            </button>
          );
        })}
      </div>
      <div className="mt-1 flex gap-1.5 font-mono text-[11px] text-white/60">
        {site.timeline.map((t) => (
          <span
            key={t.year}
            className={`flex-1 text-center ${t.year === year ? "text-orange-300" : ""}`}
          >
            {t.year}
          </span>
        ))}
      </div>
    </div>
  );
}
