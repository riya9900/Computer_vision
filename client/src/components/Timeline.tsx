"use client";

import type { TimelinePoint } from "@/lib/api";

type Props = {
  timeline: TimelinePoint[];
  year: number;
  onYear: (y: number) => void;
};

export default function Timeline({ timeline, year, onYear }: Props) {
  const max = Math.max(...timeline.map((t) => t.changedHa), 1);
  return (
    <div>
      <div className="flex h-24 items-end gap-1.5">
        {timeline.map((t) => {
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
              <span className="mb-1 text-center font-mono text-[10px] text-white/60">
                {t.changedHa.toLocaleString()}
              </span>
              <div
                className={`w-full rounded-t transition-all duration-500 ${
                  active
                    ? "bg-orange-400"
                    : "bg-orange-400/30 group-hover:bg-orange-400/60"
                }`}
                style={{ height: `${Math.max((t.changedHa / max) * 70, 3)}%` }}
              />
            </button>
          );
        })}
      </div>
      <div className="mt-1 flex gap-1.5 font-mono text-[11px] text-white/60">
        {timeline.map((t) => (
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
