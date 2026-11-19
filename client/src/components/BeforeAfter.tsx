"use client";

import { useState } from "react";
import { imageUrl, type Site } from "@/lib/sites";

type Props = { site: Site; before: number; after: number };

export default function BeforeAfter({ site, before, after }: Props) {
  const [pos, setPos] = useState(50);

  return (
    <div className="relative aspect-square w-full select-none overflow-hidden rounded-lg border border-white/10 bg-black">
      {/* biome-ignore lint/performance/noImgElement: remote WMS imagery, not optimisable by next/image */}
      <img
        src={imageUrl(site, after)}
        alt={`${site.name} in ${after}`}
        className="absolute inset-0 h-full w-full object-cover"
        draggable={false}
      />
      {/* biome-ignore lint/performance/noImgElement: remote WMS imagery, not optimisable by next/image */}
      <img
        src={imageUrl(site, before)}
        alt={`${site.name} in ${before}`}
        className="absolute inset-0 h-full w-full object-cover"
        style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }}
        draggable={false}
      />
      <div
        className="pointer-events-none absolute inset-y-0 w-0.5 bg-white shadow-[0_0_8px_rgba(0,0,0,0.8)]"
        style={{ left: `${pos}%` }}
      >
        <div className="absolute top-1/2 left-1/2 flex h-8 w-8 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full bg-white text-xs text-black shadow">
          ⇆
        </div>
      </div>
      <span className="pointer-events-none absolute top-2 left-2 rounded bg-black/60 px-2 py-0.5 font-mono text-xs text-white">
        {before}
      </span>
      <span className="pointer-events-none absolute top-2 right-2 rounded bg-black/60 px-2 py-0.5 font-mono text-xs text-white">
        {after}
      </span>
      <input
        type="range"
        min={0}
        max={100}
        value={pos}
        onChange={(e) => setPos(Number(e.target.value))}
        aria-label="Before and after slider"
        className="absolute inset-0 h-full w-full cursor-ew-resize opacity-0"
      />
    </div>
  );
}
