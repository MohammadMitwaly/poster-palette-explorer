import { useEffect, useMemo, useState } from "react";
import PosterCard, { PosterItem } from "./components/PosterCard";

type DataFile = {
  generated_at: number;
  grid: { rows: number; cols: number };
  items: PosterItem[];
};

function rgbToCss(rgb: [number, number, number]) {
  return `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
}

export default function App() {
  const [data, setData] = useState<DataFile | null>(null);
  const [sortMode, setSortMode] = useState<"grid" | "hue">("grid");

  useEffect(() => {
    fetch("/posters.json").then((r) => r.json()).then(setData);
  }, []);

  const itemsSorted = useMemo(() => {
    if (!data) return [];
    const items = [...data.items];
    if (sortMode === "grid") {
      // Snake order for nice reading
      const { rows, cols } = data.grid;
      return items.sort((a, b) => {
        if (a.grid_pos.row !== b.grid_pos.row) return a.grid_pos.row - b.grid_pos.row;
        const r = a.grid_pos.row;
        const av = r % 2 === 0 ? a.grid_pos.col : cols - 1 - a.grid_pos.col;
        const bv = r % 2 === 0 ? b.grid_pos.col : cols - 1 - b.grid_pos.col;
        return av - bv;
      });
    }
    // Hue-based fallback: sort by average hue then lightness-ish
    return items.sort((a, b) => {
      const ah = avgHue(a.hue_histogram);
      const bh = avgHue(b.hue_histogram);
      if (ah !== bh) return ah - bh;
      const al = a.avg_rgb.reduce((s, x) => s + x, 0);
      const bl = b.avg_rgb.reduce((s, x) => s + x, 0);
      return al - bl;
    });
  }, [data, sortMode]);

  // Background gradient based on first/last columns' average colors
  useEffect(() => {
    if (!data) return;
    const { rows, cols } = data.grid;
    const left: [number, number, number][] = [];
    const right: [number, number, number][] = [];
    for (const it of data.items) {
      if (it.grid_pos.col === 0) left.push(it.avg_rgb);
      if (it.grid_pos.col === cols - 1) right.push(it.avg_rgb);
    }
    const avg = (arr: number[][]) => {
      if (!arr.length) return [240, 240, 240] as [number, number, number];
      const [r, g, b] = arr.reduce((acc, cur) => [acc[0] + cur[0], acc[1] + cur[1], acc[2] + cur[2]], [0, 0, 0]);
      return [Math.round(r / arr.length), Math.round(g / arr.length), Math.round(b / arr.length)] as [number, number, number];
    };
    const L = avg(left as any);
    const R = avg(right as any);
    document.documentElement.style.setProperty("--bg", `${L[0]},${L[1]},${L[2]}`);
    document.body.style.background = `linear-gradient(90deg, ${rgbToCss(L)} 0%, ${rgbToCss(R)} 100%)`;
  }, [data]);

  if (!data) return <div className="p-8 text-center">Loading…</div>;

  const { rows, cols } = data.grid;
  return (
    <div className="min-h-screen px-6 py-8">
      <div className="max-w-7xl mx-auto">
        <header className="flex items-end justify-between gap-4 mb-6">
          <div>
            <h1 className="text-3xl font-semibold tracking-tight">Poster Palette Explorer</h1>
            <p className="text-black/70">{rows * cols} posters • generated {new Date(data.generated_at * 1000).toLocaleString()}</p>
          </div>
          <div className="flex items-center gap-2">
            <label className="text-sm font-medium">Order:</label>
            <select
              className="px-3 py-2 rounded-xl border border-black/10 bg-white shadow-soft text-sm"
              value={sortMode}
              onChange={(e) => setSortMode(e.target.value as any)}
            >
              <option value="grid">Similarity grid</option>
              <option value="hue">Average hue</option>
            </select>
          </div>
        </header>

        <section
          className="grid gap-4"
          style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }}
        >
          {itemsSorted.map((it) => (
            <PosterCard key={`${it.title}-${it.grid_pos.row}-${it.grid_pos.col}`} item={it} />)
          )}
        </section>

        <footer className="mt-10 text-center text-sm text-black/60">
          Built with TMDb data. This product uses the TMDb API but is not endorsed or certified by TMDb.
        </footer>
      </div>
    </div>
  );
}

function avgHue(hist: number[]) {
  // circular mean of 12-sector hue histogram
  const n = hist.length;
  let x = 0, y = 0;
  for (let i = 0; i < n; i++) {
    const theta = (i / n) * 2 * Math.PI;
    x += Math.cos(theta) * hist[i];
    y += Math.sin(theta) * hist[i];
  }
  const angle = Math.atan2(y, x);
  return (angle < 0 ? angle + 2 * Math.PI : angle);
}
