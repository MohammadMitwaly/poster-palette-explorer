import { motion } from "framer-motion";
import HueChart from "./HueChart";

export type PosterItem = {
  title: string;
  year?: number | null;
  tmdb_id?: number | null;
  poster_url: string;
  avg_rgb: [number, number, number];
  palette: { rgb: [number, number, number]; weight: number }[];
  hue_histogram: number[];
  grid_pos: { row: number; col: number };
};

function rgbToCss(rgb: [number, number, number]) {
  return `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
}

export default function PosterCard({ item }: { item: PosterItem }) {
  const top3 = (item.palette || []).slice(0, 3);
  return (
    <motion.div
      layout
      className="overflow-hidden rounded-2xl shadow-soft bg-white/70 backdrop-blur-sm border border-black/5"
      whileHover={{ y: -3 }}
      transition={{ type: "spring", stiffness: 300, damping: 25 }}
    >
      <img src={item.poster_url} alt={item.title} className="w-full aspect-[2/3] object-cover" loading="lazy" />
      <div className="p-3 flex items-center justify-between gap-2">
        <div className="min-w-0">
          <div className="text-sm font-medium truncate" title={item.title}>
            {item.title}
          </div>
          <div className="text-xs text-black/60">{item.year ?? ""}</div>
          <div className="mt-2 flex gap-1">
            {top3.map((p, i) => (
              <div
                key={i}
                className="h-3 w-6 rounded"
                title={`#${i + 1} ${(p.weight * 100).toFixed(1)}%`}
                style={{ backgroundColor: rgbToCss(p.rgb) }}
              />
            ))}
          </div>
        </div>
        <HueChart histogram={item.hue_histogram} />
      </div>
    </motion.div>
  );
}
