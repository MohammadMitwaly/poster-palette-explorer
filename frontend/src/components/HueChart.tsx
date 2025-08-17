import { Pie, PieChart, ResponsiveContainer, Cell } from "recharts";

const HUES = Array.from({ length: 12 }, (_, i) => i * 30);
const HUE_COLORS = HUES.map((h) => `hsl(${h}deg 80% 55%)`);

// Default size tuned for a 5-column layout
export default function HueChart({
  histogram,
  size = 72,
  thickness = 12,
}: {
  histogram: number[];
  size?: number;
  thickness?: number;
}) {
  const data = histogram.map((v, i) => ({ name: `${i}`, value: v || 0.0001 }));
  const outer = Math.floor(size / 2) - 2;
  const inner = Math.max(outer - thickness, 1);
  return (
    <div style={{ width: size, height: size }}>
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie
            dataKey="value"
            data={data}
            innerRadius={inner}
            outerRadius={outer}
            isAnimationActive={false}
          >
            {data.map((_, i) => (
              <Cell key={i} fill={HUE_COLORS[i % HUE_COLORS.length]} />
            ))}
          </Pie>
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
}