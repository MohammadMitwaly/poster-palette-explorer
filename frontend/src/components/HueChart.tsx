import { Pie, PieChart, ResponsiveContainer, Cell } from "recharts";

const HUES = Array.from({ length: 12 }, (_, i) => i * 30);
const HUE_COLORS = HUES.map(h => `hsl(${h}deg 80% 55%)`);

export default function HueChart({ histogram }: { histogram: number[] }) {
  const data = histogram.map((v, i) => ({ name: `${i}`, value: v || 0.0001 }));
  return (
    <div style={{ width: 72, height: 72 }}>
      <ResponsiveContainer>
        <PieChart>
          <Pie dataKey="value" data={data} innerRadius={18} outerRadius={32} isAnimationActive={false}>
            {data.map((_, i) => (
              <Cell key={i} fill={HUE_COLORS[i % HUE_COLORS.length]} />
            ))}
          </Pie>
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
}
