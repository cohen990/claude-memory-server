import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer,
} from 'recharts'
import type { TimelineBucket } from '../../types'

const CODES = ['U', 'I', 'N', 'D', 'M'] as const
const COLORS: Record<string, string> = {
  U: '#22c55e',
  I: '#3b82f6',
  N: '#eab308',
  D: '#f97316',
  M: '#ef4444',
}
const LABELS: Record<string, string> = {
  U: 'Used',
  I: 'Interesting',
  N: 'Noise',
  D: 'Distracting',
  M: 'Misleading',
}

function formatTick(iso: string): string {
  const d = new Date(iso)
  return `${d.getMonth() + 1}/${d.getDate()} ${d.getHours().toString().padStart(2, '0')}:00`
}

function formatPercent(value: number): string {
  return `${Math.round(value * 100)}%`
}

interface Props {
  data: TimelineBucket[]
}

export default function ReflectionStackedChart({ data }: Props) {
  if (!data.length) return null

  return (
    <div className="chart-container">
      <h4 className="chart-title">Reflection Proportions</h4>
      <ResponsiveContainer width="100%" height={240}>
        <AreaChart data={data} stackOffset="expand">
          <CartesianGrid strokeDasharray="3 3" stroke="#2a2a35" />
          <XAxis
            dataKey="bucket"
            tickFormatter={formatTick}
            stroke="#8888a0"
            fontSize={10}
            angle={-20}
            textAnchor="end"
          />
          <YAxis
            tickFormatter={formatPercent}
            stroke="#8888a0"
            fontSize={10}
          />
          <Tooltip
            contentStyle={{
              background: '#1a1a22',
              border: '1px solid #2a2a35',
              borderRadius: 4,
              fontSize: 11,
            }}
            labelFormatter={formatTick}
          />
          <Legend
            formatter={(value: string) => LABELS[value] || value}
            wrapperStyle={{ fontSize: 10 }}
          />
          {CODES.map(code => (
            <Area
              key={code}
              type="linear"
              dataKey={code}
              stackId="1"
              stroke={COLORS[code]}
              fill={COLORS[code]}
              fillOpacity={0.75}
            />
          ))}
        </AreaChart>
      </ResponsiveContainer>
    </div>
  )
}
