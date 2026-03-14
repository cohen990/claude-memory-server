import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ReferenceLine, ResponsiveContainer,
} from 'recharts'
import type { TimelineBucket, Marker } from '../../types'

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

function snapToHour(iso: string): string {
  return iso.slice(0, 13) + ':00:00'
}

interface Props {
  data: TimelineBucket[]
  markers?: Marker[]
}

export default function ReflectionLineChart({ data, markers }: Props) {
  if (!data.length) return null

  return (
    <div className="chart-container">
      <h4 className="chart-title">Reflection Timeline</h4>
      <ResponsiveContainer width="100%" height={240}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#2a2a35" />
          <XAxis
            dataKey="bucket"
            tickFormatter={formatTick}
            stroke="#8888a0"
            fontSize={10}
            angle={-20}
            textAnchor="end"
          />
          <YAxis stroke="#8888a0" fontSize={10} />
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
          {markers?.map(m => (
            <ReferenceLine
              key={m.id}
              x={snapToHour(m.created_at)}
              stroke="#ffffff"
              strokeDasharray="4 3"
              strokeOpacity={0.6}
              label={{ value: m.label, position: 'top', fill: '#ffffff', fontSize: 9 }}
            />
          ))}
          {CODES.map(code => (
            <Line
              key={code}
              type="linear"
              dataKey={code}
              stroke={COLORS[code]}
              strokeWidth={1.5}
              dot={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
