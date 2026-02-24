interface StatCardProps {
  label: string
  value: number
  className?: string
}

export default function StatCard({ label, value, className }: StatCardProps) {
  return (
    <div className="stat-card">
      <div className="stat-label">{label}</div>
      <div className={`stat-value${className ? ' ' + className : ''}`}>
        {value.toLocaleString()}
      </div>
    </div>
  )
}
