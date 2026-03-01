interface ReflectionBadgeProps {
  code: string
}

export default function ReflectionBadge({ code }: ReflectionBadgeProps) {
  const cls = code === '?' ? 'pending' : code
  return (
    <span className={`reflection-badge reflection-${cls}`}>{code}</span>
  )
}
