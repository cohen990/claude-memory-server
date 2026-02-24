interface ReflectionBadgeProps {
  code: string
}

export default function ReflectionBadge({ code }: ReflectionBadgeProps) {
  return (
    <span className={`reflection-badge reflection-${code}`}>{code}</span>
  )
}
