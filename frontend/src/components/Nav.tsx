export type Tab = 'graph' | 'recalls' | 'dreams' | 'stats'

interface NavProps {
  active: Tab
  onChange: (tab: Tab) => void
}

const tabs: { id: Tab; label: string }[] = [
  { id: 'graph', label: 'Graph' },
  { id: 'recalls', label: 'Recalls' },
  { id: 'dreams', label: 'Dreams' },
  { id: 'stats', label: 'Stats' },
]

export default function Nav({ active, onChange }: NavProps) {
  return (
    <nav>
      <span className="logo">memory</span>
      {tabs.map(t => (
        <button
          key={t.id}
          className={`tab${t.id === active ? ' active' : ''}`}
          onClick={() => onChange(t.id)}
        >
          {t.label}
        </button>
      ))}
    </nav>
  )
}
