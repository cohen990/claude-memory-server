interface GraphToolbarProps {
  showVibe: boolean
  showDetail: boolean
  onToggleVibe: () => void
  onToggleDetail: () => void
  onFit: () => void
  onRelayout: () => void
  searchQuery: string
  onSearchChange: (q: string) => void
  status: string
}

export default function GraphToolbar({
  showVibe,
  showDetail,
  onToggleVibe,
  onToggleDetail,
  onFit,
  onRelayout,
  searchQuery,
  onSearchChange,
  status,
}: GraphToolbarProps) {
  return (
    <div className="graph-toolbar">
      <label>
        <input type="checkbox" checked={showVibe} onChange={onToggleVibe} /> Vibes
      </label>
      <label>
        <input type="checkbox" checked={showDetail} onChange={onToggleDetail} /> Details
      </label>
      <button onClick={onFit}>Fit</button>
      <button onClick={onRelayout}>Relayout</button>
      <input
        className="graph-search"
        type="text"
        placeholder="Search nodes..."
        value={searchQuery}
        onChange={e => onSearchChange(e.target.value)}
      />
      <span className="graph-status">{status}</span>
    </div>
  )
}
