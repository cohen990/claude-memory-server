import { useState } from 'react'
import { fetchDreamOperations } from '../../api'
import type { DreamRun, DreamOperation } from '../../types'
import DreamOperationRow from './DreamOperationRow'

function formatTime(iso: string | null): string {
  if (!iso) return 'running...'
  return new Date(iso).toLocaleString()
}

interface ChipProps {
  label: string
  count: number
  color: string
}

function Chip({ label, count, color }: ChipProps) {
  if (count === 0) return null
  return (
    <span className="dream-chip" style={{ borderColor: color, color }}>
      {count} {label}
    </span>
  )
}

export default function DreamRunCard({ run }: { run: DreamRun }) {
  const [expanded, setExpanded] = useState(false)
  const [ops, setOps] = useState<DreamOperation[] | null>(null)
  const [loading, setLoading] = useState(false)

  const handleToggle = async () => {
    if (expanded) {
      setExpanded(false)
      return
    }
    setExpanded(true)
    if (ops === null) {
      setLoading(true)
      try {
        const data = await fetchDreamOperations(run.id)
        setOps(data.operations)
      } catch (err) {
        console.error('Failed to load operations:', err)
      } finally {
        setLoading(false)
      }
    }
  }

  return (
    <div className="dream-card">
      <div className="dream-card-header" onClick={handleToggle}>
        <span className={`dream-type-badge ${run.type}`}>{run.type}</span>
        <span className="dream-time">{formatTime(run.started_at)}</span>
        {!run.finished_at && <span className="dream-running">running</span>}
      </div>
      <div className="dream-chips">
        <Chip label="created" count={run.nodes_created} color="var(--reflection-U)" />
        <Chip label="merged" count={run.nodes_merged} color="var(--reflection-I)" />
        <Chip label="edges" count={run.edges_created} color="var(--reflection-N)" />
        <Chip label="adjusted" count={run.edges_adjusted} color="var(--reflection-D)" />
        <Chip label="resynthesized" count={run.nodes_resynthesized} color="var(--vibe)" />
        {run.chunks_processed > 0 && (
          <span className="dream-chip" style={{ borderColor: 'var(--text-dim)', color: 'var(--text-dim)' }}>
            {run.chunks_processed} chunks
          </span>
        )}
      </div>
      {run.error && <div className="dream-error">{run.error}</div>}
      {expanded && (
        <div className="dream-ops">
          {loading && <div className="dream-ops-loading">Loading...</div>}
          {ops && ops.length === 0 && <div className="dream-ops-loading">No operations</div>}
          {ops && ops.map(op => (
            <DreamOperationRow key={op.id} op={op} />
          ))}
        </div>
      )}
    </div>
  )
}
