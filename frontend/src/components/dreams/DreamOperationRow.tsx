import type { DreamOperation } from '../../types'

const OP_COLORS: Record<string, string> = {
  node_created: 'var(--reflection-U)',
  node_merged: 'var(--reflection-I)',
  edge_created: 'var(--reflection-N)',
  edge_adjusted: 'var(--reflection-D)',
  node_resynthesized: 'var(--vibe)',
  error: 'var(--reflection-M)',
}

function formatDetail(op: DreamOperation): string {
  const d = op.detail
  switch (op.operation) {
    case 'node_created':
      return String(d.text ?? '')
    case 'node_merged':
      return `sim=${typeof d.similarity === 'number' ? d.similarity.toFixed(2) : '?'} ${d.text ?? ''}`
    case 'edge_created': {
      const src = d.source_text ? String(d.source_text).slice(0, 60) : String(d.source_id ?? '').slice(0, 8)
      const tgt = d.target_text ? String(d.target_text).slice(0, 60) : String(d.target_id ?? '').slice(0, 8)
      return `${src} → ${tgt} (w=${typeof d.weight === 'number' ? d.weight.toFixed(2) : '?'})`
    }
    case 'edge_adjusted': {
      const src = d.source_text ? String(d.source_text).slice(0, 40) : String(d.source_id ?? '').slice(0, 8)
      const tgt = d.target_text ? String(d.target_text).slice(0, 40) : String(d.target_id ?? '').slice(0, 8)
      return `${src} → ${tgt} ${typeof d.old_weight === 'number' ? d.old_weight.toFixed(3) : '?'}→${typeof d.new_weight === 'number' ? d.new_weight.toFixed(3) : '?'} (${d.reflection ?? '?'})`
    }
    case 'node_resynthesized':
      return `"${d.old_text ?? ''}" → "${d.new_text ?? ''}"`
    case 'error':
      return String(d.message ?? '')
    default:
      return JSON.stringify(d)
  }
}

export default function DreamOperationRow({ op }: { op: DreamOperation }) {
  const color = OP_COLORS[op.operation] ?? 'var(--text-dim)'

  return (
    <div className="dream-op-row">
      <span className="dream-op-badge" style={{ borderColor: color, color }}>
        {op.operation.replace('node_', '').replace('edge_', '')}
      </span>
      {op.node_type && (
        <span className={`dream-op-type ${op.node_type}`}>{op.node_type}</span>
      )}
      <span className="dream-op-detail">{formatDetail(op)}</span>
    </div>
  )
}
