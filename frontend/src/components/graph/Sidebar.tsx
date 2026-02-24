import type { GraphNode, Neighbor, GraphEdge } from '../../types'

interface SidebarProps {
  node: GraphNode | null
  edges: GraphEdge[]
  neighbors: Neighbor[]
  onClose: () => void
  onSelectNeighbor: (id: string) => void
}

function formatDate(iso: string | null | undefined): string {
  if (!iso) return '—'
  const d = new Date(iso)
  return d.toLocaleDateString() + ' ' + d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

export default function Sidebar({ node, edges, neighbors, onClose, onSelectNeighbor }: SidebarProps) {
  if (!node) return null

  return (
    <aside className="sidebar">
      <button className="sidebar-close" onClick={onClose}>&times;</button>
      <h3 className={`sidebar-type ${node.type}`}>{node.type}</h3>
      <p className="sidebar-text">{node.text}</p>
      <div className="sidebar-meta">
        <div>ID: {node.id.slice(0, 8)}...</div>
        <div>Created: {formatDate(node.created_at)}</div>
        <div>Updated: {formatDate(node.updated_at)}</div>
        <div>Sources: {(node.source_ids || []).length}</div>
        <div>Edges: {edges.length}</div>
      </div>
      <h4>Neighbors</h4>
      <ul className="sidebar-neighbors">
        {neighbors.map(n => (
          <li key={n.node.id} onClick={() => onSelectNeighbor(n.node.id)}>
            <span className={`neighbor-type ${n.node.type}`}>{n.node.type}</span>
            <span className="neighbor-weight">w={n.edge.weight.toFixed(2)}</span>
            <span className="neighbor-text">{n.node.text.slice(0, 80)}</span>
          </li>
        ))}
      </ul>
    </aside>
  )
}
