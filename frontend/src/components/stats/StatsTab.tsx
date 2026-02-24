import { useState, useCallback } from 'react'
import { fetchStats, fetchReflectionTimeline } from '../../api'
import { usePolling } from '../../hooks/usePolling'
import type { StatsResponse, TimelineBucket } from '../../types'
import StatCard from './StatCard'
import ReflectionLineChart from './ReflectionLineChart'
import ReflectionStackedChart from './ReflectionStackedChart'

const POLL_INTERVAL = 7000

interface CardDef {
  key: string
  label: string
  value: number
  cls?: string
}

export default function StatsTab() {
  const [stats, setStats] = useState<StatsResponse | null>(null)
  const [timeline, setTimeline] = useState<TimelineBucket[]>([])
  const [error, setError] = useState<string | null>(null)

  const loadAll = useCallback(async () => {
    try {
      const [s, t] = await Promise.all([
        fetchStats(),
        fetchReflectionTimeline(),
      ])
      setStats(s)
      setTimeline(t)
      setError(null)
    } catch (err) {
      if (!stats) setError((err as Error).message)
    }
  }, [stats])

  usePolling(loadAll, POLL_INTERVAL)

  if (error) {
    return (
      <div className="stats-tab">
        <div style={{ color: 'var(--reflection-M)' }}>Error: {error}</div>
      </div>
    )
  }

  if (!stats) {
    return <div className="stats-tab">Loading stats...</div>
  }

  const cards: CardDef[] = []

  // Graph stats
  cards.push({ key: 'graph-nodes', label: 'Graph Nodes', value: stats.graph.total_nodes })
  if (stats.graph.nodes_by_type) {
    cards.push({ key: 'vibes', label: 'Vibes', value: stats.graph.nodes_by_type.vibe || 0, cls: 'vibe' })
    cards.push({ key: 'details', label: 'Details', value: stats.graph.nodes_by_type.detail || 0, cls: 'detail' })
  }
  cards.push({ key: 'graph-edges', label: 'Graph Edges', value: stats.graph.total_edges })
  cards.push({ key: 'activated-edges', label: 'Activated Edges', value: stats.graph.activated_edges })

  // ChromaDB stats
  if (stats.chromadb) {
    cards.push({ key: 'conversations', label: 'Conversations', value: stats.chromadb.total_documents })
    cards.push({ key: 'subchunks', label: 'Subchunks', value: stats.chromadb.total_subchunks })
    cards.push({ key: 'user-inputs', label: 'User Inputs', value: stats.chromadb.total_user_inputs })
    cards.push({ key: 'queue-pending', label: 'Queue Pending', value: stats.chromadb.queue_pending })
    if (stats.chromadb.pending_dream != null) {
      cards.push({ key: 'pending-dream', label: 'Pending Dream', value: stats.chromadb.pending_dream })
    }
  }

  // Reflection distribution
  if (stats.reflections) {
    for (const [code, count] of Object.entries(stats.reflections)) {
      cards.push({ key: `refl-${code}`, label: `Reflection: ${code}`, value: count })
    }
  }

  return (
    <div className="stats-tab">
      <div className="stats-grid">
        {cards.map(c => (
          <StatCard key={c.key} label={c.label} value={c.value} className={c.cls} />
        ))}
      </div>
      <ReflectionLineChart data={timeline} />
      <ReflectionStackedChart data={timeline} />
    </div>
  )
}
