import { useState, useCallback, useRef } from 'react'
import { fetchRecalls } from '../../api'
import { usePolling } from '../../hooks/usePolling'
import type { Recall } from '../../types'
import RecallCard from './RecallCard'

const PAGE_SIZE = 20
const POLL_INTERVAL = 7000

export default function RecallsTab() {
  const [recalls, setRecalls] = useState<Recall[]>([])
  const [sessionFilter, setSessionFilter] = useState('')
  const [hasMore, setHasMore] = useState(false)
  const [loaded, setLoaded] = useState(false)
  const recallIdsRef = useRef(new Set<string>())

  const load = useCallback(async (append = false) => {
    try {
      const params: { limit: number; session_id?: string } = { limit: PAGE_SIZE }
      if (sessionFilter) params.session_id = sessionFilter

      const data = await fetchRecalls(params)

      if (append) {
        setRecalls(prev => {
          const existing = new Set(prev.map(r => r.recall_id))
          const newOnes = data.recalls.filter(r => !existing.has(r.recall_id))
          return [...prev, ...newOnes]
        })
      } else {
        setRecalls(data.recalls)
        recallIdsRef.current = new Set(data.recalls.map(r => r.recall_id))
      }

      setHasMore(data.recalls.length >= PAGE_SIZE)
      setLoaded(true)
    } catch (err) {
      console.error('Failed to load recalls:', err)
    }
  }, [sessionFilter])

  // Poll — prepend new recalls and update existing ones (e.g. reflections)
  const poll = useCallback(async () => {
    try {
      const params: { limit: number; session_id?: string } = { limit: PAGE_SIZE }
      if (sessionFilter) params.session_id = sessionFilter

      const data = await fetchRecalls(params)
      const freshById = new Map(data.recalls.map(r => [r.recall_id, r]))

      setRecalls(prev => {
        // Update existing recalls with fresh data
        const updated = prev.map(r => freshById.get(r.recall_id) ?? r)
        // Prepend genuinely new ones
        const existingIds = new Set(prev.map(r => r.recall_id))
        const toAdd = data.recalls.filter(r => !existingIds.has(r.recall_id))
        for (const r of toAdd) recallIdsRef.current.add(r.recall_id)
        return toAdd.length > 0 ? [...toAdd, ...updated] : updated
      })
    } catch {
      // Silent poll failure
    }
  }, [sessionFilter])

  // Initial load + polling
  usePolling(loaded ? poll : () => load(), POLL_INTERVAL, true)

  const handleSessionClick = useCallback((sid: string) => {
    setSessionFilter(sid)
    setLoaded(false)
    recallIdsRef.current.clear()
  }, [])

  const handleLoad = useCallback(() => {
    setLoaded(false)
    recallIdsRef.current.clear()
  }, [])

  return (
    <div className="recalls-tab">
      <div className="recalls-toolbar">
        <input
          type="text"
          placeholder="Filter by session ID..."
          value={sessionFilter}
          onChange={e => setSessionFilter(e.target.value)}
        />
        <button onClick={handleLoad}>Load</button>
      </div>
      <div className="recalls-list">
        {recalls.map(r => (
          <RecallCard
            key={r.recall_id}
            recall={r}
            onSessionClick={handleSessionClick}
          />
        ))}
      </div>
      {hasMore && (
        <button className="recalls-more" onClick={() => load(true)}>
          Load more
        </button>
      )}
    </div>
  )
}
