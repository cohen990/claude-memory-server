import { useState, useCallback, useRef } from 'react'
import { fetchDreamRuns } from '../../api'
import { usePolling } from '../../hooks/usePolling'
import type { DreamRun } from '../../types'
import DreamRunCard from './DreamRunCard'

const POLL_INTERVAL = 7000

export default function DreamsTab() {
  const [runs, setRuns] = useState<DreamRun[]>([])
  const [loaded, setLoaded] = useState(false)
  const runIdsRef = useRef(new Set<string>())

  const load = useCallback(async () => {
    try {
      const data = await fetchDreamRuns(50)
      setRuns(data.runs)
      runIdsRef.current = new Set(data.runs.map(r => r.id))
      setLoaded(true)
    } catch (err) {
      console.error('Failed to load dream runs:', err)
    }
  }, [])

  const poll = useCallback(async () => {
    try {
      const data = await fetchDreamRuns(50)
      const newRuns = data.runs.filter(r => !runIdsRef.current.has(r.id))
      if (newRuns.length > 0) {
        for (const r of newRuns) runIdsRef.current.add(r.id)
        setRuns(data.runs)
      } else {
        // Check for updated runs (e.g. finished_at changed)
        setRuns(prev => {
          const changed = data.runs.some((r, i) =>
            i < prev.length && (r.finished_at !== prev[i].finished_at || r.error !== prev[i].error)
          )
          return changed || data.runs.length !== prev.length ? data.runs : prev
        })
      }
    } catch {
      // Silent poll failure
    }
  }, [])

  usePolling(loaded ? poll : load, POLL_INTERVAL, true)

  return (
    <div className="dreams-tab">
      <div className="dreams-list">
        {runs.length === 0 && loaded && (
          <div className="dreams-empty">No dream runs yet</div>
        )}
        {runs.map(r => (
          <DreamRunCard key={r.id} run={r} />
        ))}
      </div>
    </div>
  )
}
