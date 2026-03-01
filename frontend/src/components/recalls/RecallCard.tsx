import type { Recall } from '../../types'
import ReflectionBadge from './ReflectionBadge'

interface RecallCardProps {
  recall: Recall
  onSessionClick: (sessionId: string) => void
}

function formatDate(iso: string | null | undefined): string {
  if (!iso) return '—'
  const d = new Date(iso)
  return d.toLocaleDateString() + ' ' + d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

export default function RecallCard({ recall, onSessionClick }: RecallCardProps) {
  const sessionLabel = recall.session_id
    ? recall.session_id.slice(0, 12) + '...'
    : 'no session'

  return (
    <div className="recall-card">
      <div className="recall-header">
        <span>{formatDate(recall.created_at)}</span>
        {recall.session_id ? (
          <button
            className="session-link"
            onClick={() => onSessionClick(recall.session_id!)}
          >
            {sessionLabel}
          </button>
        ) : (
          <span>{sessionLabel}</span>
        )}
      </div>

      {(recall.general_surprisal != null || recall.personal_surprisal != null) && (
        <div className="recall-surprisal">
          {recall.general_surprisal != null && recall.personal_surprisal != null && recall.personal_surprisal > 0 && (
            <span
              className="recall-surprisal-ratio"
              title="gen/pers ratio (higher = better retrieval candidate)"
            >
              {(recall.general_surprisal / recall.personal_surprisal).toFixed(2)}
            </span>
          )}
          {recall.general_surprisal != null && (
            <span title="General surprisal (higher = more substantive)">
              gen: {recall.general_surprisal.toFixed(1)}
            </span>
          )}
          {recall.personal_surprisal != null && (
            <span title="Personal surprisal (lower = more familiar)">
              pers: {recall.personal_surprisal.toFixed(1)}
            </span>
          )}
        </div>
      )}

      {recall.query_text && (
        <details className="recall-query">
          <summary>Query</summary>
          <p>{recall.query_text}</p>
        </details>
      )}

      {recall.results.map((r, i) => (
        <div className="recall-result" key={i}>
          <span className={`recall-result-type ${r.type || ''}`}>
            {r.type || '?'}
          </span>
          <span className="recall-result-text">
            {r.text || '(deleted node)'}
          </span>
          <span className="recall-result-meta">
            <span className="recall-result-sim">
              {(r.similarity || 0).toFixed(3)}
            </span>{' '}
            <ReflectionBadge code={r.reflection || '?'} />
          </span>
        </div>
      ))}
    </div>
  )
}
