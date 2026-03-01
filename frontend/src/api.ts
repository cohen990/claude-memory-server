import type {
  FullGraphResponse,
  NodeDetailResponse,
  NeighborsResponse,
  RecallsResponse,
  StatsResponse,
  TimelineBucket,
  DreamRun,
  DreamOperation,
} from './types'

async function get<T>(url: string): Promise<T> {
  const resp = await fetch(url)
  if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText}`)
  return resp.json()
}

export function fetchGraph(): Promise<FullGraphResponse> {
  return get('/api/graph')
}

export function fetchNode(id: string): Promise<NodeDetailResponse> {
  return get(`/api/nodes/${encodeURIComponent(id)}`)
}

export function fetchNeighbors(id: string): Promise<NeighborsResponse> {
  return get(`/api/nodes/${encodeURIComponent(id)}/neighbors`)
}

export function fetchRecalls(params?: {
  limit?: number
  session_id?: string
}): Promise<RecallsResponse> {
  const qs = new URLSearchParams()
  if (params?.limit) qs.set('limit', String(params.limit))
  if (params?.session_id) qs.set('session_id', params.session_id)
  const q = qs.toString()
  return get(`/api/recalls${q ? '?' + q : ''}`)
}

export function fetchStats(): Promise<StatsResponse> {
  return get('/api/stats')
}

export function fetchReflectionTimeline(): Promise<TimelineBucket[]> {
  return get('/api/reflection-timeline')
}

export function fetchDreamRuns(limit?: number): Promise<{ runs: DreamRun[] }> {
  const qs = limit ? `?limit=${limit}` : ''
  return get(`/api/dream-runs${qs}`)
}

export function fetchDreamOperations(runId: string): Promise<{ operations: DreamOperation[] }> {
  return get(`/api/dream-runs/${encodeURIComponent(runId)}/operations`)
}
