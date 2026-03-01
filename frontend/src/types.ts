// API response types for the memory browser

export interface GraphNode {
  id: string
  type: 'vibe' | 'detail'
  text: string
  created_at: string
  updated_at: string
  source_ids: string[]
  position?: { x: number; y: number } | null
}

export interface GraphEdge {
  source_id: string
  target_id: string
  weight: number
  created_at: string
  last_activated: string | null
  activation_count: number
}

export interface FullGraphResponse {
  nodes: GraphNode[]
  edges: GraphEdge[]
}

export interface NodeDetailResponse {
  node: GraphNode
  edges: GraphEdge[]
}

export interface Neighbor {
  node: GraphNode
  edge: GraphEdge
}

export interface NeighborsResponse {
  neighbors: Neighbor[]
}

export interface RecallResult {
  node_id: string
  similarity: number
  source: string
  connected_via: string | null
  reflection: string | null
  type: string | null
  text: string | null
}

export interface Recall {
  recall_id: string
  created_at: string
  session_id: string | null
  query_text?: string
  results: RecallResult[]
}

export interface RecallsResponse {
  recalls: Recall[]
}

export interface GraphStats {
  total_nodes: number
  nodes_by_type: { vibe: number; detail: number }
  total_edges: number
  activated_edges: number
}

export interface ChromaDBStats {
  total_documents: number
  total_subchunks: number
  total_user_inputs: number
  queue_pending: number
  pending_dream?: number
}

export interface StatsResponse {
  graph: GraphStats
  reflections: Record<string, number> | null
  chromadb: ChromaDBStats | null
}

export interface TimelineBucket {
  bucket: string
  U: number
  I: number
  N: number
  D: number
  M: number
}

export interface DreamRun {
  id: string
  type: 'consolidate' | 'reconsolidate'
  started_at: string
  finished_at: string | null
  chunks_processed: number
  nodes_created: number
  nodes_merged: number
  edges_created: number
  edges_adjusted: number
  nodes_resynthesized: number
  error: string | null
}

export interface DreamOperation {
  id: number
  run_id: string
  timestamp: string
  operation: string
  node_id: string | null
  node_type: string | null
  detail: Record<string, unknown>
}
