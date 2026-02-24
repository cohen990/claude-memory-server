import { useState, useEffect, useRef, useCallback } from 'react'
import { fetchGraph, fetchNode, fetchNeighbors } from '../../api'
import type { FullGraphResponse, GraphNode, GraphEdge, Neighbor } from '../../types'
import GraphToolbar from './GraphToolbar'
import CytoscapeGraph, { type CytoscapeGraphHandle } from './CytoscapeGraph'
import Sidebar from './Sidebar'

export default function GraphTab() {
  const [data, setData] = useState<FullGraphResponse | null>(null)
  const [status, setStatus] = useState('Loading graph...')

  // Filters
  const [showVibe, setShowVibe] = useState(true)
  const [showDetail, setShowDetail] = useState(true)
  const [searchQuery, setSearchQuery] = useState('')

  // Sidebar
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null)
  const [selectedEdges, setSelectedEdges] = useState<GraphEdge[]>([])
  const [neighbors, setNeighbors] = useState<Neighbor[]>([])

  const cyRef = useRef<CytoscapeGraphHandle | null>(null)
  const searchTimerRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)

  useEffect(() => {
    fetchGraph()
      .then(setData)
      .catch(err => setStatus('Error: ' + err.message))
  }, [])

  // Filter toggle effects
  const handleToggleVibe = useCallback(() => {
    setShowVibe(v => {
      const next = !v
      cyRef.current?.applyFilters(next, showDetail)
      return next
    })
  }, [showDetail])

  const handleToggleDetail = useCallback(() => {
    setShowDetail(v => {
      const next = !v
      cyRef.current?.applyFilters(showVibe, next)
      return next
    })
  }, [showVibe])

  // Debounced search
  const handleSearchChange = useCallback((q: string) => {
    setSearchQuery(q)
    clearTimeout(searchTimerRef.current)
    searchTimerRef.current = setTimeout(() => {
      cyRef.current?.applySearch(q)
    }, 200)
  }, [])

  const loadNodeSidebar = useCallback(async (nodeId: string) => {
    try {
      const [nodeData, neighborsData] = await Promise.all([
        fetchNode(nodeId),
        fetchNeighbors(nodeId),
      ])
      setSelectedNode(nodeData.node)
      setSelectedEdges(nodeData.edges)
      setNeighbors(neighborsData.neighbors)
    } catch (err) {
      console.error('Failed to load node:', err)
    }
  }, [])

  const handleNodeClick = useCallback((nodeId: string) => {
    loadNodeSidebar(nodeId)
  }, [loadNodeSidebar])

  const handleClose = useCallback(() => {
    setSelectedNode(null)
  }, [])

  const handleSelectNeighbor = useCallback((nodeId: string) => {
    cyRef.current?.selectAndCenter(nodeId)
    loadNodeSidebar(nodeId)
  }, [loadNodeSidebar])

  return (
    <>
      <GraphToolbar
        showVibe={showVibe}
        showDetail={showDetail}
        onToggleVibe={handleToggleVibe}
        onToggleDetail={handleToggleDetail}
        onFit={() => cyRef.current?.fit()}
        onRelayout={() => cyRef.current?.relayout()}
        searchQuery={searchQuery}
        onSearchChange={handleSearchChange}
        status={status}
      />
      <CytoscapeGraph
        ref={cyRef}
        data={data}
        onNodeClick={handleNodeClick}
        onBackgroundClick={handleClose}
        onStatusChange={setStatus}
      />
      {selectedNode && (
        <Sidebar
          node={selectedNode}
          edges={selectedEdges}
          neighbors={neighbors}
          onClose={handleClose}
          onSelectNeighbor={handleSelectNeighbor}
        />
      )}
    </>
  )
}
