import { useEffect, useRef, useImperativeHandle, forwardRef } from 'react'
import cytoscape from 'cytoscape'
import type { FullGraphResponse } from '../../types'

export interface CytoscapeGraphHandle {
  fit: () => void
  relayout: () => void
  applyFilters: (showVibe: boolean, showDetail: boolean) => void
  applySearch: (query: string) => void
  selectAndCenter: (nodeId: string) => void
}

interface Props {
  data: FullGraphResponse | null
  onNodeClick: (nodeId: string) => void
  onBackgroundClick: () => void
  onStatusChange: (status: string) => void
}

const CytoscapeGraph = forwardRef<CytoscapeGraphHandle, Props>(
  ({ data, onNodeClick, onBackgroundClick, onStatusChange }, ref) => {
    const containerRef = useRef<HTMLDivElement>(null)
    const cyRef = useRef<cytoscape.Core | null>(null)
    const degreeRef = useRef<Record<string, number>>({})

    useImperativeHandle(ref, () => ({
      fit() {
        cyRef.current?.fit()
      },
      relayout() {
        const cy = cyRef.current
        if (!cy || !data) return
        const posMap = new Map(
          data.nodes
            .filter(n => n.position != null)
            .map(n => [n.id, n.position!])
        )
        if (posMap.size === 0) return
        cy.batch(() => {
          cy.nodes().forEach(node => {
            const pos = posMap.get(node.id())
            if (pos) node.position(pos)
          })
        })
        cy.fit()
      },
      applyFilters(showVibe: boolean, showDetail: boolean) {
        const cy = cyRef.current
        if (!cy) return
        cy.batch(() => {
          cy.nodes().forEach(node => {
            const type = node.data('type')
            if ((type === 'vibe' && !showVibe) || (type === 'detail' && !showDetail)) {
              node.style('display', 'none')
            } else {
              node.style('display', 'element')
            }
          })
        })
      },
      applySearch(query: string) {
        const cy = cyRef.current
        if (!cy) return
        const term = query.trim().toLowerCase()
        cy.batch(() => {
          if (!term) {
            cy.nodes().forEach(node => {
              node.style({ opacity: 1, 'border-width': 0, 'border-color': '#fff' })
            })
            cy.edges().style({ opacity: undefined as unknown as number })
            return
          }
          cy.nodes().forEach(node => {
            const text = (node.data('fullText') || '').toLowerCase()
            if (text.includes(term)) {
              node.style({ opacity: 1, 'border-width': 3, 'border-color': '#fff' })
            } else {
              node.style({ opacity: 0.15, 'border-width': 0 })
            }
          })
          cy.edges().style({ opacity: 0.05 })
        })
      },
      selectAndCenter(nodeId: string) {
        const cy = cyRef.current
        if (!cy) return
        const node = cy.getElementById(nodeId)
        if (node.length) {
          cy.animate({ center: { eles: node }, zoom: 2 } as cytoscape.AnimateOptions, { duration: 300 })
          node.select()
        }
      },
    }))

    useEffect(() => {
      if (!data || !containerRef.current) return

      // Build degree map
      const degree: Record<string, number> = {}
      for (const edge of data.edges) {
        degree[edge.source_id] = (degree[edge.source_id] || 0) + 1
        degree[edge.target_id] = (degree[edge.target_id] || 0) + 1
      }
      degreeRef.current = degree

      const hasPositions = data.nodes.every(n => n.position != null)

      const elements: cytoscape.ElementDefinition[] = []
      for (const node of data.nodes) {
        const isVibe = node.type === 'vibe'
        const el: cytoscape.ElementDefinition = {
          data: {
            id: node.id,
            label: node.text.slice(0, 60),
            type: node.type,
            fullText: node.text,
            color: isVibe ? '#6366f1' : '#14b8a6',
            borderColor: isVibe ? 'rgba(20, 184, 166, 0.25)' : 'rgba(99, 102, 241, 0.25)',
          },
        }
        if (hasPositions && node.position) {
          el.position = { x: node.position.x, y: node.position.y }
        }
        elements.push(el)
      }
      for (const edge of data.edges) {
        elements.push({
          data: {
            id: edge.source_id + '-' + edge.target_id,
            source: edge.source_id,
            target: edge.target_id,
            weight: edge.weight,
          },
        })
      }

      const cy = cytoscape({
        container: containerRef.current,
        elements,
        style: [
          {
            selector: 'node',
            style: {
              label: '',
              width(ele: cytoscape.NodeSingular) {
                const d = degree[ele.id()] || 1
                return Math.min(8 + d * 3, 40)
              },
              height(ele: cytoscape.NodeSingular) {
                const d = degree[ele.id()] || 1
                return Math.min(8 + d * 3, 40)
              },
              'background-color': 'data(color)',
              'border-width': 1,
              'border-color': 'data(borderColor)',
            } as unknown as cytoscape.Css.Node,
          },
          {
            selector: 'node:selected',
            style: { 'border-width': 2, 'border-color': '#fff' },
          },
          {
            selector: 'edge',
            style: {
              width(ele: cytoscape.EdgeSingular) {
                return 0.5 + ele.data('weight') * 3
              },
              'line-color': '#2a2a35',
              opacity(ele: cytoscape.EdgeSingular) {
                return 0.2 + ele.data('weight') * 0.6
              },
              'curve-style': 'bezier',
            } as unknown as cytoscape.Css.Edge,
          },
          {
            selector: 'edge.edge-highlighted',
            style: {
              'line-color': '#818cf8',
              opacity: 0.9,
              width(ele: cytoscape.EdgeSingular) {
                return 1.5 + ele.data('weight') * 3
              },
            } as unknown as cytoscape.Css.Edge,
          },
          {
            selector: 'node.neighbor-highlighted',
            style: { 'border-width': 1.5, 'border-color': '#818cf8' },
          },
        ],
        layout: { name: 'preset' },
        autoungrabify: false,
        minZoom: 0.1,
        maxZoom: 5,
      })

      cy.on('tap', 'node', evt => {
        cy.edges().removeClass('edge-highlighted')
        cy.nodes().removeClass('neighbor-highlighted')
        const node = evt.target
        const connected = node.connectedEdges()
        connected.addClass('edge-highlighted')
        connected.connectedNodes().not(node).addClass('neighbor-highlighted')
        onNodeClick(node.id())
      })

      cy.on('tap', evt => {
        if (evt.target === cy) {
          cy.edges().removeClass('edge-highlighted')
          cy.nodes().removeClass('neighbor-highlighted')
          onBackgroundClick()
        }
      })

      cyRef.current = cy

      if (hasPositions) {
        onStatusChange(`${data.nodes.length} nodes, ${data.edges.length} edges (pre-rendered)`)
        requestAnimationFrame(() => cy.fit())
      } else {
        onStatusChange(`${data.nodes.length} nodes, ${data.edges.length} edges — laying out...`)
        requestAnimationFrame(() => {
          cy.layout({
            name: 'cose',
            animate: true,
            animationDuration: 500,
            fit: true,
            nodeOverlap: 20,
            componentSpacing: 40,
          } as cytoscape.LayoutOptions).run()
          cy.one('layoutstop', () => {
            onStatusChange(`${data.nodes.length} nodes, ${data.edges.length} edges`)
          })
        })
      }

      return () => {
        cy.destroy()
        cyRef.current = null
      }
    }, [data]) // eslint-disable-line react-hooks/exhaustive-deps

    return <div ref={containerRef} className="cy-container" />
  },
)

CytoscapeGraph.displayName = 'CytoscapeGraph'
export default CytoscapeGraph
