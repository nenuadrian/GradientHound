import { useRef, useEffect, useCallback } from 'react'
import { CanvasEngine } from '../engine/CanvasEngine'
import { useGraphStore } from '../store/graphStore'

export function GraphCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const engineRef = useRef<CanvasEngine | null>(null)

  const graph = useGraphStore(s => s.graph)
  const expandedGroups = useGraphStore(s => s.expandedGroups)
  const selectNode = useGraphStore(s => s.selectNode)
  const hoverNode = useGraphStore(s => s.hoverNode)
  const toggleGroup = useGraphStore(s => s.toggleGroup)

  // Initialize engine
  useEffect(() => {
    if (!canvasRef.current) return
    const engine = new CanvasEngine(canvasRef.current)
    engineRef.current = engine

    engine.setCallbacks({
      onNodeSelected: (id) => selectNode(id),
      onNodeHovered: (id) => hoverNode(id),
      onNodeToggled: (id) => toggleGroup(id)
    })

    return () => {
      engine.dispose()
      engineRef.current = null
    }
  }, [])

  // Update graph when data or expansion changes
  useEffect(() => {
    if (!engineRef.current || !graph) return
    engineRef.current.setGraph(graph, expandedGroups)
  }, [graph])

  // Update layout when expand state changes
  useEffect(() => {
    if (!engineRef.current || !graph) return
    engineRef.current.updateExpandedGroups(expandedGroups)
  }, [expandedGroups])

  return (
    <canvas
      ref={canvasRef}
      style={{
        width: '100%',
        height: '100%',
        display: 'block',
        cursor: 'default'
      }}
    />
  )
}
