import { useGraphStore } from '../store/graphStore'

export function StatusBar() {
  const graph = useGraphStore(s => s.graph)
  const backendConnected = useGraphStore(s => s.backendConnected)
  const checkpointFilename = useGraphStore(s => s.checkpointFilename)

  const nodeCount = graph?.nodes.length ?? 0
  const edgeCount = graph?.edges.length ?? 0

  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      gap: '16px',
      padding: '4px 12px',
      background: '#0a0a18',
      borderTop: '1px solid #2a2a4a',
      color: '#888',
      fontSize: '11px',
      flexShrink: 0
    }}>
      {checkpointFilename && (
        <span>{checkpointFilename}</span>
      )}
      {graph && (
        <>
          <span>{nodeCount} nodes</span>
          <span>{edgeCount} edges</span>
          <span>{graph.model_name}</span>
        </>
      )}
      <div style={{ flex: 1 }} />
      <span>
        Backend: {backendConnected ? 'Connected' : 'Disconnected'}
      </span>
      <span>Space=fit | Esc=deselect | +/-=zoom | DblClick=expand</span>
    </div>
  )
}
