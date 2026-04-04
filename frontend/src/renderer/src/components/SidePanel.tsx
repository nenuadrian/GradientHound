import { useUIStore } from '../store/uiStore'
import { NodeDetails } from './NodeDetails'

export function SidePanel() {
  const sidePanelOpen = useUIStore(s => s.sidePanelOpen)

  if (!sidePanelOpen) return null

  return (
    <div style={{
      width: '320px',
      flexShrink: 0,
      background: '#0d0d1a',
      borderLeft: '1px solid #2a2a4a',
      overflowY: 'auto',
      overflowX: 'hidden'
    }}>
      <div style={{
        padding: '8px 12px',
        borderBottom: '1px solid #2a2a4a',
        fontSize: '12px',
        fontWeight: 'bold',
        color: '#90CAF9',
        textTransform: 'uppercase',
        letterSpacing: '0.5px'
      }}>
        Node Details
      </div>
      <NodeDetails />
    </div>
  )
}
