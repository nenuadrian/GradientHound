import { useCallback, useState, useEffect } from 'react'
import { useGraphStore } from '../store/graphStore'
import { useUIStore } from '../store/uiStore'
import { apiClient } from '../api/client'
import { CheckpointInfo } from '../types/api'

export function Toolbar() {
  const loadGraph = useGraphStore(s => s.loadGraph)
  const isLoading = useGraphStore(s => s.isLoading)
  const checkpointFilename = useGraphStore(s => s.checkpointFilename)
  const setBackendConnected = useGraphStore(s => s.setBackendConnected)
  const backendConnected = useGraphStore(s => s.backendConnected)
  const toggleSidePanel = useUIStore(s => s.toggleSidePanel)
  const setSearchOpen = useUIStore(s => s.setSearchOpen)

  const [checkpoints, setCheckpoints] = useState<CheckpointInfo[]>([])
  const [checkpointDir, setCheckpointDir] = useState('')

  // Check backend health on mount
  useEffect(() => {
    const check = async () => {
      const ok = await apiClient.health()
      setBackendConnected(ok)
      if (ok) {
        const list = await apiClient.listCheckpoints()
        setCheckpoints(list)
      }
    }
    check()
    const interval = setInterval(check, 10000)
    return () => clearInterval(interval)
  }, [])

  const handleOpenFile = useCallback(async () => {
    // In Electron, use the native dialog
    if (window.electronAPI) {
      const filePath = await window.electronAPI.openFile()
      if (filePath) {
        // Extract directory and filename
        const parts = filePath.replace(/\\/g, '/').split('/')
        const filename = parts.pop()!
        const dir = parts.join('/')
        setCheckpointDir(dir)
        await loadGraph(filename, dir)
      }
    }
  }, [loadGraph])

  const handleOpenDir = useCallback(async () => {
    if (window.electronAPI) {
      const dirPath = await window.electronAPI.openDirectory()
      if (dirPath) {
        setCheckpointDir(dirPath)
        const list = await apiClient.listCheckpoints(dirPath)
        setCheckpoints(list)
      }
    }
  }, [])

  const handleLoadCheckpoint = useCallback(async (filename: string) => {
    await loadGraph(filename, checkpointDir || undefined)
  }, [loadGraph, checkpointDir])

  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      gap: '8px',
      padding: '6px 12px',
      background: '#0d0d1a',
      borderBottom: '1px solid #2a2a4a',
      color: '#e0e0e0',
      fontSize: '13px',
      flexShrink: 0
    }}>
      <span style={{ fontWeight: 'bold', color: '#ff4444', marginRight: '8px' }}>
        GradientHound
      </span>

      <button onClick={handleOpenFile} style={btnStyle} title="Open .ghound file">
        Open File
      </button>

      <button onClick={handleOpenDir} style={btnStyle} title="Open checkpoint directory">
        Open Dir
      </button>

      <div style={{ width: 1, height: 20, background: '#2a2a4a' }} />

      {checkpoints.length > 0 && (
        <select
          onChange={e => e.target.value && handleLoadCheckpoint(e.target.value)}
          value={checkpointFilename || ''}
          style={{
            background: '#1a1a2e',
            color: '#e0e0e0',
            border: '1px solid #2a2a4a',
            borderRadius: '4px',
            padding: '4px 8px',
            fontSize: '12px'
          }}
        >
          <option value="">Select checkpoint...</option>
          {checkpoints.map(c => (
            <option key={c.filename} value={c.filename}>
              {c.model_name} - {c.filename}
            </option>
          ))}
        </select>
      )}

      <div style={{ flex: 1 }} />

      <button onClick={() => setSearchOpen(true)} style={btnStyle} title="Search (Cmd+F)">
        Search
      </button>

      <button onClick={toggleSidePanel} style={btnStyle}>
        Panel
      </button>

      <div style={{
        width: 8,
        height: 8,
        borderRadius: '50%',
        background: backendConnected ? '#4CAF50' : '#F44336',
        title: backendConnected ? 'Backend connected' : 'Backend disconnected'
      }} />

      {isLoading && <span style={{ color: '#90CAF9' }}>Loading...</span>}
    </div>
  )
}

const btnStyle: React.CSSProperties = {
  background: '#1a1a2e',
  color: '#e0e0e0',
  border: '1px solid #2a2a4a',
  borderRadius: '4px',
  padding: '4px 10px',
  cursor: 'pointer',
  fontSize: '12px'
}
