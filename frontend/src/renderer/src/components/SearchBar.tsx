import { useRef, useEffect, useCallback } from 'react'
import { useUIStore } from '../store/uiStore'
import { useGraphStore } from '../store/graphStore'
import { apiClient } from '../api/client'

export function SearchBar() {
  const searchOpen = useUIStore(s => s.searchOpen)
  const searchQuery = useUIStore(s => s.searchQuery)
  const setSearchOpen = useUIStore(s => s.setSearchOpen)
  const setSearchQuery = useUIStore(s => s.setSearchQuery)
  const setSearchResults = useUIStore(s => s.setSearchResults)
  const checkpointFilename = useGraphStore(s => s.checkpointFilename)
  const checkpointDir = useGraphStore(s => s.checkpointDir)
  const inputRef = useRef<HTMLInputElement>(null)

  // Cmd+F shortcut
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'f') {
        e.preventDefault()
        setSearchOpen(true)
      }
      if (e.key === 'Escape' && searchOpen) {
        setSearchOpen(false)
        setSearchQuery('')
        setSearchResults([])
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [searchOpen])

  // Auto-focus input
  useEffect(() => {
    if (searchOpen && inputRef.current) {
      inputRef.current.focus()
    }
  }, [searchOpen])

  const handleSearch = useCallback(async (query: string) => {
    setSearchQuery(query)
    if (!query || !checkpointFilename) {
      setSearchResults([])
      return
    }
    try {
      const results = await apiClient.search(
        checkpointFilename, query, false, checkpointDir ?? undefined
      )
      setSearchResults(results.map(n => n.id))
    } catch {
      setSearchResults([])
    }
  }, [checkpointFilename, checkpointDir])

  if (!searchOpen) return null

  return (
    <div style={{
      position: 'absolute',
      top: '48px',
      left: '50%',
      transform: 'translateX(-50%)',
      background: '#1a1a2e',
      border: '1px solid #2a2a4a',
      borderRadius: '8px',
      padding: '8px 12px',
      display: 'flex',
      alignItems: 'center',
      gap: '8px',
      zIndex: 100,
      boxShadow: '0 4px 20px rgba(0,0,0,0.5)'
    }}>
      <input
        ref={inputRef}
        type="text"
        placeholder="Search nodes..."
        value={searchQuery}
        onChange={e => handleSearch(e.target.value)}
        style={{
          background: '#0d0d1a',
          border: '1px solid #2a2a4a',
          borderRadius: '4px',
          color: '#e0e0e0',
          padding: '6px 10px',
          fontSize: '13px',
          width: '300px',
          outline: 'none'
        }}
      />
      <button
        onClick={() => {
          setSearchOpen(false)
          setSearchQuery('')
          setSearchResults([])
        }}
        style={{
          background: 'none',
          border: 'none',
          color: '#888',
          cursor: 'pointer',
          fontSize: '16px'
        }}
      >
        x
      </button>
    </div>
  )
}
