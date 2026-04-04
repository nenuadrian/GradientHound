import { create } from 'zustand'
import { ModelGraph, GraphNode } from '../types/graph'
import { apiClient } from '../api/client'

interface GraphState {
  graph: ModelGraph | null
  expandedGroups: Set<string>
  selectedNodeId: string | null
  hoveredNodeId: string | null
  checkpointFilename: string | null
  checkpointDir: string | null
  isLoading: boolean
  error: string | null
  backendConnected: boolean

  // Actions
  loadGraph: (filename: string, dir?: string) => Promise<void>
  toggleGroup: (nodeId: string) => void
  selectNode: (nodeId: string | null) => void
  hoverNode: (nodeId: string | null) => void
  setBackendConnected: (connected: boolean) => void
  getSelectedNode: () => GraphNode | null
}

export const useGraphStore = create<GraphState>((set, get) => ({
  graph: null,
  expandedGroups: new Set<string>(),
  selectedNodeId: null,
  hoveredNodeId: null,
  checkpointFilename: null,
  checkpointDir: null,
  isLoading: false,
  error: null,
  backendConnected: false,

  loadGraph: async (filename: string, dir?: string) => {
    set({ isLoading: true, error: null })
    try {
      const graph = await apiClient.getGraph(filename, dir)
      // Auto-expand root
      const expanded = new Set<string>()
      expanded.add(graph.root_id)
      set({
        graph,
        expandedGroups: expanded,
        checkpointFilename: filename,
        checkpointDir: dir ?? null,
        isLoading: false,
        selectedNodeId: null
      })
    } catch (e) {
      set({ error: String(e), isLoading: false })
    }
  },

  toggleGroup: (nodeId: string) => {
    set(state => {
      const expanded = new Set(state.expandedGroups)
      if (expanded.has(nodeId)) {
        expanded.delete(nodeId)
      } else {
        expanded.add(nodeId)
      }
      return { expandedGroups: expanded }
    })
  },

  selectNode: (nodeId: string | null) => set({ selectedNodeId: nodeId }),
  hoverNode: (nodeId: string | null) => set({ hoveredNodeId: nodeId }),
  setBackendConnected: (connected: boolean) => set({ backendConnected: connected }),

  getSelectedNode: () => {
    const { graph, selectedNodeId } = get()
    if (!graph || !selectedNodeId) return null
    return graph.nodes.find(n => n.id === selectedNodeId) ?? null
  }
}))
