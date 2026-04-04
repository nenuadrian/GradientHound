import { ModelGraph, GraphNode } from '../types/graph'
import { CheckpointInfo, ParameterStats } from '../types/api'

const DEFAULT_URL = 'http://localhost:8642'

class ApiClient {
  baseUrl = DEFAULT_URL

  setBaseUrl(url: string): void {
    this.baseUrl = url.replace(/\/$/, '')
  }

  async health(): Promise<boolean> {
    try {
      const res = await fetch(`${this.baseUrl}/api/health`)
      return res.ok
    } catch {
      return false
    }
  }

  async listCheckpoints(dir?: string): Promise<CheckpointInfo[]> {
    const params = dir ? `?dir=${encodeURIComponent(dir)}` : ''
    const res = await fetch(`${this.baseUrl}/api/checkpoints${params}`)
    if (!res.ok) throw new Error(`Failed to list checkpoints: ${res.statusText}`)
    return res.json()
  }

  async getGraph(filename: string, dir?: string): Promise<ModelGraph> {
    const params = dir ? `?dir=${encodeURIComponent(dir)}` : ''
    const res = await fetch(`${this.baseUrl}/api/checkpoints/${encodeURIComponent(filename)}/graph${params}`)
    if (!res.ok) throw new Error(`Failed to load graph: ${res.statusText}`)
    return res.json()
  }

  async getNode(filename: string, nodeId: string, dir?: string): Promise<GraphNode> {
    const params = dir ? `?dir=${encodeURIComponent(dir)}` : ''
    const res = await fetch(
      `${this.baseUrl}/api/checkpoints/${encodeURIComponent(filename)}/node/${encodeURIComponent(nodeId)}${params}`
    )
    if (!res.ok) throw new Error(`Failed to load node: ${res.statusText}`)
    return res.json()
  }

  async getParameterStats(filename: string, nodeId: string, dir?: string): Promise<ParameterStats> {
    const params = dir ? `?dir=${encodeURIComponent(dir)}` : ''
    const res = await fetch(
      `${this.baseUrl}/api/checkpoints/${encodeURIComponent(filename)}/node/${encodeURIComponent(nodeId)}/params${params}`
    )
    if (!res.ok) throw new Error(`Failed to load params: ${res.statusText}`)
    return res.json()
  }

  async search(filename: string, query: string, regex = false, dir?: string): Promise<GraphNode[]> {
    const params = new URLSearchParams({ q: query })
    if (regex) params.set('regex', 'true')
    if (dir) params.set('dir', dir)
    const res = await fetch(
      `${this.baseUrl}/api/checkpoints/${encodeURIComponent(filename)}/search?${params}`
    )
    if (!res.ok) throw new Error(`Search failed: ${res.statusText}`)
    return res.json()
  }
}

export const apiClient = new ApiClient()
