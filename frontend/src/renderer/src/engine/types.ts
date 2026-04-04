export interface Rect {
  x: number
  y: number
  width: number
  height: number
}

export interface VisualNode {
  id: string
  name: string
  op: string
  isLeaf: boolean
  moduleType?: string
  parentId?: string
  children: string[]
  attributes: Record<string, unknown>
  inputShapes: unknown[][]
  outputShapes: unknown[][]
  paramCount: number
  rect: Rect
  depth: number
  isExpanded: boolean
}

export interface VisualEdge {
  id: string
  source: string
  target: string
  points: Array<{ x: number; y: number }>
}

export type EngineEvent =
  | { type: 'pan'; dx: number; dy: number }
  | { type: 'zoom'; screenX: number; screenY: number; factor: number }
  | { type: 'click'; worldX: number; worldY: number }
  | { type: 'dblclick'; worldX: number; worldY: number }
  | { type: 'hover'; worldX: number; worldY: number }
  | { type: 'key'; key: string; ctrl: boolean; shift: boolean }

export interface EngineCallbacks {
  onNodeSelected?: (nodeId: string | null) => void
  onNodeHovered?: (nodeId: string | null) => void
  onNodeToggled?: (nodeId: string) => void
}
