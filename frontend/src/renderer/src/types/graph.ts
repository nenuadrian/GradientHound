export interface GraphNode {
  id: string
  name: string
  op: string
  is_leaf: boolean
  module_type?: string
  parent_id?: string
  children: string[]
  attributes: Record<string, unknown>
  input_shapes: unknown[][]
  output_shapes: unknown[][]
  param_count: number
}

export interface GraphEdge {
  id: string
  source: string
  target: string
  tensor_shape?: Record<string, unknown>
}

export interface ModelGraph {
  model_name: string
  model_class: string
  nodes: GraphNode[]
  edges: GraphEdge[]
  root_id: string
  input_nodes: string[]
  output_nodes: string[]
}
