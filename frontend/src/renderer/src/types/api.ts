export interface CheckpointManifest {
  format: string
  version: string
  created_at: string
  pytorch_version?: string
  collector_version?: string
}

export interface CheckpointMetadata {
  step?: number
  epoch?: number
  loss?: number
}

export interface CheckpointInfo {
  filename: string
  manifest: CheckpointManifest
  metadata: CheckpointMetadata
  model_name: string
  node_count: number
  edge_count: number
  param_count: number
}

export interface ParameterStats {
  name: string
  shape: number[]
  dtype: string
  numel: number
  mean: number
  std: number
  min: number
  max: number
  histogram: number[]
  histogram_edges: number[]
}
