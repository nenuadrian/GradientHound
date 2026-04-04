// Node visual constants
export const OP_NODE_MIN_WIDTH = 140
export const OP_NODE_HEIGHT = 44
export const OP_NODE_RADIUS = 6
export const OP_NODE_FONT_SIZE = 12

export const LAYER_HEADER_HEIGHT = 32
export const LAYER_RADIUS = 8
export const LAYER_PADDING = { top: 40, right: 20, bottom: 20, left: 20 }
export const LAYER_COLLAPSED_WIDTH = 160
export const LAYER_COLLAPSED_HEIGHT = 44

// Colors
export const COLORS = {
  background: '#1a1a2e',
  gridDot: '#2a2a4a',

  // Nodes
  opNodeBg: '#ffffff',
  opNodeBorder: '#d0d0d0',
  opNodeText: '#333333',
  opNodeHoverBg: '#e3f2fd',

  layerHeaderBg: '#1976D2',
  layerHeaderText: '#ffffff',
  layerBodyBg: '#f5f5f5',
  layerBodyBorder: '#e0e0e0',
  layerCollapsedBg: '#e3f2fd',

  // Selection
  selectedBorder: '#1565C0',
  selectedGlow: 'rgba(21, 101, 192, 0.3)',

  // Edges
  edgeDefault: '#90A4AE',
  edgeSelected: '#1976D2',
  edgeInput: '#4CAF50',
  edgeOutput: '#F44336',
  edgeArrow: '#607D8B',

  // Search highlights
  searchMatch: 'rgba(255, 193, 7, 0.4)',
} as const

// Zoom
export const ZOOM_MIN = 0.05
export const ZOOM_MAX = 5.0
export const ZOOM_SPEED = 0.002

// LOD thresholds
export const LOD_HIDE_TEXT = 0.25
export const LOD_SIMPLIFY_NODES = 0.1

// Layout
export const ELK_SPACING_NODE = 30
export const ELK_SPACING_LAYER = 80
