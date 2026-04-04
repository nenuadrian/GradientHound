import { VisualNode, VisualEdge, Rect } from './types'
import { Camera } from './Camera'
import { SceneGraph } from './SceneGraph'
import {
  COLORS,
  OP_NODE_RADIUS,
  LAYER_RADIUS,
  LAYER_HEADER_HEIGHT,
  LOD_HIDE_TEXT,
  LOD_SIMPLIFY_NODES
} from './constants'

export class Renderer {
  private canvas: HTMLCanvasElement
  private ctx: CanvasRenderingContext2D
  private dpr = 1
  private resizeObserver: ResizeObserver

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas
    this.ctx = canvas.getContext('2d')!
    this.dpr = window.devicePixelRatio || 1
    this.resize()

    this.resizeObserver = new ResizeObserver(() => this.resize())
    this.resizeObserver.observe(canvas)
  }

  dispose(): void {
    this.resizeObserver.disconnect()
  }

  getCanvasSize(): { width: number; height: number } {
    return {
      width: this.canvas.clientWidth,
      height: this.canvas.clientHeight
    }
  }

  private resize(): void {
    this.dpr = window.devicePixelRatio || 1
    this.canvas.width = this.canvas.clientWidth * this.dpr
    this.canvas.height = this.canvas.clientHeight * this.dpr
  }

  render(
    camera: Camera,
    scene: SceneGraph,
    selectedId: string | null,
    hoveredId: string | null,
    searchMatches: Set<string>
  ): void {
    const ctx = this.ctx
    const { width, height } = this.getCanvasSize()
    const viewport = camera.getViewportRect(width, height)

    // Reset transform and clear
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0)
    ctx.fillStyle = COLORS.background
    ctx.fillRect(0, 0, width, height)

    // Draw grid dots
    this.drawGrid(ctx, camera, width, height)

    // Apply camera transform
    ctx.save()
    ctx.setTransform(
      this.dpr * camera.zoom, 0, 0,
      this.dpr * camera.zoom,
      -camera.x * this.dpr * camera.zoom,
      -camera.y * this.dpr * camera.zoom
    )

    const visibleEdges = scene.getVisibleEdges(viewport)
    const visibleNodes = scene.getVisibleNodes(viewport)

    // Build edge lookup for selection highlighting
    const selectedEdges = new Map<string, 'input' | 'output' | 'selected'>()
    if (selectedId) {
      for (const edge of scene.edges) {
        if (edge.source === selectedId) {
          selectedEdges.set(edge.id, 'output')
        } else if (edge.target === selectedId) {
          selectedEdges.set(edge.id, 'input')
        }
      }
    }

    // Draw edges
    for (const edge of visibleEdges) {
      const edgeType = selectedEdges.get(edge.id)
      this.drawEdge(ctx, edge, camera.zoom, edgeType)
    }

    // Draw nodes
    for (const node of visibleNodes) {
      const isSelected = node.id === selectedId
      const isHovered = node.id === hoveredId
      const isMatch = searchMatches.has(node.id)
      this.drawNode(ctx, node, camera.zoom, isSelected, isHovered, isMatch)
    }

    ctx.restore()
  }

  private drawGrid(
    ctx: CanvasRenderingContext2D,
    camera: Camera,
    width: number,
    height: number
  ): void {
    const gridSize = 40
    const viewport = camera.getViewportRect(width, height)
    const startX = Math.floor(viewport.x / gridSize) * gridSize
    const startY = Math.floor(viewport.y / gridSize) * gridSize

    ctx.save()
    ctx.setTransform(
      this.dpr * camera.zoom, 0, 0,
      this.dpr * camera.zoom,
      -camera.x * this.dpr * camera.zoom,
      -camera.y * this.dpr * camera.zoom
    )
    ctx.fillStyle = COLORS.gridDot
    const dotSize = 1.5 / camera.zoom

    for (let x = startX; x < viewport.x + viewport.width; x += gridSize) {
      for (let y = startY; y < viewport.y + viewport.height; y += gridSize) {
        ctx.fillRect(x - dotSize / 2, y - dotSize / 2, dotSize, dotSize)
      }
    }
    ctx.restore()
  }

  private drawNode(
    ctx: CanvasRenderingContext2D,
    node: VisualNode,
    zoom: number,
    isSelected: boolean,
    isHovered: boolean,
    isMatch: boolean
  ): void {
    const { x, y, width, height } = node.rect

    if (zoom < LOD_SIMPLIFY_NODES) {
      // Ultra-simplified: just a colored rectangle
      ctx.fillStyle = node.isLeaf ? COLORS.opNodeBg : COLORS.layerHeaderBg
      ctx.fillRect(x, y, width, height)
      return
    }

    if (node.isLeaf) {
      this.drawOpNode(ctx, node, zoom, isSelected, isHovered, isMatch)
    } else if (node.isExpanded) {
      this.drawExpandedLayer(ctx, node, zoom, isSelected, isHovered, isMatch)
    } else {
      this.drawCollapsedLayer(ctx, node, zoom, isSelected, isHovered, isMatch)
    }
  }

  private drawOpNode(
    ctx: CanvasRenderingContext2D,
    node: VisualNode,
    zoom: number,
    isSelected: boolean,
    isHovered: boolean,
    isMatch: boolean
  ): void {
    const { x, y, width, height } = node.rect

    // Background
    ctx.fillStyle = isHovered ? COLORS.opNodeHoverBg : COLORS.opNodeBg
    this.roundRect(ctx, x, y, width, height, OP_NODE_RADIUS)
    ctx.fill()

    // Search match highlight
    if (isMatch) {
      ctx.fillStyle = COLORS.searchMatch
      this.roundRect(ctx, x, y, width, height, OP_NODE_RADIUS)
      ctx.fill()
    }

    // Border
    ctx.strokeStyle = isSelected ? COLORS.selectedBorder : COLORS.opNodeBorder
    ctx.lineWidth = isSelected ? 2.5 / zoom : 1 / zoom
    this.roundRect(ctx, x, y, width, height, OP_NODE_RADIUS)
    ctx.stroke()

    // Selection glow
    if (isSelected) {
      ctx.shadowColor = COLORS.selectedGlow
      ctx.shadowBlur = 12 / zoom
      ctx.strokeStyle = COLORS.selectedBorder
      ctx.lineWidth = 2 / zoom
      this.roundRect(ctx, x, y, width, height, OP_NODE_RADIUS)
      ctx.stroke()
      ctx.shadowBlur = 0
    }

    // Label
    if (zoom > LOD_HIDE_TEXT) {
      ctx.fillStyle = COLORS.opNodeText
      ctx.font = `${12 / zoom < 3 ? 12 : 12}px Inter, system-ui, sans-serif`
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'

      // Primary label: name
      const label = node.name
      ctx.fillText(label, x + width / 2, y + height / 2 - (node.moduleType ? 6 : 0), width - 12)

      // Secondary label: module type (smaller, dimmer)
      if (node.moduleType) {
        const shortType = node.moduleType.split('.').pop() || ''
        ctx.fillStyle = '#999'
        ctx.font = `${10}px Inter, system-ui, sans-serif`
        ctx.fillText(shortType, x + width / 2, y + height / 2 + 8, width - 12)
      }
    }
  }

  private drawExpandedLayer(
    ctx: CanvasRenderingContext2D,
    node: VisualNode,
    zoom: number,
    isSelected: boolean,
    isHovered: boolean,
    isMatch: boolean
  ): void {
    const { x, y, width, height } = node.rect

    // Body background
    ctx.fillStyle = COLORS.layerBodyBg
    this.roundRect(ctx, x, y, width, height, LAYER_RADIUS)
    ctx.fill()

    if (isMatch) {
      ctx.fillStyle = COLORS.searchMatch
      this.roundRect(ctx, x, y, width, height, LAYER_RADIUS)
      ctx.fill()
    }

    // Body border
    ctx.strokeStyle = isSelected ? COLORS.selectedBorder : COLORS.layerBodyBorder
    ctx.lineWidth = isSelected ? 2.5 / zoom : 1 / zoom
    this.roundRect(ctx, x, y, width, height, LAYER_RADIUS)
    ctx.stroke()

    // Header bar
    ctx.fillStyle = COLORS.layerHeaderBg
    this.roundRectTop(ctx, x, y, width, LAYER_HEADER_HEIGHT, LAYER_RADIUS)
    ctx.fill()

    // Header text
    if (zoom > LOD_HIDE_TEXT) {
      ctx.fillStyle = COLORS.layerHeaderText
      ctx.font = `bold 12px Inter, system-ui, sans-serif`
      ctx.textAlign = 'left'
      ctx.textBaseline = 'middle'

      // Collapse arrow
      const arrowX = x + 12
      const arrowY = y + LAYER_HEADER_HEIGHT / 2
      ctx.beginPath()
      ctx.moveTo(arrowX, arrowY - 4)
      ctx.lineTo(arrowX + 6, arrowY)
      ctx.lineTo(arrowX, arrowY + 4)
      ctx.closePath()
      // Rotate arrow down for expanded
      ctx.save()
      ctx.translate(arrowX + 3, arrowY)
      ctx.rotate(Math.PI / 2)
      ctx.translate(-(arrowX + 3), -arrowY)
      ctx.beginPath()
      ctx.moveTo(arrowX, arrowY - 4)
      ctx.lineTo(arrowX + 6, arrowY)
      ctx.lineTo(arrowX, arrowY + 4)
      ctx.closePath()
      ctx.fill()
      ctx.restore()

      const label = `${node.name} (${node.children.length})`
      ctx.fillText(label, x + 28, y + LAYER_HEADER_HEIGHT / 2, width - 40)
    }
  }

  private drawCollapsedLayer(
    ctx: CanvasRenderingContext2D,
    node: VisualNode,
    zoom: number,
    isSelected: boolean,
    isHovered: boolean,
    isMatch: boolean
  ): void {
    const { x, y, width, height } = node.rect

    // Background
    ctx.fillStyle = isHovered ? COLORS.opNodeHoverBg : COLORS.layerCollapsedBg
    this.roundRect(ctx, x, y, width, height, OP_NODE_RADIUS)
    ctx.fill()

    if (isMatch) {
      ctx.fillStyle = COLORS.searchMatch
      this.roundRect(ctx, x, y, width, height, OP_NODE_RADIUS)
      ctx.fill()
    }

    // Border
    ctx.strokeStyle = isSelected ? COLORS.selectedBorder : COLORS.layerHeaderBg
    ctx.lineWidth = isSelected ? 2.5 / zoom : 1.5 / zoom
    this.roundRect(ctx, x, y, width, height, OP_NODE_RADIUS)
    ctx.stroke()

    // Label
    if (zoom > LOD_HIDE_TEXT) {
      ctx.fillStyle = COLORS.layerHeaderBg
      ctx.font = `bold 12px Inter, system-ui, sans-serif`
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'

      // Right arrow for collapsed
      const arrowX = x + 14
      const arrowY = y + height / 2
      ctx.beginPath()
      ctx.moveTo(arrowX, arrowY - 4)
      ctx.lineTo(arrowX + 5, arrowY)
      ctx.lineTo(arrowX, arrowY + 4)
      ctx.closePath()
      ctx.fill()

      const label = `${node.name} (${node.children.length})`
      ctx.fillText(label, x + width / 2 + 8, y + height / 2, width - 40)
    }
  }

  private drawEdge(
    ctx: CanvasRenderingContext2D,
    edge: VisualEdge,
    zoom: number,
    type?: 'input' | 'output' | 'selected'
  ): void {
    if (edge.points.length < 2) return

    ctx.beginPath()
    ctx.moveTo(edge.points[0].x, edge.points[0].y)

    if (edge.points.length === 2) {
      ctx.lineTo(edge.points[1].x, edge.points[1].y)
    } else {
      // Smooth curve through bend points
      for (let i = 1; i < edge.points.length - 1; i++) {
        const curr = edge.points[i]
        const next = edge.points[i + 1]
        const midX = (curr.x + next.x) / 2
        const midY = (curr.y + next.y) / 2
        ctx.quadraticCurveTo(curr.x, curr.y, midX, midY)
      }
      const last = edge.points[edge.points.length - 1]
      ctx.lineTo(last.x, last.y)
    }

    // Style based on type
    switch (type) {
      case 'input':
        ctx.strokeStyle = COLORS.edgeInput
        ctx.lineWidth = 2 / zoom
        break
      case 'output':
        ctx.strokeStyle = COLORS.edgeOutput
        ctx.lineWidth = 2 / zoom
        break
      case 'selected':
        ctx.strokeStyle = COLORS.edgeSelected
        ctx.lineWidth = 2.5 / zoom
        break
      default:
        ctx.strokeStyle = COLORS.edgeDefault
        ctx.lineWidth = 1 / zoom
    }

    ctx.stroke()

    // Arrow head at the end
    const last = edge.points[edge.points.length - 1]
    const prev = edge.points[edge.points.length - 2]
    this.drawArrowHead(ctx, prev.x, prev.y, last.x, last.y, zoom, type)
  }

  private drawArrowHead(
    ctx: CanvasRenderingContext2D,
    fromX: number, fromY: number,
    toX: number, toY: number,
    zoom: number,
    type?: 'input' | 'output' | 'selected'
  ): void {
    const size = 6 / Math.max(zoom, 0.3)
    const angle = Math.atan2(toY - fromY, toX - fromX)

    ctx.save()
    ctx.translate(toX, toY)
    ctx.rotate(angle)
    ctx.beginPath()
    ctx.moveTo(0, 0)
    ctx.lineTo(-size, -size / 2)
    ctx.lineTo(-size, size / 2)
    ctx.closePath()

    switch (type) {
      case 'input': ctx.fillStyle = COLORS.edgeInput; break
      case 'output': ctx.fillStyle = COLORS.edgeOutput; break
      case 'selected': ctx.fillStyle = COLORS.edgeSelected; break
      default: ctx.fillStyle = COLORS.edgeArrow
    }
    ctx.fill()
    ctx.restore()
  }

  private roundRect(
    ctx: CanvasRenderingContext2D,
    x: number, y: number, w: number, h: number, r: number
  ): void {
    ctx.beginPath()
    ctx.moveTo(x + r, y)
    ctx.lineTo(x + w - r, y)
    ctx.quadraticCurveTo(x + w, y, x + w, y + r)
    ctx.lineTo(x + w, y + h - r)
    ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h)
    ctx.lineTo(x + r, y + h)
    ctx.quadraticCurveTo(x, y + h, x, y + h - r)
    ctx.lineTo(x, y + r)
    ctx.quadraticCurveTo(x, y, x + r, y)
    ctx.closePath()
  }

  private roundRectTop(
    ctx: CanvasRenderingContext2D,
    x: number, y: number, w: number, h: number, r: number
  ): void {
    ctx.beginPath()
    ctx.moveTo(x + r, y)
    ctx.lineTo(x + w - r, y)
    ctx.quadraticCurveTo(x + w, y, x + w, y + r)
    ctx.lineTo(x + w, y + h)
    ctx.lineTo(x, y + h)
    ctx.lineTo(x, y + r)
    ctx.quadraticCurveTo(x, y, x + r, y)
    ctx.closePath()
  }
}
