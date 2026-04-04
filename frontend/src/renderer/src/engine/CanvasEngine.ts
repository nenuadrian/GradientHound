import { Camera } from './Camera'
import { Renderer } from './Renderer'
import { InputHandler } from './InputHandler'
import { LayoutEngine } from './LayoutEngine'
import { SceneGraph } from './SceneGraph'
import { HitTester } from './HitTester'
import { EngineEvent, EngineCallbacks } from './types'
import { ModelGraph } from '../types/graph'

export class CanvasEngine {
  private camera: Camera
  private renderer: Renderer
  private inputHandler: InputHandler
  private layoutEngine: LayoutEngine
  private sceneGraph: SceneGraph
  private hitTester: HitTester

  private dirty = true
  private animFrameId: number | null = null
  private callbacks: EngineCallbacks = {}
  private selectedId: string | null = null
  private hoveredId: string | null = null
  private searchMatches = new Set<string>()
  private expandedGroups = new Set<string>()
  private currentGraph: ModelGraph | null = null
  private isDragging = false

  constructor(canvas: HTMLCanvasElement) {
    this.camera = new Camera()
    this.renderer = new Renderer(canvas)
    this.layoutEngine = new LayoutEngine()
    this.sceneGraph = new SceneGraph()
    this.hitTester = new HitTester()
    this.inputHandler = new InputHandler(canvas, this.handleEvent.bind(this))

    this.startRenderLoop()
  }

  dispose(): void {
    if (this.animFrameId !== null) {
      cancelAnimationFrame(this.animFrameId)
    }
    this.inputHandler.dispose()
    this.renderer.dispose()
  }

  setCallbacks(callbacks: EngineCallbacks): void {
    this.callbacks = callbacks
  }

  async setGraph(graph: ModelGraph, expandedGroups: Set<string>): Promise<void> {
    this.currentGraph = graph
    this.expandedGroups = expandedGroups
    await this.relayout()

    // Auto-fit on first load
    const bounds = this.sceneGraph.getBoundingRect()
    if (bounds) {
      const { width, height } = this.renderer.getCanvasSize()
      this.camera.fitToRect(bounds, width, height)
    }
    this.markDirty()
  }

  async updateExpandedGroups(expandedGroups: Set<string>): Promise<void> {
    this.expandedGroups = expandedGroups
    await this.relayout()
    this.markDirty()
  }

  setSelection(nodeId: string | null): void {
    this.selectedId = nodeId
    this.markDirty()
  }

  setSearchMatches(matches: Set<string>): void {
    this.searchMatches = matches
    this.markDirty()
  }

  fitToView(): void {
    const bounds = this.sceneGraph.getBoundingRect()
    if (bounds) {
      const { width, height } = this.renderer.getCanvasSize()
      this.camera.fitToRect(bounds, width, height)
      this.markDirty()
    }
  }

  getZoom(): number {
    return this.camera.zoom
  }

  private async relayout(): Promise<void> {
    if (!this.currentGraph) return
    const { nodes, edges } = await this.layoutEngine.layout(
      this.currentGraph,
      this.expandedGroups
    )
    this.sceneGraph.update(nodes, edges)
  }

  private handleEvent(event: EngineEvent): void {
    switch (event.type) {
      case 'pan':
        this.isDragging = true
        this.camera.pan(event.dx, event.dy)
        this.markDirty()
        break

      case 'zoom':
        this.camera.zoomAt(event.screenX, event.screenY, event.factor)
        this.markDirty()
        break

      case 'click': {
        this.isDragging = false
        const hit = this.hitTester.test(
          event.worldX, event.worldY,
          this.camera, this.sceneGraph.nodes
        )
        const newSelection = hit?.nodeId ?? null
        if (newSelection !== this.selectedId) {
          this.selectedId = newSelection
          this.callbacks.onNodeSelected?.(newSelection)
          this.markDirty()
        }
        break
      }

      case 'dblclick': {
        const hit = this.hitTester.test(
          event.worldX, event.worldY,
          this.camera, this.sceneGraph.nodes
        )
        if (hit) {
          const node = this.sceneGraph.nodes.get(hit.nodeId)
          if (node && !node.isLeaf) {
            this.callbacks.onNodeToggled?.(hit.nodeId)
          }
        }
        break
      }

      case 'hover': {
        const hit = this.hitTester.test(
          event.worldX, event.worldY,
          this.camera, this.sceneGraph.nodes
        )
        const newHover = hit?.nodeId ?? null
        if (newHover !== this.hoveredId) {
          this.hoveredId = newHover
          this.callbacks.onNodeHovered?.(newHover)
          this.markDirty()
        }
        break
      }

      case 'key':
        if (event.key === 'Escape') {
          this.selectedId = null
          this.callbacks.onNodeSelected?.(null)
          this.markDirty()
        } else if (event.key === '=' || event.key === '+') {
          const { width, height } = this.renderer.getCanvasSize()
          this.camera.zoomAt(width / 2, height / 2, 1.2)
          this.markDirty()
        } else if (event.key === '-') {
          const { width, height } = this.renderer.getCanvasSize()
          this.camera.zoomAt(width / 2, height / 2, 0.8)
          this.markDirty()
        } else if (event.key === ' ') {
          this.fitToView()
        }
        break
    }
  }

  private markDirty(): void {
    this.dirty = true
  }

  private startRenderLoop(): void {
    const loop = () => {
      if (this.dirty) {
        this.dirty = false
        this.renderer.render(
          this.camera,
          this.sceneGraph,
          this.selectedId,
          this.hoveredId,
          this.searchMatches
        )
      }
      this.animFrameId = requestAnimationFrame(loop)
    }
    this.animFrameId = requestAnimationFrame(loop)
  }
}
