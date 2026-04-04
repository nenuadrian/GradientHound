import { EngineEvent } from './types'
import { ZOOM_SPEED } from './constants'

export class InputHandler {
  private canvas: HTMLCanvasElement
  private onEvent: (event: EngineEvent) => void
  private isDragging = false
  private lastX = 0
  private lastY = 0
  private hoverThrottleId: number | null = null

  constructor(canvas: HTMLCanvasElement, onEvent: (event: EngineEvent) => void) {
    this.canvas = canvas
    this.onEvent = onEvent
    this.attach()
  }

  private attach(): void {
    this.canvas.addEventListener('mousedown', this.onMouseDown)
    this.canvas.addEventListener('mousemove', this.onMouseMove)
    this.canvas.addEventListener('mouseup', this.onMouseUp)
    this.canvas.addEventListener('mouseleave', this.onMouseUp)
    this.canvas.addEventListener('wheel', this.onWheel, { passive: false })
    this.canvas.addEventListener('click', this.onClick)
    this.canvas.addEventListener('dblclick', this.onDblClick)
    window.addEventListener('keydown', this.onKeyDown)
  }

  dispose(): void {
    this.canvas.removeEventListener('mousedown', this.onMouseDown)
    this.canvas.removeEventListener('mousemove', this.onMouseMove)
    this.canvas.removeEventListener('mouseup', this.onMouseUp)
    this.canvas.removeEventListener('mouseleave', this.onMouseUp)
    this.canvas.removeEventListener('wheel', this.onWheel)
    this.canvas.removeEventListener('click', this.onClick)
    this.canvas.removeEventListener('dblclick', this.onDblClick)
    window.removeEventListener('keydown', this.onKeyDown)
    if (this.hoverThrottleId !== null) cancelAnimationFrame(this.hoverThrottleId)
  }

  private onMouseDown = (e: MouseEvent): void => {
    if (e.button === 0) {
      this.isDragging = true
      this.lastX = e.clientX
      this.lastY = e.clientY
      this.canvas.style.cursor = 'grabbing'
    }
  }

  private onMouseMove = (e: MouseEvent): void => {
    if (this.isDragging) {
      const dx = e.clientX - this.lastX
      const dy = e.clientY - this.lastY
      this.lastX = e.clientX
      this.lastY = e.clientY
      this.onEvent({ type: 'pan', dx, dy })
    } else {
      // Throttle hover events
      if (this.hoverThrottleId === null) {
        this.hoverThrottleId = requestAnimationFrame(() => {
          this.hoverThrottleId = null
        })
        const rect = this.canvas.getBoundingClientRect()
        this.onEvent({
          type: 'hover',
          worldX: e.clientX - rect.left,
          worldY: e.clientY - rect.top
        })
      }
    }
  }

  private onMouseUp = (): void => {
    this.isDragging = false
    this.canvas.style.cursor = 'default'
  }

  private onWheel = (e: WheelEvent): void => {
    e.preventDefault()
    const rect = this.canvas.getBoundingClientRect()
    const screenX = e.clientX - rect.left
    const screenY = e.clientY - rect.top

    if (e.ctrlKey || e.metaKey) {
      // Zoom
      const factor = 1 - e.deltaY * ZOOM_SPEED
      this.onEvent({ type: 'zoom', screenX, screenY, factor })
    } else {
      // Pan
      this.onEvent({ type: 'pan', dx: -e.deltaX, dy: -e.deltaY })
    }
  }

  private onClick = (e: MouseEvent): void => {
    if (this.isDragging) return
    const rect = this.canvas.getBoundingClientRect()
    this.onEvent({
      type: 'click',
      worldX: e.clientX - rect.left,
      worldY: e.clientY - rect.top
    })
  }

  private onDblClick = (e: MouseEvent): void => {
    const rect = this.canvas.getBoundingClientRect()
    this.onEvent({
      type: 'dblclick',
      worldX: e.clientX - rect.left,
      worldY: e.clientY - rect.top
    })
  }

  private onKeyDown = (e: KeyboardEvent): void => {
    this.onEvent({
      type: 'key',
      key: e.key,
      ctrl: e.ctrlKey || e.metaKey,
      shift: e.shiftKey
    })
  }
}
