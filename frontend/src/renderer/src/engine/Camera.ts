import { Rect } from './types'
import { ZOOM_MIN, ZOOM_MAX } from './constants'

export class Camera {
  x = 0
  y = 0
  zoom = 1

  screenToWorld(sx: number, sy: number): { x: number; y: number } {
    return {
      x: sx / this.zoom + this.x,
      y: sy / this.zoom + this.y
    }
  }

  worldToScreen(wx: number, wy: number): { x: number; y: number } {
    return {
      x: (wx - this.x) * this.zoom,
      y: (wy - this.y) * this.zoom
    }
  }

  pan(dx: number, dy: number): void {
    this.x -= dx / this.zoom
    this.y -= dy / this.zoom
  }

  zoomAt(screenX: number, screenY: number, factor: number): void {
    // Record world point under cursor before zoom
    const before = this.screenToWorld(screenX, screenY)

    // Apply zoom
    this.zoom = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, this.zoom * factor))

    // Adjust pan so the same world point stays under the cursor
    const after = this.screenToWorld(screenX, screenY)
    this.x += before.x - after.x
    this.y += before.y - after.y
  }

  fitToRect(rect: Rect, canvasWidth: number, canvasHeight: number, padding = 60): void {
    const scaleX = (canvasWidth - padding * 2) / rect.width
    const scaleY = (canvasHeight - padding * 2) / rect.height
    this.zoom = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, Math.min(scaleX, scaleY)))
    this.x = rect.x - padding / this.zoom
    this.y = rect.y - padding / this.zoom
  }

  applyTransform(ctx: CanvasRenderingContext2D): void {
    ctx.setTransform(this.zoom, 0, 0, this.zoom, -this.x * this.zoom, -this.y * this.zoom)
  }

  getViewportRect(canvasWidth: number, canvasHeight: number): Rect {
    const topLeft = this.screenToWorld(0, 0)
    const bottomRight = this.screenToWorld(canvasWidth, canvasHeight)
    return {
      x: topLeft.x,
      y: topLeft.y,
      width: bottomRight.x - topLeft.x,
      height: bottomRight.y - topLeft.y
    }
  }
}
