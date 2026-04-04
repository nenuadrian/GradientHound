import { VisualNode, Rect } from './types'
import { Camera } from './Camera'
import { LAYER_HEADER_HEIGHT } from './constants'

export interface HitResult {
  nodeId: string
  isHeader: boolean  // true if hit was on a layer's header bar
}

export class HitTester {
  test(
    screenX: number,
    screenY: number,
    camera: Camera,
    nodes: Map<string, VisualNode>
  ): HitResult | null {
    const world = camera.screenToWorld(screenX, screenY)

    // Sort nodes by depth descending (topmost/deepest first)
    const sorted = Array.from(nodes.values()).sort((a, b) => b.depth - a.depth)

    for (const node of sorted) {
      if (this.pointInRect(world.x, world.y, node.rect)) {
        const isHeader = !node.isLeaf && this.pointInHeader(world.x, world.y, node.rect)
        return { nodeId: node.id, isHeader }
      }
    }

    return null
  }

  private pointInRect(x: number, y: number, rect: Rect): boolean {
    return (
      x >= rect.x &&
      x <= rect.x + rect.width &&
      y >= rect.y &&
      y <= rect.y + rect.height
    )
  }

  private pointInHeader(x: number, y: number, rect: Rect): boolean {
    return (
      x >= rect.x &&
      x <= rect.x + rect.width &&
      y >= rect.y &&
      y <= rect.y + LAYER_HEADER_HEIGHT
    )
  }
}
