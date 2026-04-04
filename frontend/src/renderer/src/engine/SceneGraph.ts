import { VisualNode, VisualEdge, Rect } from './types'

export class SceneGraph {
  nodes = new Map<string, VisualNode>()
  edges: VisualEdge[] = []

  update(nodes: Map<string, VisualNode>, edges: VisualEdge[]): void {
    this.nodes = nodes
    this.edges = edges
  }

  getVisibleNodes(viewport: Rect): VisualNode[] {
    const visible: VisualNode[] = []
    for (const node of this.nodes.values()) {
      if (this.rectsOverlap(node.rect, viewport)) {
        visible.push(node)
      }
    }
    return visible.sort((a, b) => a.depth - b.depth)
  }

  getVisibleEdges(viewport: Rect): VisualEdge[] {
    return this.edges.filter(edge => {
      // Include edge if any of its points are in the viewport
      return edge.points.some(p =>
        p.x >= viewport.x - 100 &&
        p.x <= viewport.x + viewport.width + 100 &&
        p.y >= viewport.y - 100 &&
        p.y <= viewport.y + viewport.height + 100
      )
    })
  }

  getBoundingRect(): Rect | null {
    if (this.nodes.size === 0) return null
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity
    for (const node of this.nodes.values()) {
      minX = Math.min(minX, node.rect.x)
      minY = Math.min(minY, node.rect.y)
      maxX = Math.max(maxX, node.rect.x + node.rect.width)
      maxY = Math.max(maxY, node.rect.y + node.rect.height)
    }
    return { x: minX, y: minY, width: maxX - minX, height: maxY - minY }
  }

  private rectsOverlap(a: Rect, b: Rect): boolean {
    return !(
      a.x + a.width < b.x ||
      b.x + b.width < a.x ||
      a.y + a.height < b.y ||
      b.y + b.height < a.y
    )
  }
}
