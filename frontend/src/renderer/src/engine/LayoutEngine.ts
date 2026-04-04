import ELK, { ElkNode, ElkExtendedEdge } from 'elkjs/lib/elk.bundled.js'
import { GraphNode, GraphEdge, ModelGraph } from '../types/graph'
import { Rect, VisualNode, VisualEdge } from './types'
import {
  OP_NODE_MIN_WIDTH,
  OP_NODE_HEIGHT,
  LAYER_COLLAPSED_WIDTH,
  LAYER_COLLAPSED_HEIGHT,
  LAYER_HEADER_HEIGHT,
  ELK_SPACING_NODE,
  ELK_SPACING_LAYER
} from './constants'

const elk = new ELK()

export class LayoutEngine {
  async layout(
    graph: ModelGraph,
    expandedGroups: Set<string>
  ): Promise<{ nodes: Map<string, VisualNode>; edges: VisualEdge[] }> {
    const nodeMap = new Map<string, GraphNode>()
    for (const n of graph.nodes) {
      nodeMap.set(n.id, n)
    }

    // Build ELK graph
    const elkGraph = this.buildElkGraph(graph, nodeMap, expandedGroups, graph.root_id)

    // Run layout
    const result = await elk.layout(elkGraph)

    // Extract results
    const visualNodes = new Map<string, VisualNode>()
    const visualEdges: VisualEdge[] = []

    this.extractLayout(result, nodeMap, expandedGroups, visualNodes, 0)
    this.extractEdges(result, visualEdges)

    return { nodes: visualNodes, edges: visualEdges }
  }

  private buildElkGraph(
    graph: ModelGraph,
    nodeMap: Map<string, GraphNode>,
    expandedGroups: Set<string>,
    rootId: string
  ): ElkNode {
    const rootNode = nodeMap.get(rootId)

    const elkRoot: ElkNode = {
      id: rootId,
      layoutOptions: {
        'elk.algorithm': 'layered',
        'elk.direction': 'DOWN',
        'elk.hierarchyHandling': 'INCLUDE_CHILDREN',
        'elk.spacing.nodeNode': String(ELK_SPACING_NODE),
        'elk.layered.spacing.nodeNodeBetweenLayers': String(ELK_SPACING_LAYER),
        'elk.padding': `[top=${LAYER_HEADER_HEIGHT + 20},left=20,bottom=20,right=20]`,
        'elk.layered.crossingMinimization.strategy': 'LAYER_SWEEP'
      },
      children: [],
      edges: []
    }

    // Get visible children of root
    const visibleChildren = this.getVisibleChildren(rootNode!, nodeMap, expandedGroups)

    for (const childId of visibleChildren) {
      const child = nodeMap.get(childId)
      if (!child) continue
      elkRoot.children!.push(this.buildElkNode(child, nodeMap, expandedGroups))
    }

    // Build edges between visible nodes
    const visibleNodeIds = this.collectVisibleNodeIds(rootNode!, nodeMap, expandedGroups)
    elkRoot.edges = this.buildElkEdges(graph.edges, visibleNodeIds, nodeMap, expandedGroups)

    return elkRoot
  }

  private buildElkNode(
    node: GraphNode,
    nodeMap: Map<string, GraphNode>,
    expandedGroups: Set<string>
  ): ElkNode {
    const isGroup = !node.is_leaf
    const isExpanded = isGroup && expandedGroups.has(node.id)

    if (!isGroup || !isExpanded) {
      // Leaf node or collapsed group
      const width = isGroup ? LAYER_COLLAPSED_WIDTH : Math.max(OP_NODE_MIN_WIDTH, node.name.length * 8 + 40)
      const height = isGroup ? LAYER_COLLAPSED_HEIGHT : OP_NODE_HEIGHT
      return { id: node.id, width, height }
    }

    // Expanded group
    const elkNode: ElkNode = {
      id: node.id,
      layoutOptions: {
        'elk.algorithm': 'layered',
        'elk.direction': 'DOWN',
        'elk.padding': `[top=${LAYER_HEADER_HEIGHT + 20},left=20,bottom=20,right=20]`,
        'elk.spacing.nodeNode': String(ELK_SPACING_NODE),
        'elk.layered.spacing.nodeNodeBetweenLayers': String(ELK_SPACING_LAYER)
      },
      children: [],
      edges: []
    }

    const visibleChildren = this.getVisibleChildren(node, nodeMap, expandedGroups)
    for (const childId of visibleChildren) {
      const child = nodeMap.get(childId)
      if (!child) continue
      elkNode.children!.push(this.buildElkNode(child, nodeMap, expandedGroups))
    }

    return elkNode
  }

  private getVisibleChildren(
    node: GraphNode,
    nodeMap: Map<string, GraphNode>,
    expandedGroups: Set<string>
  ): string[] {
    const result: string[] = []
    for (const childId of node.children) {
      const child = nodeMap.get(childId)
      if (!child) continue
      result.push(childId)
    }
    return result
  }

  private collectVisibleNodeIds(
    root: GraphNode,
    nodeMap: Map<string, GraphNode>,
    expandedGroups: Set<string>
  ): Set<string> {
    const result = new Set<string>()
    const collect = (node: GraphNode) => {
      result.add(node.id)
      if (!node.is_leaf && expandedGroups.has(node.id)) {
        for (const childId of node.children) {
          const child = nodeMap.get(childId)
          if (child) collect(child)
        }
      }
    }
    collect(root)
    return result
  }

  private buildElkEdges(
    edges: GraphEdge[],
    visibleNodeIds: Set<string>,
    _nodeMap: Map<string, GraphNode>,
    _expandedGroups: Set<string>
  ): ElkExtendedEdge[] {
    const elkEdges: ElkExtendedEdge[] = []
    for (const edge of edges) {
      if (visibleNodeIds.has(edge.source) && visibleNodeIds.has(edge.target)) {
        elkEdges.push({
          id: edge.id,
          sources: [edge.source],
          targets: [edge.target]
        })
      }
    }
    return elkEdges
  }

  private extractLayout(
    elkNode: ElkNode,
    nodeMap: Map<string, GraphNode>,
    expandedGroups: Set<string>,
    result: Map<string, VisualNode>,
    depth: number,
    offsetX = 0,
    offsetY = 0
  ): void {
    const x = (elkNode.x || 0) + offsetX
    const y = (elkNode.y || 0) + offsetY
    const w = elkNode.width || 0
    const h = elkNode.height || 0

    const graphNode = nodeMap.get(elkNode.id)
    if (graphNode) {
      const isGroup = !graphNode.is_leaf
      result.set(elkNode.id, {
        id: graphNode.id,
        name: graphNode.name,
        op: graphNode.op,
        isLeaf: graphNode.is_leaf,
        moduleType: graphNode.module_type,
        parentId: graphNode.parent_id,
        children: graphNode.children,
        attributes: graphNode.attributes,
        inputShapes: graphNode.input_shapes,
        outputShapes: graphNode.output_shapes,
        paramCount: graphNode.param_count,
        rect: { x, y, width: w, height: h },
        depth,
        isExpanded: isGroup && expandedGroups.has(graphNode.id)
      })
    }

    // Recurse into children
    if (elkNode.children) {
      for (const child of elkNode.children) {
        this.extractLayout(child, nodeMap, expandedGroups, result, depth + 1, x, y)
      }
    }
  }

  private extractEdges(elkNode: ElkNode, result: VisualEdge[]): void {
    if (elkNode.edges) {
      for (const edge of elkNode.edges) {
        const points: Array<{ x: number; y: number }> = []
        if (edge.sections) {
          for (const section of edge.sections) {
            points.push({ x: section.startPoint.x, y: section.startPoint.y })
            if (section.bendPoints) {
              for (const bp of section.bendPoints) {
                points.push({ x: bp.x, y: bp.y })
              }
            }
            points.push({ x: section.endPoint.x, y: section.endPoint.y })
          }
        }
        result.push({
          id: edge.id,
          source: edge.sources[0],
          target: edge.targets[0],
          points
        })
      }
    }

    // Recurse
    if (elkNode.children) {
      for (const child of elkNode.children) {
        this.extractEdges(child, result)
      }
    }
  }
}
