import { useGraphStore } from '../store/graphStore'

export function NodeDetails() {
  const graph = useGraphStore(s => s.graph)
  const selectedNodeId = useGraphStore(s => s.selectedNodeId)

  if (!graph || !selectedNodeId) {
    return (
      <div style={{ padding: '16px', color: '#888', fontSize: '13px' }}>
        Select a node to view details
      </div>
    )
  }

  const node = graph.nodes.find(n => n.id === selectedNodeId)
  if (!node) return null

  return (
    <div style={{ padding: '12px', fontSize: '13px', color: '#e0e0e0' }}>
      <h3 style={{ margin: '0 0 12px', fontSize: '15px', color: '#fff' }}>
        {node.name}
      </h3>

      <Section title="Info">
        <Row label="ID" value={node.id} />
        <Row label="Operation" value={node.op} />
        {node.module_type && <Row label="Type" value={node.module_type.split('.').pop() || ''} />}
        {node.module_type && <Row label="Full Type" value={node.module_type} mono />}
        <Row label="Parameters" value={node.param_count.toLocaleString()} />
        <Row label="Leaf" value={node.is_leaf ? 'Yes' : 'No'} />
        {node.children.length > 0 && (
          <Row label="Children" value={String(node.children.length)} />
        )}
      </Section>

      {node.input_shapes.length > 0 && (
        <Section title="Input Shapes">
          {node.input_shapes.map((shape, i) => (
            <div key={i} style={shapeStyle}>
              {Array.isArray(shape) ? `${shape[0]} ${JSON.stringify(shape[1])}` : JSON.stringify(shape)}
            </div>
          ))}
        </Section>
      )}

      {node.output_shapes.length > 0 && (
        <Section title="Output Shapes">
          {node.output_shapes.map((shape, i) => (
            <div key={i} style={shapeStyle}>
              {Array.isArray(shape) ? `${shape[0]} ${JSON.stringify(shape[1])}` : JSON.stringify(shape)}
            </div>
          ))}
        </Section>
      )}

      {Object.keys(node.attributes).length > 0 && (
        <Section title="Attributes">
          {Object.entries(node.attributes).map(([key, val]) => (
            <Row key={key} label={key} value={JSON.stringify(val)} />
          ))}
        </Section>
      )}

      {/* Connected edges */}
      <Section title="Connections">
        {(() => {
          const inputs = graph.edges.filter(e => e.target === node.id)
          const outputs = graph.edges.filter(e => e.source === node.id)
          return (
            <>
              {inputs.length > 0 && (
                <div style={{ marginBottom: '6px' }}>
                  <span style={{ color: '#4CAF50', fontSize: '11px' }}>INPUTS ({inputs.length})</span>
                  {inputs.map(e => (
                    <div key={e.id} style={{ paddingLeft: '8px', fontSize: '12px', color: '#aaa' }}>
                      {e.source}
                    </div>
                  ))}
                </div>
              )}
              {outputs.length > 0 && (
                <div>
                  <span style={{ color: '#F44336', fontSize: '11px' }}>OUTPUTS ({outputs.length})</span>
                  {outputs.map(e => (
                    <div key={e.id} style={{ paddingLeft: '8px', fontSize: '12px', color: '#aaa' }}>
                      {e.target}
                    </div>
                  ))}
                </div>
              )}
              {inputs.length === 0 && outputs.length === 0 && (
                <div style={{ color: '#666' }}>No connections</div>
              )}
            </>
          )
        })()}
      </Section>
    </div>
  )
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div style={{ marginBottom: '14px' }}>
      <div style={{
        fontSize: '11px',
        fontWeight: 'bold',
        color: '#90CAF9',
        textTransform: 'uppercase',
        marginBottom: '6px',
        letterSpacing: '0.5px'
      }}>
        {title}
      </div>
      {children}
    </div>
  )
}

function Row({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div style={{
      display: 'flex',
      justifyContent: 'space-between',
      padding: '2px 0',
      gap: '12px'
    }}>
      <span style={{ color: '#888', flexShrink: 0 }}>{label}</span>
      <span style={{
        textAlign: 'right',
        wordBreak: 'break-all',
        fontFamily: mono ? 'monospace' : 'inherit',
        fontSize: mono ? '11px' : '13px'
      }}>
        {value}
      </span>
    </div>
  )
}

const shapeStyle: React.CSSProperties = {
  fontFamily: 'monospace',
  fontSize: '12px',
  padding: '3px 8px',
  background: '#1a1a2e',
  borderRadius: '3px',
  marginBottom: '3px'
}
