import { Toolbar } from './components/Toolbar'
import { GraphCanvas } from './components/GraphCanvas'
import { SidePanel } from './components/SidePanel'
import { SearchBar } from './components/SearchBar'
import { StatusBar } from './components/StatusBar'

export default function App() {
  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      height: '100vh',
      width: '100vw',
      overflow: 'hidden',
      background: '#1a1a2e'
    }}>
      <Toolbar />
      <div style={{
        display: 'flex',
        flex: 1,
        overflow: 'hidden',
        position: 'relative'
      }}>
        <div style={{ flex: 1, position: 'relative' }}>
          <GraphCanvas />
          <SearchBar />
        </div>
        <SidePanel />
      </div>
      <StatusBar />
    </div>
  )
}
