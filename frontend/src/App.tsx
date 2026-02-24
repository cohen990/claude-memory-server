import { useState } from 'react'
import Nav, { type Tab } from './components/Nav'
import GraphTab from './components/graph/GraphTab'
import RecallsTab from './components/recalls/RecallsTab'
import StatsTab from './components/stats/StatsTab'

export default function App() {
  const [tab, setTab] = useState<Tab>('graph')

  return (
    <>
      <Nav active={tab} onChange={setTab} />
      <main>
        {/* Graph stays mounted — expensive Cytoscape layout */}
        <div style={{ display: tab === 'graph' ? 'flex' : 'none', flexDirection: 'column', height: '100%' }}>
          <GraphTab />
        </div>
        {tab === 'recalls' && <RecallsTab />}
        {tab === 'stats' && <StatsTab />}
      </main>
    </>
  )
}
