import { useMemo, useState } from 'react'
import { Activity, BarChart3, CheckCircle2, FileSearch, Settings2, ShieldCheck } from 'lucide-react'
import { AppShell } from '../components/layout/AppShell'
import { VerificationWorkspace } from '../features/verification/components/VerificationWorkspace'
import { useVerification } from '../features/verification/hooks/useVerification'
import type { VerificationHistoryItem } from '../features/verification/types'

const initialHistory: VerificationHistoryItem[] = []

function App() {
  const [history, setHistory] = useState<VerificationHistoryItem[]>(initialHistory)
  const [activeView, setActiveView] = useState('verify')
  const verification = useVerification()

  const stats = useMemo(() => {
    const completed = history.length
    const confirmed = history.filter((item) => item.label === 'Real').length
    const flagged = history.filter((item) => item.label === 'Fake').length
    return { completed, confirmed, flagged }
  }, [history])

  const handleCompleted = (item: VerificationHistoryItem) => {
    setHistory((current) => [item, ...current.filter((entry) => entry.id !== item.id)].slice(0, 8))
  }

  return (
    <AppShell
      activeView={activeView}
      onViewChange={setActiveView}
      stats={stats}
      navItems={[
        { id: 'verify', label: 'Verify content', icon: FileSearch },
        { id: 'activity', label: 'Activity', icon: Activity, count: history.length || undefined },
        { id: 'insights', label: 'Insights', icon: BarChart3 },
      ]}
      utilityItems={[{ id: 'settings', label: 'Settings', icon: Settings2 }]}
    >
      {activeView === 'verify' ? (
        <VerificationWorkspace
          history={history}
          result={verification.result}
          isLoading={verification.isLoading}
          error={verification.error}
          onVerifyText={verification.verifyText}
          onVerifyImage={verification.verifyImage}
          onSelectHistory={verification.selectResult}
          onClearResult={verification.clearResult}
          onCompleted={handleCompleted}
        />
      ) : (
        <section className="placeholder-view" aria-labelledby="placeholder-title">
          <div className="placeholder-icon"><ShieldCheck size={24} /></div>
          <p className="eyebrow">TruthLens workspace</p>
          <h1 id="placeholder-title">{activeView === 'activity' ? 'Verification activity' : 'Insights are coming next'}</h1>
          <p className="placeholder-copy">
            {activeView === 'activity'
              ? 'Your recent checks will appear here as you verify claims and images.'
              : 'Keep verifying content to build a useful picture of the signals in your session.'}
          </p>
          {activeView === 'activity' && history.length > 0 && (
            <div className="placeholder-list">
              {history.map((item) => (
                <button key={item.id} className="placeholder-list-item" onClick={() => { verification.selectResult(item.result); setActiveView('verify') }}>
                  <span className={`mini-status mini-status-${item.label.toLowerCase()}`}><CheckCircle2 size={14} /></span>
                  <span>{item.title}</span>
                  <span className="muted-text">{item.label}</span>
                </button>
              ))}
            </div>
          )}
        </section>
      )}
    </AppShell>
  )
}

export default App
