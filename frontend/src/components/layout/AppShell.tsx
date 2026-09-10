import type { ReactNode } from 'react'
import { Bell, ChevronRight, CircleHelp, ShieldCheck } from 'lucide-react'
import type { LucideIcon } from 'lucide-react'

type NavItem = { id: string; label: string; icon: LucideIcon; count?: number }

type AppShellProps = {
  children: ReactNode
  activeView: string
  onViewChange: (view: string) => void
  navItems: NavItem[]
  utilityItems: NavItem[]
  stats: { completed: number; confirmed: number; flagged: number }
}

export function AppShell({ children, activeView, onViewChange, navItems, utilityItems, stats }: AppShellProps) {
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-lockup">
          <div className="brand-mark"><ShieldCheck size={21} strokeWidth={2.25} /></div>
          <div>
            <span className="brand-name">TruthLens</span>
            <span className="brand-caption">Signal intelligence</span>
          </div>
        </div>

        <div className="sidebar-section">
          <span className="sidebar-label">Workspace</span>
          <nav className="sidebar-nav" aria-label="Workspace navigation">
            {navItems.map((item) => {
              const Icon = item.icon
              return (
                <button key={item.id} className={`nav-item ${activeView === item.id ? 'nav-item-active' : ''}`} onClick={() => onViewChange(item.id)}>
                  <Icon size={18} strokeWidth={activeView === item.id ? 2.3 : 1.8} />
                  <span>{item.label}</span>
                  {item.count ? <span className="nav-count">{item.count}</span> : <ChevronRight className="nav-arrow" size={15} />}
                </button>
              )
            })}
          </nav>
        </div>

        <div className="sidebar-rule" />
        <div className="sidebar-section sidebar-section-lower">
          <span className="sidebar-label">Account</span>
          <nav className="sidebar-nav" aria-label="Account navigation">
            {utilityItems.map((item) => {
              const Icon = item.icon
              return <button key={item.id} className={`nav-item ${activeView === item.id ? 'nav-item-active' : ''}`} onClick={() => onViewChange(item.id)}><Icon size={18} strokeWidth={1.8} /><span>{item.label}</span><ChevronRight className="nav-arrow" size={15} /></button>
            })}
          </nav>
        </div>

        <div className="sidebar-bottom">
          <div className="mini-summary">
            <div className="mini-summary-heading"><span>Session overview</span><CircleHelp size={14} /></div>
            <div className="summary-number">{stats.completed.toString().padStart(2, '0')}</div>
            <div className="summary-caption">checks completed</div>
            <div className="summary-split"><span><i className="dot dot-real" /> {stats.confirmed} real</span><span><i className="dot dot-fake" /> {stats.flagged} fake</span></div>
          </div>
          <div className="sidebar-user"><div className="avatar">SL</div><div><strong>Session user</strong><span>Local workspace</span></div><Bell size={16} className="user-bell" /></div>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div className="breadcrumb"><span>Workspace</span><ChevronRight size={14} /><strong>{activeView === 'verify' ? 'Verify content' : activeView[0].toUpperCase() + activeView.slice(1)}</strong></div>
          <div className="topbar-actions"><span className="api-status"><span className="status-pulse" /> Ready to verify</span><button className="icon-button" aria-label="Notifications"><Bell size={18} /></button></div>
        </header>
        <div className="content-wrap">{children}</div>
      </main>
    </div>
  )
}
