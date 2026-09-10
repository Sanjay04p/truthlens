import { useRef, useState } from 'react'
import { ArrowUpRight, BarChart3, Check, CircleAlert, Clock3, FileText, ImagePlus, LoaderCircle, RefreshCw, ScanSearch, Shield, Sparkles, Upload, X } from 'lucide-react'
import type { VerificationHistoryItem, VerificationResult } from '../types'
import { isTextResult } from '../types'

type VerificationWorkspaceProps = {
  history: VerificationHistoryItem[]
  result: VerificationResult | null
  isLoading: boolean
  error: string | null
  onVerifyText: (text: string) => Promise<VerificationResult | null>
  onVerifyImage: (file: File) => Promise<VerificationResult | null>
  onSelectHistory: (result: VerificationResult) => void
  onClearResult: () => void
  onCompleted: (item: VerificationHistoryItem) => void
}

type Mode = 'text' | 'image'

const examples = ['The Earth is flat', 'Water is made of H2O', '5G causes illness']

export function VerificationWorkspace({ history, result, isLoading, error, onVerifyText, onVerifyImage, onSelectHistory, onClearResult, onCompleted }: VerificationWorkspaceProps) {
  const [mode, setMode] = useState<Mode>('text')
  const [claim, setClaim] = useState('')
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleTextSubmit = async () => {
    if (!claim.trim() || isLoading) return
    const verified = await onVerifyText(claim.trim())
    if (verified) onCompleted({ id: crypto.randomUUID(), title: claim.trim(), label: verified.label, kind: 'text', result: verified, createdAt: new Date().toISOString() })
  }

  const handleImageSubmit = async () => {
    if (!selectedFile || isLoading) return
    const verified = await onVerifyImage(selectedFile)
    if (verified) onCompleted({ id: crypto.randomUUID(), title: selectedFile.name, label: verified.label, kind: 'image', result: verified, createdAt: new Date().toISOString() })
  }

  const chooseFile = (file?: File) => {
    if (!file || !file.type.startsWith('image/')) return
    setSelectedFile(file)
    setPreviewUrl(URL.createObjectURL(file))
    onClearResult()
  }

  const clearFile = () => { setSelectedFile(null); if (previewUrl) URL.revokeObjectURL(previewUrl); setPreviewUrl(null); if (fileInputRef.current) fileInputRef.current.value = '' }

  return (
    <div className="verification-page">
      <section className="page-intro">
        <div><p className="eyebrow">TruthLens / verification desk</p><h1>See what the evidence says.</h1><p className="intro-copy">Check a claim against live sources or inspect an image for synthetic signals.</p></div>
        <div className="intro-mark"><ScanSearch size={26} /><span>Multimodal<br />analysis</span></div>
      </section>

      <div className="workspace-grid">
        <section className="input-panel panel" aria-labelledby="input-panel-title">
          <div className="panel-header"><div><span className="panel-kicker">01 / Input</span><h2 id="input-panel-title">What should we verify?</h2></div><span className="secure-note"><Shield size={14} /> Session-scoped history</span></div>
          <div className="mode-tabs" role="tablist" aria-label="Verification type">
            <button role="tab" aria-selected={mode === 'text'} className={`mode-tab ${mode === 'text' ? 'mode-tab-active' : ''}`} onClick={() => setMode('text')}><FileText size={16} /> Claim text</button>
            <button role="tab" aria-selected={mode === 'image'} className={`mode-tab ${mode === 'image' ? 'mode-tab-active' : ''}`} onClick={() => setMode('image')}><ImagePlus size={16} /> Image file</button>
          </div>

          {mode === 'text' ? <>
            <label className="field-label" htmlFor="claim-input">Paste or write a factual claim</label>
            <textarea id="claim-input" value={claim} onChange={(event) => { setClaim(event.target.value); if (result) onClearResult() }} placeholder="e.g. The Earth is flat and space images are fake..." maxLength={1200} />
            <div className="field-meta"><span>{claim.length} / 1,200 characters</span><span className="input-hint">Enter a complete claim for a stronger result</span></div>
            <div className="examples"><span className="examples-label">Try an example</span>{examples.map((example) => <button key={example} onClick={() => setClaim(example)}>{example}</button>)}</div>
            <button className="primary-button" disabled={!claim.trim() || isLoading} onClick={handleTextSubmit}>{isLoading ? <LoaderCircle className="spin" size={17} /> : <Sparkles size={17} />} {isLoading ? 'Analyzing evidence...' : 'Verify claim'}<ArrowUpRight size={17} /></button>
          </> : <>
            <label className="field-label" htmlFor="image-input">Upload an image to inspect</label>
            {!previewUrl ? <button className="dropzone" onClick={() => fileInputRef.current?.click()} onDragOver={(event) => event.preventDefault()} onDrop={(event) => { event.preventDefault(); chooseFile(event.dataTransfer.files[0]) }}><span className="upload-icon"><Upload size={20} /></span><strong>Drop an image here</strong><span>or browse from your device</span><small>JPG, PNG or WebP - up to 10 MB</small></button> : <div className="image-preview"><img src={previewUrl} alt="Selected content" /><div className="image-preview-meta"><div><strong>{selectedFile?.name}</strong><span>{selectedFile ? `${(selectedFile.size / 1024 / 1024).toFixed(2)} MB` : ''}</span></div><button className="icon-button small" aria-label="Remove selected image" onClick={clearFile}><X size={16} /></button></div></div>}
            <input ref={fileInputRef} id="image-input" className="visually-hidden" type="file" accept="image/jpeg,image/png,image/webp" onChange={(event) => chooseFile(event.target.files?.[0])} />
            <div className="image-note"><CircleAlert size={15} /><span>Image verification looks for synthetic generation signals. It is not a determination of who created the image.</span></div>
            <button className="primary-button" disabled={!selectedFile || isLoading} onClick={handleImageSubmit}>{isLoading ? <LoaderCircle className="spin" size={17} /> : <ScanSearch size={17} />} {isLoading ? 'Inspecting image...' : 'Inspect image'}<ArrowUpRight size={17} /></button>
          </>}
          {error && <div className="error-banner" role="alert"><CircleAlert size={17} /><span>{error}</span><button onClick={onClearResult} aria-label="Dismiss error"><X size={15} /></button></div>}
        </section>

        <ResultPanel result={result} isLoading={isLoading} onClear={onClearResult} />
      </div>

      <section className="history-section"><div className="section-heading"><div><span className="panel-kicker">Recent checks</span><h2>Session history</h2></div>{history.length > 0 && <span className="history-count">{history.length} {history.length === 1 ? 'check' : 'checks'}</span>}</div>{history.length === 0 ? <div className="empty-history"><Clock3 size={18} /><span>Your completed verifications will appear here.</span></div> : <div className="history-list">{history.slice(0, 4).map((item) => <button key={item.id} className="history-row" onClick={() => onSelectHistory(item.result)}><span className={`history-dot history-dot-${item.label.toLowerCase()}`} /> <span className="history-title">{item.title}</span><span className="history-type">{item.kind === 'text' ? 'Claim' : 'Image'}</span><span className={`history-label history-label-${item.label.toLowerCase()}`}>{item.label}</span><span className="history-time">{formatTime(item.createdAt)}</span><ArrowUpRight size={15} /></button>)}</div>}</section>
    </div>
  )
}

function ResultPanel({ result, isLoading, onClear }: { result: VerificationResult | null; isLoading: boolean; onClear: () => void }) {
  if (isLoading) return <section className="result-panel panel result-loading"><div className="loading-orbit"><LoaderCircle size={33} /></div><span className="panel-kicker">02 / Analysis</span><h2>Reading the signals...</h2><p>TruthLens is comparing the input with its available evidence. This can take a moment.</p><div className="loading-lines"><i /><i /><i /></div></section>
  if (!result) return <section className="result-panel panel result-empty"><div className="empty-shield"><Shield size={27} /></div><span className="panel-kicker">02 / Analysis</span><h2>Your result will land here.</h2><p>Start with a claim or an image. The result will include a confidence score, signal breakdown, and source context.</p><div className="result-steps"><span><i>1</i> Submit input</span><span><i>2</i> Analyze signals</span><span><i>3</i> Review evidence</span></div></section>

  const score = parseScore(result.confidence_score)
  const isText = isTextResult(result)
  const breakdown = result.breakdown || {}
  const breakdownEntries = Object.entries(breakdown)
  const labelStyle = result.label === 'Error' ? 'unverified' : result.label.toLowerCase()
  return <section className="result-panel panel" aria-live="polite">
    <div className="panel-header result-header"><div><span className="panel-kicker">02 / Analysis complete</span><h2>Verification result</h2></div><button className="text-button" onClick={onClear}><RefreshCw size={14} /> New check</button></div>
    <div className="verdict-row"><div className={`verdict-badge verdict-${labelStyle}`}><span className="verdict-icon">{result.label === 'Real' ? <Check size={19} /> : result.label === 'Fake' ? <X size={19} /> : <CircleAlert size={19} />}</span><span>{result.label}</span></div><div className="confidence-wrap"><div className={`confidence-ring ring-${labelStyle}`} style={{ '--score': `${score * 3.6}deg` } as React.CSSProperties}><div><strong>{score}%</strong><span>confidence</span></div></div></div></div>
    <div className="reason-block"><span className="reason-label">Assessment</span><p>{result.reason || 'No additional assessment was returned.'}</p></div>
    <div className="signal-block"><div className="signal-heading"><span>Signal breakdown</span><BarChart3 size={15} /></div>{breakdownEntries.length ? breakdownEntries.map(([key, value]) => <div className="signal-row" key={key}><div><span>{prettyKey(key)}</span><strong>{value}%</strong></div><div className="signal-track"><i className={`signal-fill fill-${labelStyle}`} style={{ width: `${Math.min(100, Math.max(0, Number(value) || 0))}%` }} /></div></div>) : <div className="no-signal-data">The engine did not return a detailed signal breakdown for this check.</div>}</div>
    <div className="result-footer"><span>{isText ? `Tier ${result.tier} - ${verificationType(result.verification_type)}` : result.provider_used || 'Image analysis'}</span>{isText && result.sources && result.sources.length > 0 && <span>{result.sources.length} {result.sources.length === 1 ? 'source' : 'sources'} reviewed</span>}</div>
    {isText && result.sources && result.sources.length > 0 && <div className="sources-block"><div className="signal-heading"><span>Evidence sources</span><span className="source-count">{result.sources.length}</span></div>{result.sources.slice(0, 3).map((source) => <a key={`${source.name}-${source.url}`} className="source-row" href={source.url || '#'} target="_blank" rel="noreferrer"><span className="source-favicon">{source.name.slice(0, 1).toUpperCase()}</span><span><strong>{source.name}</strong><small>{source.domain || source.role || 'Reference source'}</small></span><ArrowUpRight size={14} /></a>)}</div>}
  </section>
}

function parseScore(value?: string) { const parsed = Number.parseFloat(value || '0'); return Number.isFinite(parsed) ? Math.round(parsed) : 0 }
function prettyKey(value: string) { return value.replace(/_/g, ' ').replace(/\b\w/g, (letter) => letter.toUpperCase()) }
function verificationType(value?: string) { return value?.replace(/_/g, ' ') || 'Live verification' }
function formatTime(value: string) { const date = new Date(value); return date.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' }) }
