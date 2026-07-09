import { useState, useRef, useEffect, useCallback } from 'react'

// ── API helpers ──────────────────────────────────────────────────────────────
async function apiFetch(path, opts = {}) {
  const res = await fetch(path, opts)
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Unknown error' }))
    throw new Error(err.detail || 'Request failed')
  }
  return res.json()
}

// ── Sidebar ──────────────────────────────────────────────────────────────────
function Sidebar({ docs, selected, onToggle, onUpload, onDelete, uploading }) {
  const inputRef = useRef()

  function handleFiles(e) {
    const files = Array.from(e.target.files)
    if (files.length) onUpload(files)
    e.target.value = ''
  }

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <h2>Documents</h2>
        <button className="upload-btn" onClick={() => inputRef.current.click()} disabled={uploading}>
          {uploading ? 'Uploading…' : '+ Upload'}
        </button>
        <input
          ref={inputRef}
          type="file"
          multiple
          accept=".md,.txt,.pdf,.docx"
          style={{ display: 'none' }}
          onChange={handleFiles}
        />
      </div>

      <p className="sidebar-hint">
        {docs.length === 0
          ? 'No documents yet. Upload files to get started.'
          : 'Select files to search only those, or leave all unchecked to search everything.'}
      </p>

      <ul className="doc-list">
        {docs.map(doc => (
          <li key={doc.file} className={`doc-item ${selected.includes(doc.file) ? 'active' : ''}`}>
            <label className="doc-label">
              <input
                type="checkbox"
                checked={selected.includes(doc.file)}
                onChange={() => onToggle(doc.file)}
              />
              <span className="doc-name" title={doc.file}>
                {doc.file.length > 24 ? doc.file.slice(0, 22) + '…' : doc.file}
              </span>
              <span className="doc-chunks">{doc.chunks} chunks</span>
            </label>
            <button className="delete-btn" onClick={() => onDelete(doc.file)} title="Remove">✕</button>
          </li>
        ))}
      </ul>

      <p className="sidebar-footer">
        Supported: .md · .txt · .pdf · .docx
      </p>
    </aside>
  )
}

// ── Message ──────────────────────────────────────────────────────────────────
function Message({ msg }) {
  const isUser = msg.role === 'user'
  return (
    <div className={`message ${isUser ? 'user' : 'assistant'}`}>
      <div className="bubble">
        <p style={{ whiteSpace: 'pre-wrap' }}>{msg.content}</p>
        {msg.sources?.length > 0 && (
          <details className="sources">
            <summary>Sources ({msg.sources.length})</summary>
            {msg.sources.map((src, i) => (
              <div key={i} className="source-item">
                <span className="source-file">📄 {src.file}</span>
                <p className="source-text">{src.text.slice(0, 160)}…</p>
              </div>
            ))}
          </details>
        )}
      </div>
    </div>
  )
}

function TypingIndicator() {
  return (
    <div className="message assistant">
      <div className="bubble typing"><span /><span /><span /></div>
    </div>
  )
}

// ── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const [docs, setDocs] = useState([])
  const [selected, setSelected] = useState([])        // selected file names
  const [messages, setMessages] = useState([
    { role: 'assistant', content: 'Hi! Upload your notes on the left, then ask me anything.' }
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState('')
  const bottomRef = useRef()

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages, loading])

  const loadDocs = useCallback(async () => {
    try {
      const data = await apiFetch('/documents')
      setDocs(data.documents)
    } catch { /* ignore */ }
  }, [])

  useEffect(() => { loadDocs() }, [loadDocs])

  function toggleFile(file) {
    setSelected(prev => prev.includes(file) ? prev.filter(f => f !== file) : [...prev, file])
  }

  async function handleUpload(files) {
    setUploading(true)
    setError('')
    const form = new FormData()
    files.forEach(f => form.append('files', f))
    try {
      const data = await apiFetch('/upload', { method: 'POST', body: form })
      await loadDocs()
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `✅ Uploaded ${data.uploaded.map(u => u.file).join(', ')} — ${data.total_chunks} total chunks indexed.`
      }])
    } catch (e) {
      setError(e.message)
    } finally {
      setUploading(false)
    }
  }

  async function handleDelete(file) {
    try {
      await apiFetch(`/documents/${encodeURIComponent(file)}`, { method: 'DELETE' })
      setSelected(prev => prev.filter(f => f !== file))
      await loadDocs()
    } catch (e) {
      setError(e.message)
    }
  }

  async function sendMessage(e) {
    e.preventDefault()
    const question = input.trim()
    if (!question || loading) return

    setMessages(prev => [...prev, { role: 'user', content: question }])
    setInput('')
    setLoading(true)
    setError('')

    try {
      const data = await apiFetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, selected_files: selected }),
      })
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: data.answer,
        sources: data.sources,
      }])
    } catch (e) {
      setMessages(prev => [...prev, { role: 'assistant', content: `Error: ${e.message}` }])
    } finally {
      setLoading(false)
    }
  }

  const placeholder = selected.length
    ? `Ask about ${selected.length === 1 ? selected[0] : `${selected.length} selected files`}…`
    : 'Ask about all your notes…'

  return (
    <div className="app">
      <Sidebar
        docs={docs}
        selected={selected}
        onToggle={toggleFile}
        onUpload={handleUpload}
        onDelete={handleDelete}
        uploading={uploading}
      />

      <div className="main">
        <header>
          <div>
            <h1>AI Chat with My Notes</h1>
            <p>Powered by RAG — retrieves from your documents, not the internet</p>
          </div>
          {selected.length > 0 && (
            <div className="filter-chips">
              <span className="filter-label">Searching:</span>
              {selected.map(f => (
                <span key={f} className="chip">
                  {f.length > 20 ? f.slice(0, 18) + '…' : f}
                  <button onClick={() => toggleFile(f)}>✕</button>
                </span>
              ))}
            </div>
          )}
        </header>

        {error && <div className="error-banner">{error} <button onClick={() => setError('')}>✕</button></div>}

        <div className="chat-window">
          {messages.map((msg, i) => <Message key={i} msg={msg} />)}
          {loading && <TypingIndicator />}
          <div ref={bottomRef} />
        </div>

        <form className="input-bar" onSubmit={sendMessage}>
          <input
            value={input}
            onChange={e => setInput(e.target.value)}
            placeholder={placeholder}
            disabled={loading}
            autoFocus
          />
          <button type="submit" disabled={loading || !input.trim()}>
            {loading ? '…' : 'Send'}
          </button>
        </form>
      </div>
    </div>
  )
}
