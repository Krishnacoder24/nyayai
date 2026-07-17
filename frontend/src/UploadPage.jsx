import { useRef, useState } from 'react'

export default function UploadPage({ onFileSelected, status, error }) {
  const inputRef = useRef(null)
  const [isDragOver, setIsDragOver] = useState(false)

  const busy = status === 'PENDING' || status === 'STARTED'

  function handleFiles(fileList) {
    const file = fileList?.[0]
    if (file && file.type === 'application/pdf') {
      onFileSelected(file)
    }
  }

  return (
    <div className="upload-page">
      <div className="upload-page-intro">
        <span className="upload-eyebrow">NyayAI</span>
        <h1>Check a document before it's filed.</h1>
        <p>
          Upload an FIR, contract, or court notice. NyayAI checks spelling, grammar, citation
          accuracy, and name consistency, entirely on this machine.
        </p>
      </div>

      <div
        className={`upload-dropzone${isDragOver ? ' upload-dropzone--active' : ''}${busy ? ' upload-dropzone--busy' : ''}`}
        onDragOver={(e) => {
          e.preventDefault()
          setIsDragOver(true)
        }}
        onDragLeave={() => setIsDragOver(false)}
        onDrop={(e) => {
          e.preventDefault()
          setIsDragOver(false)
          if (!busy) handleFiles(e.dataTransfer.files)
        }}
        onClick={() => !busy && inputRef.current?.click()}
        role="button"
        tabIndex={0}
      >
        <input
          ref={inputRef}
          type="file"
          accept="application/pdf"
          hidden
          onChange={(e) => handleFiles(e.target.files)}
        />
        {busy ? (
          <>
            <div className="upload-spinner" aria-hidden="true" />
            <p>{status === 'PENDING' ? 'Queued...' : 'Checking the document...'}</p>
          </>
        ) : (
          <>
            <p className="upload-dropzone-title">Drop a PDF here, or click to choose one</p>
            <p className="upload-dropzone-hint">PDF only, processed locally</p>
          </>
        )}
      </div>

      {error && (
        <p className="upload-error" role="alert">
          {error}
        </p>
      )}
    </div>
  )
}