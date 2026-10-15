import React, { useState, useCallback } from 'react'

let _addToast = null

export function useToast() {
  const success = msg => _addToast && _addToast(msg, 'success')
  const error   = msg => _addToast && _addToast(msg, 'error')
  const info    = msg => _addToast && _addToast(msg, 'info')
  return { success, error, info }
}

export default function ToastContainer() {
  const [toasts, setToasts] = useState([])

  _addToast = useCallback((msg, type = 'success') => {
    const id = Date.now()
    setToasts(prev => [...prev, { id, msg, type }])
    setTimeout(() => setToasts(prev => prev.filter(t => t.id !== id)), 3500)
  }, [])

  return (
    <div className="toast-container">
      {toasts.map(t => (
        <div key={t.id} className={`toast ${t.type}`}>
          <span>{t.msg}</span>
        </div>
      ))}
    </div>
  )
}
