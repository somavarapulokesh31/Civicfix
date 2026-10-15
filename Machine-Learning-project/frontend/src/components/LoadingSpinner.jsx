import React from 'react'

export default function LoadingSpinner({ message = 'Loading...' }) {
  return (
    <div className="spinner-wrap" style={{ flexDirection: 'column', gap: 12 }}>
      <div className="spinner"></div>
      <p style={{ color: 'var(--gray-400)', fontSize: 13 }}>{message}</p>
    </div>
  )
}
