import React from 'react'
import { formatDateTime } from '../utils/helpers'

const STEPS = ['Open', 'In Progress', 'Closed']

export default function ComplaintTimeline({ history = [], currentStatus }) {
  return (
    <div className="timeline">
      {history.length > 0 ? (
        history.map((h, i) => (
          <div key={i} className="tl-item">
            <div className={`tl-dot ${h.status === currentStatus ? '' : 'pending'}`}>
              {i + 1}
            </div>
            <div className="tl-content">
              <strong>{h.status}</strong>
              <small>{h.note} · {formatDateTime(h.created_at)} · {h.updated_by}</small>
            </div>
          </div>
        ))
      ) : (
        STEPS.map((step, i) => {
          const stepIndex = STEPS.indexOf(currentStatus)
          const done = i <= stepIndex
          return (
            <div key={step} className="tl-item">
              <div className={`tl-dot ${done ? '' : 'pending'}`}>{i + 1}</div>
              <div className="tl-content">
                <strong>{step}</strong>
                <small>{done ? 'Completed' : 'Pending'}</small>
              </div>
            </div>
          )
        })
      )}
    </div>
  )
}
