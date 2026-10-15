import React from 'react'

function PredItem({ label, value, confidence }) {
  return (
    <div className="pred-item">
      <div className="pred-label">{label}</div>
      <div className="pred-value">{value}</div>
      <div className="pred-conf">Confidence: {confidence}%</div>
      <div className="conf-bar">
        <div className="conf-fill" style={{ width: `${confidence}%` }}></div>
      </div>
    </div>
  )
}

export default function PredictionCard({ prediction }) {
  if (!prediction) return null
  return (
    <div className="prediction-result">
      <h3 style={{ color: 'var(--green)', fontSize: 15, marginBottom: 4 }}>
        ✓ AI Analysis Complete
      </h3>
      <p style={{ fontSize: 12.5, color: 'var(--gray-500)' }}>
        The ML model has analyzed your complaint and predicted the following:
      </p>
      <div className="prediction-grid">
        <PredItem label="Issue Category"      value={prediction.predicted_category}
                  confidence={prediction.category_confidence} />
        <PredItem label="Priority Level"      value={prediction.predicted_priority}
                  confidence={prediction.priority_confidence} />
        <PredItem label="Responsible Dept."   value={prediction.predicted_department}
                  confidence={prediction.department_confidence} />
      </div>
    </div>
  )
}
