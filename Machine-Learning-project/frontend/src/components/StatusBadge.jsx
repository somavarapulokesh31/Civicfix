import React from 'react'
import { statusBadgeClass } from '../utils/helpers'

export default function StatusBadge({ status }) {
  return <span className={`badge ${statusBadgeClass(status)}`}>{status}</span>
}
