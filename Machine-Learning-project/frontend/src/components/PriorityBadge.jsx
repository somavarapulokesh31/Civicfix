import React from 'react'
import { priorityBadgeClass } from '../utils/helpers'

export default function PriorityBadge({ priority }) {
  return <span className={`badge ${priorityBadgeClass(priority)}`}>{priority}</span>
}
