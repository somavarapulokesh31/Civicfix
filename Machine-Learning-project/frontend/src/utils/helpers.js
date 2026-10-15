export function formatDate(iso) {
  if (!iso) return '—'
  return new Date(iso).toLocaleDateString('en-US', { year:'numeric', month:'short', day:'numeric' })
}

export function formatDateTime(iso) {
  if (!iso) return '—'
  return new Date(iso).toLocaleString('en-US', { month:'short', day:'numeric',
    hour:'2-digit', minute:'2-digit' })
}

export function statusBadgeClass(status) {
  const map = { 'Open':'badge-open', 'In Progress':'badge-in-progress',
                'Closed':'badge-closed', 'Rejected':'badge-rejected' }
  return map[status] || 'badge-open'
}

export function priorityBadgeClass(priority) {
  const map = { 'Critical':'badge-critical', 'High':'badge-high',
                'Medium':'badge-medium', 'Low':'badge-low' }
  return map[priority] || 'badge-medium'
}

export function truncate(str, n = 60) {
  return str && str.length > n ? str.slice(0, n) + '…' : str
}

export function getInitials(name) {
  return name ? name.split(' ').map(n => n[0]).join('').toUpperCase().slice(0,2) : 'CF'
}
