// CivicFix — Flask App JS

// Sidebar toggle
document.addEventListener('DOMContentLoaded', function () {
  const toggle = document.getElementById('sidebarToggle')
  const sidebar = document.getElementById('sidebar')
  if (toggle && sidebar) {
    toggle.addEventListener('click', () => sidebar.classList.toggle('show'))
  }

  // Auto-dismiss alerts after 4s
  document.querySelectorAll('.alert').forEach(el => {
    setTimeout(() => {
      el.style.transition = 'opacity .5s'
      el.style.opacity = '0'
      setTimeout(() => el.remove(), 500)
    }, 4000)
  })
})

// Update complaint status (admin)
function updateStatus(ticketId, newStatus) {
  if (!confirm('Change status to "' + newStatus + '"?')) return
  fetch('/admin/update_status', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ticket_id: ticketId, status: newStatus })
  })
  .then(r => r.json())
  .then(data => {
    if (data.success) {
      const badge = document.getElementById('badge-' + ticketId)
      if (badge) {
        const cls = { 'Open':'badge-open','In Progress':'badge-progress','Closed':'badge-closed' }
        badge.className = cls[newStatus] || 'badge-open'
        badge.textContent = newStatus
      }
      showToast('Status updated to: ' + newStatus)
    }
  })
  .catch(() => showToast('Update failed', 'error'))
}

// Toast
function showToast(msg, type) {
  const t = document.createElement('div')
  t.className = 'alert alert-' + (type === 'error' ? 'danger' : 'success')
  t.style.cssText = 'position:fixed;bottom:20px;right:20px;z-index:9999;min-width:250px;box-shadow:0 4px 12px rgba(0,0,0,.15)'
  t.innerHTML = msg
  document.body.appendChild(t)
  setTimeout(() => { t.style.opacity='0'; t.style.transition='opacity .5s'; setTimeout(()=>t.remove(),500) }, 3000)
}
