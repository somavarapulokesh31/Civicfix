import React, { useState, useEffect } from 'react'
import { Users, Mail, Phone, Calendar } from 'lucide-react'
import Sidebar from '../../components/Sidebar'
import LoadingSpinner from '../../components/LoadingSpinner'
import { adminAPI } from '../../services/api'
import { formatDate } from '../../utils/helpers'

export default function AdminUsers() {
  const [users, setUsers]     = useState([])
  const [loading, setLoading] = useState(true)
  const [search, setSearch]   = useState('')

  useEffect(() => {
    adminAPI.getUsers().then(d => { setUsers(d); setLoading(false) })
  }, [])

  const filtered = users.filter(u =>
    u.name.toLowerCase().includes(search.toLowerCase()) ||
    u.email.toLowerCase().includes(search.toLowerCase())
  )

  return (
    <div className="app-layout"><Sidebar />
      <main className="main-content"><div className="page-content">
        <div className="page-header">
          <div>
            <h1>Citizens</h1>
            <p>{filtered.length} registered users</p>
          </div>
        </div>

        <div className="filter-bar">
          <input
            className="search-input"
            style={{ paddingLeft: 13 }}
            placeholder="Search by name or email…"
            value={search}
            onChange={e => setSearch(e.target.value)}
          />
        </div>

        <div className="card" style={{ padding: 0 }}>
          {loading ? <LoadingSpinner /> : filtered.length === 0 ? (
            <div className="empty-state">
              <Users size={48} style={{ margin: '0 auto 12px', display: 'block', color: 'var(--gray-300)' }} />
              <h3>No users found</h3>
            </div>
          ) : (
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>Name</th>
                    <th>Email</th>
                    <th>Phone</th>
                    <th>Role</th>
                    <th>Joined</th>
                  </tr>
                </thead>
                <tbody>
                  {filtered.map(u => (
                    <tr key={u.id}>
                      <td style={{ color: 'var(--gray-400)', fontSize: 12 }}>#{u.id}</td>
                      <td style={{ fontWeight: 500 }}>{u.name}</td>
                      <td style={{ fontSize: 12, color: 'var(--blue)' }}>{u.email}</td>
                      <td style={{ fontSize: 12 }}>{u.phone || '—'}</td>
                      <td>
                        <span className="badge badge-closed" style={{ textTransform: 'capitalize' }}>
                          {u.role}
                        </span>
                      </td>
                      <td style={{ fontSize: 12 }}>{formatDate(u.created_at)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div></main>
    </div>
  )
}
