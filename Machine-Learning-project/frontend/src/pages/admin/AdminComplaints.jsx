import React, { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { Search } from 'lucide-react'
import Sidebar from '../../components/Sidebar'
import StatusBadge from '../../components/StatusBadge'
import PriorityBadge from '../../components/PriorityBadge'
import LoadingSpinner from '../../components/LoadingSpinner'
import { adminAPI } from '../../services/api'
import { formatDate, truncate } from '../../utils/helpers'

export default function AdminComplaints() {
  const [complaints, setComplaints] = useState([])
  const [total, setTotal]   = useState(0)
  const [loading, setLoading] = useState(true)
  const [search, setSearch] = useState('')
  const [statusF, setStatusF] = useState('')
  const [priorityF, setPriorityF] = useState('')
  const [page, setPage]     = useState(1)
  const navigate = useNavigate()

  useEffect(() => {
    setLoading(true)
    adminAPI.getComplaints({ search, status: statusF, priority: priorityF, page, perPage:20 })
      .then(d => { setComplaints(d.complaints); setTotal(d.total); setLoading(false) })
  }, [search, statusF, priorityF, page])

  return (
    <div className="app-layout"><Sidebar/>
      <main className="main-content"><div className="page-content">
        <div className="page-header">
          <div><h1>All Complaints</h1><p>{total} total</p></div>
        </div>
        <div className="filter-bar">
          <div className="search-wrap">
            <Search size={15} className="search-icon"/>
            <input className="search-input" placeholder="Search by title…" value={search} onChange={e => { setSearch(e.target.value); setPage(1) }}/>
          </div>
          <select className="filter-select" value={statusF} onChange={e => { setStatusF(e.target.value); setPage(1) }}>
            <option value="">All Status</option><option>Open</option><option>In Progress</option><option>Closed</option>
          </select>
          <select className="filter-select" value={priorityF} onChange={e => { setPriorityF(e.target.value); setPage(1) }}>
            <option value="">All Priority</option><option>Critical</option><option>High</option><option>Medium</option><option>Low</option>
          </select>
        </div>
        <div className="card" style={{ padding:0 }}>
          {loading ? <LoadingSpinner/> : (
            <div className="table-wrap">
              <table>
                <thead><tr><th>ID</th><th>Citizen</th><th>Title</th><th>Category</th><th>Priority</th><th>Dept.</th><th>Date</th><th>Status</th><th>Action</th></tr></thead>
                <tbody>
                  {complaints.map(c => (
                    <tr key={c.id}>
                      <td style={{ color:'var(--gray-400)',fontSize:12 }}>#{c.id}</td>
                      <td style={{ fontSize:12 }}>{c.user_name}</td>
                      <td>{truncate(c.title,32)}</td>
                      <td style={{ fontSize:12 }}>{c.category}</td>
                      <td><PriorityBadge priority={c.priority}/></td>
                      <td style={{ fontSize:12 }}>{truncate(c.department,20)}</td>
                      <td style={{ fontSize:12 }}>{formatDate(c.created_at)}</td>
                      <td><StatusBadge status={c.status}/></td>
                      <td><button className="btn btn-secondary btn-sm" onClick={() => navigate(`/admin/complaints/${c.id}`)}>Manage</button></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
        <div className="pagination">
          <button className="page-btn" disabled={page===1} onClick={() => setPage(p=>p-1)}>← Prev</button>
          <span style={{ fontSize:13,color:'var(--gray-500)' }}>Page {page} · {total} results</span>
          <button className="page-btn" disabled={complaints.length<20} onClick={() => setPage(p=>p+1)}>Next →</button>
        </div>
      </div></main>
    </div>
  )
}
