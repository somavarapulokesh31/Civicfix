import React, { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { Search, PlusCircle } from 'lucide-react'
import Sidebar from '../../components/Sidebar'
import StatusBadge from '../../components/StatusBadge'
import PriorityBadge from '../../components/PriorityBadge'
import LoadingSpinner from '../../components/LoadingSpinner'
import { complaintsAPI } from '../../services/api'
import { useAuth } from '../../context/AuthContext'
import { formatDate, truncate } from '../../utils/helpers'

export default function MyComplaints() {
  const [all, setAll]         = useState([])
  const [filtered, setFiltered] = useState([])
  const [loading, setLoading] = useState(true)
  const [search, setSearch]   = useState('')
  const [statusF, setStatusF] = useState('')
  const { user } = useAuth()
  const navigate = useNavigate()

  useEffect(() => {
    if (!user) return
    complaintsAPI.getMyComplaints(user.id).then(d => { setAll(d); setFiltered(d); setLoading(false) })
  }, [user])

  useEffect(() => {
    let d = all
    if (statusF) d = d.filter(c => c.status === statusF)
    if (search)  d = d.filter(c => c.title.toLowerCase().includes(search.toLowerCase()))
    setFiltered(d)
  }, [search, statusF, all])

  return (
    <div className="app-layout"><Sidebar />
      <main className="main-content"><div className="page-content">
        <div className="page-header">
          <div><h1>My Complaints</h1><p>{filtered.length} complaints found</p></div>
          <button className="btn btn-primary" onClick={() => navigate('/citizen/report')}><PlusCircle size={15}/> New Complaint</button>
        </div>
        <div className="filter-bar">
          <div className="search-wrap">
            <Search size={15} className="search-icon"/>
            <input className="search-input" placeholder="Search…" value={search} onChange={e => setSearch(e.target.value)}/>
          </div>
          <select className="filter-select" value={statusF} onChange={e => setStatusF(e.target.value)}>
            <option value="">All Status</option>
            <option>Open</option><option>In Progress</option><option>Closed</option>
          </select>
        </div>
        <div className="card" style={{ padding:0 }}>
          {loading ? <LoadingSpinner/> : filtered.length === 0 ? (
            <div className="empty-state"><h3>No complaints found</h3></div>
          ) : (
            <div className="table-wrap">
              <table>
                <thead><tr><th>ID</th><th>Description</th><th>Urgency</th><th>Assigned Dept.</th><th>Date</th><th>Completed</th><th>Status</th><th>Action</th></tr></thead>
                <tbody>
                  {filtered.map(c => (
                    <tr key={c.id}>
                      <td style={{ color:'var(--gray-400)',fontSize:12 }}>#{c.id}</td>
                      <td style={{ maxWidth:200 }}>{truncate(c.description || c.title, 40)}</td>
                      <td><PriorityBadge priority={c.priority}/></td>
                      <td style={{ fontSize:12,color:'var(--blue)',fontWeight:500 }}>{c.department || '—'}</td>
                      <td style={{ fontSize:12 }}>{formatDate(c.created_at)}</td>
                      <td style={{ fontSize:12,color: c.completed_at ? 'var(--green-dark)' : 'var(--gray-400)', fontWeight: c.completed_at ? 600 : 400 }}>
                        {c.completed_at ? formatDate(c.completed_at) : '—'}
                      </td>
                      <td><StatusBadge status={c.status}/></td>
                      <td><button className="btn btn-secondary btn-sm" onClick={() => navigate(`/citizen/complaints/${c.id}`)}>View</button></td>
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
