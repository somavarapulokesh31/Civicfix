import React, { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { FileText, CheckCircle, Clock, Loader, PlusCircle } from 'lucide-react'
import Sidebar from '../../components/Sidebar'
import StatCard from '../../components/StatCard'
import StatusBadge from '../../components/StatusBadge'
import PriorityBadge from '../../components/PriorityBadge'
import LoadingSpinner from '../../components/LoadingSpinner'
import { complaintsAPI } from '../../services/api'
import { useAuth } from '../../context/AuthContext'
import { formatDate, truncate } from '../../utils/helpers'

export default function CitizenDashboard() {
  const [complaints, setComplaints] = useState([])
  const [loading, setLoading]       = useState(true)
  const { user } = useAuth()
  const navigate = useNavigate()

  useEffect(() => {
    if (!user) return
    complaintsAPI.getMyComplaints(user.id).then(data => {
      setComplaints(data)
      setLoading(false)
    })
  }, [user])

  const total  = complaints.length
  const closed = complaints.filter(c => c.status === 'Closed').length
  const open   = complaints.filter(c => c.status === 'Open').length
  const inprog = complaints.filter(c => c.status === 'In Progress').length

  return (
    <div className="app-layout">
      <Sidebar />
      <main className="main-content">
        <div className="page-content">
          <div className="page-header">
            <div>
              <h1>My Dashboard</h1>
              <p>Welcome back, <strong>{user?.name}</strong>! Track your civic complaints here.</p>
            </div>
            <button className="btn btn-primary" onClick={() => navigate('/citizen/report')}>
              <PlusCircle size={15} /> Report New Issue
            </button>
          </div>

          <div className="stats-grid">
            <StatCard label="Total Complaints" value={total}  icon={<FileText size={22}/>}    color="blue" />
            <StatCard label="Closed"           value={closed} icon={<CheckCircle size={22}/>} color="green" />
            <StatCard label="Open"             value={open}   icon={<Clock size={22}/>}        color="orange" />
            <StatCard label="In Progress"      value={inprog} icon={<Loader size={22}/>}       color="blue" />
          </div>

          <div className="card">
            <div className="card-header">
              <span className="card-title">Recent Complaints</span>
              <button className="btn btn-secondary btn-sm" onClick={() => navigate('/citizen/complaints')}>View All</button>
            </div>
            {loading ? <LoadingSpinner /> : complaints.length === 0 ? (
              <div className="empty-state">
                <FileText size={48} style={{ margin:'0 auto 12px', display:'block', color:'var(--gray-300)' }} />
                <h3>No complaints yet</h3>
                <p style={{ marginBottom: 16 }}>Start by reporting a civic issue in your area.</p>
                <button className="btn btn-primary" onClick={() => navigate('/citizen/report')}>Report an Issue</button>
              </div>
            ) : (
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr><th>Title</th><th>Category</th><th>Priority</th><th>Status</th><th>Date</th><th>Action</th></tr>
                  </thead>
                  <tbody>
                    {complaints.slice(0,8).map(c => (
                      <tr key={c.id}>
                        <td>{truncate(c.title, 40)}</td>
                        <td style={{ fontSize:12 }}>{c.category}</td>
                        <td><PriorityBadge priority={c.priority} /></td>
                        <td><StatusBadge status={c.status} /></td>
                        <td style={{ fontSize:12 }}>{formatDate(c.created_at)}</td>
                        <td>
                          <button className="btn btn-secondary btn-sm"
                            onClick={() => navigate(`/citizen/complaints/${c.id}`)}>View</button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  )
}
