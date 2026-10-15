import React, { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, Legend } from 'recharts'
import { FileText, CheckCircle, Clock, Users } from 'lucide-react'
import Sidebar from '../../components/Sidebar'
import StatCard from '../../components/StatCard'
import StatusBadge from '../../components/StatusBadge'
import PriorityBadge from '../../components/PriorityBadge'
import LoadingSpinner from '../../components/LoadingSpinner'
import { adminAPI } from '../../services/api'
import { useAuth } from '../../context/AuthContext'
import { formatDate, truncate } from '../../utils/helpers'

const COLORS = ['#16a34a','#3b82f6','#f97316','#ef4444','#8b5cf6','#06b6d4']

export default function AdminDashboard() {
  const [stats, setStats]         = useState(null)
  const [analytics, setAnalytics] = useState(null)
  const [complaints, setComplaints] = useState([])
  const [loading, setLoading]     = useState(true)
  const { user } = useAuth()
  const navigate = useNavigate()

  useEffect(() => {
    Promise.all([adminAPI.getStatistics(), adminAPI.getAnalytics(), adminAPI.getComplaints({ perPage:8 })])
      .then(([s, a, c]) => { setStats(s); setAnalytics(a); setComplaints(c.complaints); setLoading(false) })
  }, [])

  if (loading) return <div className="app-layout"><Sidebar/><main className="main-content"><LoadingSpinner/></main></div>

  return (
    <div className="app-layout"><Sidebar/>
      <main className="main-content"><div className="page-content">
        <div className="page-header">
          <div><h1>Admin Dashboard</h1><p>Welcome back, <strong>{user?.name}</strong>. Here's the current overview.</p></div>
        </div>
        <div className="stats-grid">
          <StatCard label="Total Complaints" value={stats?.total}       icon={<FileText size={22}/>}    color="blue"/>
          <StatCard label="Resolved"         value={stats?.resolved}    icon={<CheckCircle size={22}/>} color="green" sub={`${stats?.resolution_rate}% rate`}/>
          <StatCard label="Open / Pending"   value={stats?.pending}     icon={<Clock size={22}/>}       color="orange"/>
          <StatCard label="Total Citizens"   value={stats?.total_users} icon={<Users size={22}/>}       color="blue"/>
        </div>
        <div className="grid-2" style={{ marginBottom:20 }}>
          <div className="card">
            <div className="card-header"><span className="card-title">By Category</span></div>
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={analytics?.category_distribution?.slice(0,8)||[]} margin={{ top:5,right:10,left:-10,bottom:60 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9"/>
                <XAxis dataKey="category" tick={{ fontSize:10 }} angle={-35} textAnchor="end"/>
                <YAxis tick={{ fontSize:11 }}/>
                <Tooltip/>
                <Bar dataKey="count" fill="#16a34a" radius={[4,4,0,0]}/>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="card">
            <div className="card-header"><span className="card-title">Status Distribution</span></div>
            <ResponsiveContainer width="100%" height={240}>
              <PieChart>
                <Pie data={analytics?.status_distribution||[]} dataKey="count" nameKey="status" cx="50%" cy="50%" outerRadius={80} label={({ status, percent }) => `${status} ${(percent*100).toFixed(0)}%`}>
                  {(analytics?.status_distribution||[]).map((_,i) => <Cell key={i} fill={['#16a34a','#3b82f6','#f97316','#ef4444'][i]||COLORS[i%COLORS.length]}/>)}
                </Pie>
                <Legend/><Tooltip/>
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>
        <div className="card">
          <div className="card-header">
            <span className="card-title">Recent Complaints</span>
            <button className="btn btn-secondary btn-sm" onClick={() => navigate('/admin/complaints')}>View All</button>
          </div>
          <div className="table-wrap">
            <table>
              <thead><tr><th>ID</th><th>Citizen</th><th>Title</th><th>Category</th><th>Priority</th><th>Status</th><th>Date</th><th>Action</th></tr></thead>
              <tbody>
                {complaints.map(c => (
                  <tr key={c.id}>
                    <td style={{ color:'var(--gray-400)',fontSize:12 }}>#{c.id}</td>
                    <td style={{ fontSize:12 }}>{c.user_name}</td>
                    <td>{truncate(c.title,30)}</td>
                    <td style={{ fontSize:12 }}>{c.category}</td>
                    <td><PriorityBadge priority={c.priority}/></td>
                    <td><StatusBadge status={c.status}/></td>
                    <td style={{ fontSize:12 }}>{formatDate(c.created_at)}</td>
                    <td><button className="btn btn-secondary btn-sm" onClick={() => navigate(`/admin/complaints/${c.id}`)}>Manage</button></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div></main>
    </div>
  )
}
