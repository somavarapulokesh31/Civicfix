import React, { useState, useEffect } from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, Legend } from 'recharts'
import { Brain, Target, TrendingUp, Award } from 'lucide-react'
import Sidebar from '../../components/Sidebar'
import StatCard from '../../components/StatCard'
import LoadingSpinner from '../../components/LoadingSpinner'
import { adminAPI } from '../../services/api'

const COLORS = ['#16a34a','#3b82f6','#f97316','#ef4444','#8b5cf6','#06b6d4']

export default function MLPredictions() {
  const [stats, setStats]     = useState(null)
  const [loading, setLoading] = useState(true)
  useEffect(() => { adminAPI.getMLStats().then(d => { setStats(d); setLoading(false) }) }, [])

  if (loading) return <div className="app-layout"><Sidebar/><main className="main-content"><LoadingSpinner/></main></div>
  const p = stats?.model_performance || {}

  return (
    <div className="app-layout"><Sidebar/>
      <main className="main-content"><div className="page-content">
        <div className="page-header"><div><h1>ML Predictions</h1><p>Model performance and prediction statistics</p></div></div>
        <div className="stats-grid">
          <StatCard label="Accuracy"  value={`${p.accuracy}%`}  icon={<Target size={22}/>} color="green"/>
          <StatCard label="Precision" value={`${p.precision}%`} icon={<Award size={22}/>}  color="blue"/>
          <StatCard label="Recall"    value={`${p.recall}%`}    icon={<TrendingUp size={22}/>} color="orange"/>
          <StatCard label="F1-Score"  value={`${p.f1_score}%`}  icon={<Brain size={22}/>}  color="blue"/>
        </div>
        <div className="card" style={{ marginBottom:20 }}>
          <div style={{ display:'flex',gap:24,marginBottom:10,flexWrap:'wrap' }}>
            <div><div style={{ fontSize:12,color:'var(--gray-400)' }}>Total Predictions</div><div style={{ fontSize:26,fontWeight:700 }}>{stats?.total_predictions||0}</div></div>
            <div><div style={{ fontSize:12,color:'var(--gray-400)' }}>Avg. Confidence</div><div style={{ fontSize:26,fontWeight:700 }}>{stats?.average_confidence||0}%</div></div>
          </div>
          <p style={{ fontSize:13,color:'var(--gray-500)',lineHeight:1.6 }}>CivicFix ML uses rule-based classification for instant predictions — no backend required. Predictions cover <strong>Issue Category</strong>, <strong>Priority Level</strong>, and <strong>Responsible Department</strong>.</p>
        </div>
        <div className="grid-2">
          <div className="card">
            <div className="card-header"><span className="card-title">Category Predictions</span></div>
            {stats?.category_distribution?.length > 0 ? (
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={stats.category_distribution} margin={{ top:5,right:10,left:-10,bottom:70 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9"/>
                  <XAxis dataKey="category" tick={{ fontSize:10 }} angle={-35} textAnchor="end"/>
                  <YAxis tick={{ fontSize:11 }}/><Tooltip/>
                  <Bar dataKey="count" fill="#16a34a" radius={[4,4,0,0]}/>
                </BarChart>
              </ResponsiveContainer>
            ) : <div className="empty-state"><p>No predictions yet. Submit complaints to see data.</p></div>}
          </div>
          <div className="card">
            <div className="card-header"><span className="card-title">Priority Predictions</span></div>
            {stats?.priority_distribution?.length > 0 ? (
              <ResponsiveContainer width="100%" height={260}>
                <PieChart>
                  <Pie data={stats.priority_distribution} dataKey="count" nameKey="priority" cx="50%" cy="50%" outerRadius={90} label={({ priority, percent }) => `${priority} ${(percent*100).toFixed(0)}%`}>
                    {stats.priority_distribution.map((_,i) => <Cell key={i} fill={COLORS[i%COLORS.length]}/>)}
                  </Pie>
                  <Legend/><Tooltip/>
                </PieChart>
              </ResponsiveContainer>
            ) : <div className="empty-state"><p>No predictions yet.</p></div>}
          </div>
        </div>
        <div className="card" style={{ marginTop:20 }}>
          <div className="card-title" style={{ marginBottom:14 }}>ML Pipeline</div>
          <div style={{ display:'grid',gridTemplateColumns:'1fr 1fr',gap:10 }}>
            {[['Dataset','erm2-nwe9.csv — 50,000 NYC 311 records'],['Target','status: Open | In Progress | Closed'],
              ['Features','complaint_type, agency, borough, channel'],['Algorithm','Rule-based + Logistic Regression'],
              ['Dataset Growth','New complaints auto-appended to erm2-nwe9.csv'],['Model Files','Place .pkl in backend/ml_models/'],
            ].map(([l,v],i) => (
              <div key={i} style={{ background:'var(--gray-50)',padding:'12px 14px',borderRadius:8 }}>
                <div style={{ fontSize:11,color:'var(--gray-400)',marginBottom:4,textTransform:'uppercase' }}>{l}</div>
                <div style={{ fontSize:13,color:'var(--gray-700)' }}>{v}</div>
              </div>
            ))}
          </div>
        </div>
      </div></main>
    </div>
  )
}
