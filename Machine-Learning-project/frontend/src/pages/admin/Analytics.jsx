import React, { useState, useEffect } from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, Legend } from 'recharts'
import Sidebar from '../../components/Sidebar'
import LoadingSpinner from '../../components/LoadingSpinner'
import { adminAPI } from '../../services/api'

const COLORS = ['#16a34a','#3b82f6','#f97316','#ef4444','#8b5cf6','#06b6d4','#f59e0b']

export default function Analytics() {
  const [data, setData]       = useState(null)
  const [loading, setLoading] = useState(true)
  useEffect(() => { adminAPI.getAnalytics().then(d => { setData(d); setLoading(false) }) }, [])

  if (loading) return <div className="app-layout"><Sidebar/><main className="main-content"><LoadingSpinner/></main></div>
  const s = data?.stats || {}

  return (
    <div className="app-layout"><Sidebar/>
      <main className="main-content"><div className="page-content">
        <div className="page-header"><div><h1>Analytics</h1><p>Complaint trends and distribution insights</p></div></div>
        <div style={{ display:'grid',gridTemplateColumns:'repeat(4,1fr)',gap:14,marginBottom:24 }}>
          {[['Total',s.total,'#eff6ff','#1d4ed8'],['Resolved',s.resolved,'#f0fdf4','#15803d'],
            ['Pending',s.pending,'#fff7ed','#c2410c'],['Resolution %',`${s.resolution_rate}%`,'#fdf4ff','#7e22ce']
          ].map(([l,v,bg,color],i) => (
            <div key={i} style={{ background:bg,borderRadius:12,padding:'18px 20px' }}>
              <div style={{ fontSize:11,color:'var(--gray-500)',textTransform:'uppercase',letterSpacing:'.05em',marginBottom:6 }}>{l}</div>
              <div style={{ fontSize:28,fontWeight:700,color }}>{v??'—'}</div>
            </div>
          ))}
        </div>
        <div className="grid-2" style={{ marginBottom:20 }}>
          <div className="card">
            <div className="card-header"><span className="card-title">By Category</span></div>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={data?.category_distribution?.slice(0,10)||[]} margin={{ top:5,right:10,left:-10,bottom:80 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9"/>
                <XAxis dataKey="category" tick={{ fontSize:10 }} angle={-40} textAnchor="end"/>
                <YAxis tick={{ fontSize:11 }}/><Tooltip/>
                <Bar dataKey="count" fill="#3b82f6" radius={[4,4,0,0]}/>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="card">
            <div className="card-header"><span className="card-title">Status Distribution</span></div>
            <ResponsiveContainer width="100%" height={280}>
              <PieChart>
                <Pie data={data?.status_distribution||[]} dataKey="count" nameKey="status" cx="50%" cy="50%" outerRadius={95}
                  label={({ status,percent }) => `${status} ${(percent*100).toFixed(0)}%`}>
                  {(data?.status_distribution||[]).map((_,i) => <Cell key={i} fill={['#16a34a','#3b82f6','#f97316','#ef4444'][i]||COLORS[i%COLORS.length]}/>)}
                </Pie>
                <Legend/><Tooltip/>
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>
        <div className="card" style={{ marginBottom:20 }}>
          <div className="card-header"><span className="card-title">By Priority</span></div>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={data?.priority_distribution||[]} margin={{ top:5,right:20,left:-10,bottom:5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9"/>
              <XAxis dataKey="priority" tick={{ fontSize:12 }}/>
              <YAxis tick={{ fontSize:11 }}/><Tooltip/>
              <Bar dataKey="count" radius={[4,4,0,0]}>
                {(data?.priority_distribution||[]).map((d,i) => <Cell key={i} fill={({'Critical':'#ef4444','High':'#f97316','Medium':'#eab308','Low':'#16a34a'})[d.priority]||'#3b82f6'}/>)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="card">
          <div className="card-header"><span className="card-title">Category Breakdown</span></div>
          {(data?.category_distribution||[]).slice(0,8).map((d,i) => {
            const tot = (data?.category_distribution||[]).reduce((a,x)=>a+x.count,0)
            const pct = tot > 0 ? Math.round(d.count/tot*100) : 0
            return (
              <div key={i} style={{ marginBottom:12 }}>
                <div style={{ display:'flex',justifyContent:'space-between',marginBottom:4,fontSize:13 }}>
                  <span>{d.category}</span><span style={{ color:'var(--gray-500)' }}>{d.count} ({pct}%)</span>
                </div>
                <div style={{ height:8,background:'var(--gray-100)',borderRadius:4,overflow:'hidden' }}>
                  <div style={{ height:'100%',width:`${pct}%`,background:COLORS[i%COLORS.length],borderRadius:4 }}></div>
                </div>
              </div>
            )
          })}
        </div>
      </div></main>
    </div>
  )
}
