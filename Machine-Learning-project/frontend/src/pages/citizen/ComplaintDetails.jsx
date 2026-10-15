import React, { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { ArrowLeft, Building2, CalendarCheck, Clock, MapPin } from 'lucide-react'
import Sidebar from '../../components/Sidebar'
import StatusBadge from '../../components/StatusBadge'
import PriorityBadge from '../../components/PriorityBadge'
import ComplaintTimeline from '../../components/ComplaintTimeline'
import LoadingSpinner from '../../components/LoadingSpinner'
import { complaintsAPI } from '../../services/api'
import { formatDate, formatDateTime } from '../../utils/helpers'

export default function ComplaintDetails() {
  const { id }    = useParams()
  const navigate  = useNavigate()
  const [c, setC] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    complaintsAPI.getComplaint(id)
      .then(d => { setC(d); setLoading(false) })
      .catch(() => setLoading(false))
  }, [id])

  if (loading) return <div className="app-layout"><Sidebar/><main className="main-content"><LoadingSpinner/></main></div>
  if (!c)      return <div className="app-layout"><Sidebar/><main className="main-content"><div className="page-content"><p>Complaint not found.</p></div></main></div>

  return (
    <div className="app-layout"><Sidebar/>
      <main className="main-content"><div className="page-content">

        {/* Header */}
        <div className="page-header">
          <div>
            <button className="btn btn-secondary btn-sm" style={{ marginBottom:8 }}
              onClick={() => navigate('/citizen/complaints')}>
              <ArrowLeft size={14}/> Back to My Complaints
            </button>
            <h1>Complaint #{c.id}</h1>
            <p style={{ color:'var(--gray-500)',fontSize:13 }}>Submitted on {formatDate(c.created_at)}</p>
          </div>
          <StatusBadge status={c.status}/>
        </div>

        {/* ── KEY INFO STRIP (Department + Completion) ── */}
        <div style={{ display:'grid',gridTemplateColumns:'repeat(3,1fr)',gap:14,marginBottom:20 }}>

          {/* Assigned Department */}
          <div style={{ background:c.department ? '#eff6ff' : '#f8fafc',
                        border:`1.5px solid ${c.department ? '#bfdbfe' : 'var(--gray-200)'}`,
                        borderRadius:12,padding:18,display:'flex',alignItems:'flex-start',gap:12 }}>
            <Building2 size={22} style={{ color:'var(--blue)',flexShrink:0,marginTop:2 }}/>
            <div>
              <div style={{ fontSize:11,color:'var(--gray-400)',textTransform:'uppercase',letterSpacing:'.05em',marginBottom:4 }}>
                Assigned Department
              </div>
              <div style={{ fontSize:15,fontWeight:700,color:'var(--navy)' }}>
                {c.department || 'Pending assignment'}
              </div>
              <div style={{ fontSize:11.5,color:'var(--gray-500)',marginTop:2 }}>
                Auto-assigned by AI
              </div>
            </div>
          </div>

          {/* Urgency / Priority */}
          <div style={{ background:'#fff7ed',border:'1.5px solid #fed7aa',borderRadius:12,padding:18,
                        display:'flex',alignItems:'flex-start',gap:12 }}>
            <Clock size={22} style={{ color:'var(--orange)',flexShrink:0,marginTop:2 }}/>
            <div>
              <div style={{ fontSize:11,color:'var(--gray-400)',textTransform:'uppercase',letterSpacing:'.05em',marginBottom:4 }}>
                Urgency Level
              </div>
              <div style={{ fontSize:15,fontWeight:700,color:'var(--navy)' }}>
                <PriorityBadge priority={c.priority}/>
              </div>
            </div>
          </div>

          {/* Completion Date */}
          <div style={{ background: c.completed_at ? '#f0fdf4' : '#f8fafc',
                        border:`1.5px solid ${c.completed_at ? '#bbf7d0' : 'var(--gray-200)'}`,
                        borderRadius:12,padding:18,display:'flex',alignItems:'flex-start',gap:12 }}>
            <CalendarCheck size={22} style={{ color: c.completed_at ? 'var(--green)' : 'var(--gray-300)',flexShrink:0,marginTop:2 }}/>
            <div>
              <div style={{ fontSize:11,color:'var(--gray-400)',textTransform:'uppercase',letterSpacing:'.05em',marginBottom:4 }}>
                Completion Date
              </div>
              {c.completed_at ? (
                <>
                  <div style={{ fontSize:15,fontWeight:700,color:'var(--green-dark)' }}>
                    {formatDate(c.completed_at)}
                  </div>
                  <div style={{ fontSize:11.5,color:'var(--gray-500)',marginTop:2 }}>
                    {formatDateTime(c.completed_at)}
                  </div>
                </>
              ) : (
                <div style={{ fontSize:14,color:'var(--gray-400)' }}>
                  {c.status === 'In Progress' ? 'Being worked on…' : 'Not yet resolved'}
                </div>
              )}
            </div>
          </div>

        </div>

        {/* ── MAIN CONTENT ── */}
        <div className="grid-2" style={{ gap:20 }}>

          {/* Left — Complaint details */}
          <div style={{ display:'flex',flexDirection:'column',gap:16 }}>
            <div className="card">
              <div className="card-title" style={{ marginBottom:14 }}>Problem Description</div>
              <p style={{ color:'var(--gray-700)',lineHeight:1.75,fontSize:14,marginBottom:16 }}>{c.description}</p>

              {c.image_path && (
                <div style={{ marginBottom:14 }}>
                  <div style={{ fontSize:12,color:'var(--gray-400)',marginBottom:6 }}>PHOTO</div>
                  <img src={c.image_path} alt="complaint" style={{ maxWidth:'100%',borderRadius:8,border:'1px solid var(--gray-200)' }}/>
                </div>
              )}

              <div style={{ display:'grid',gridTemplateColumns:'1fr 1fr',gap:10 }}>
                {[
                  ['Category',   c.category],
                  ['Location',   c.location || '—'],
                  ['Submitted',  formatDate(c.created_at)],
                  ['Last Updated', formatDate(c.updated_at)],
                ].map(([label, value], i) => (
                  <div key={i} style={{ background:'var(--gray-50)',padding:'10px 12px',borderRadius:8 }}>
                    <div style={{ fontSize:11,color:'var(--gray-400)',textTransform:'uppercase',marginBottom:4 }}>{label}</div>
                    <div style={{ fontSize:13.5,fontWeight:500 }}>{value || '—'}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Right — Timeline */}
          <div className="card">
            <div className="card-title" style={{ marginBottom:16 }}>Status Timeline</div>
            <ComplaintTimeline history={c.history} currentStatus={c.status}/>

            {c.status === 'Closed' && c.completed_at && (
              <div style={{ marginTop:16,padding:'12px 14px',background:'#f0fdf4',
                            border:'1px solid #bbf7d0',borderRadius:8 }}>
                <div style={{ fontSize:12,color:'var(--green-dark)',fontWeight:600 }}>
                  ✅ Issue Resolved
                </div>
                <div style={{ fontSize:13,color:'var(--gray-600)',marginTop:4 }}>
                  Completed on {formatDateTime(c.completed_at)}
                </div>
              </div>
            )}
          </div>

        </div>
      </div></main>
    </div>
  )
}
