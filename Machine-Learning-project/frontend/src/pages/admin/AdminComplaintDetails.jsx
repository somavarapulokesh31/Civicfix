import React, { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { ArrowLeft } from 'lucide-react'
import Sidebar from '../../components/Sidebar'
import StatusBadge from '../../components/StatusBadge'
import PriorityBadge from '../../components/PriorityBadge'
import PredictionCard from '../../components/PredictionCard'
import ComplaintTimeline from '../../components/ComplaintTimeline'
import LoadingSpinner from '../../components/LoadingSpinner'
import { complaintsAPI, adminAPI } from '../../services/api'
import { useAuth } from '../../context/AuthContext'
import { formatDate, formatDateTime } from '../../utils/helpers'
import { useToast } from '../../components/Toast'

const DEPTS = ['NYPD','Department of Transportation','Sanitation Department',
  'Department of Environmental Protection','Housing Preservation','Department of Health','Municipal Corporation']

export default function AdminComplaintDetails() {
  const { id } = useParams()
  const navigate = useNavigate()
  const { user } = useAuth()
  const toast = useToast()
  const [c, setC]           = useState(null)
  const [loading, setLoading] = useState(true)
  const [note, setNote]     = useState('')
  const [newStatus, setNewStatus] = useState('')
  const [newDept, setNewDept]     = useState('')
  const [newPri, setNewPri]       = useState('')
  const [saving, setSaving] = useState(false)

  const load = () => complaintsAPI.getComplaint(id).then(d => { setC(d); setLoading(false) }).catch(() => setLoading(false))
  useEffect(() => { load() }, [id])

  const doStatus = async () => {
    if (!newStatus) { toast.error('Select a status'); return }
    setSaving(true)
    try {
      await adminAPI.updateStatus(id, newStatus, note || `Status changed to ${newStatus}`, user.name)
      toast.success('Status updated!'); load(); setNote(''); setNewStatus('')
    } catch { toast.error('Failed') }
    setSaving(false)
  }

  const doAssign = async () => {
    setSaving(true)
    try {
      await adminAPI.assignComplaint(id, { department:newDept, priority:newPri, note:note||'Assigned by admin', adminName:user.name })
      toast.success('Assigned!'); load(); setNote(''); setNewDept(''); setNewPri('')
    } catch { toast.error('Failed') }
    setSaving(false)
  }

  if (loading) return <div className="app-layout"><Sidebar/><main className="main-content"><LoadingSpinner/></main></div>
  if (!c) return <div className="app-layout"><Sidebar/><main className="main-content"><div className="page-content"><p>Not found</p></div></main></div>

  return (
    <div className="app-layout"><Sidebar/>
      <main className="main-content"><div className="page-content">
        <div className="page-header">
          <div>
            <button className="btn btn-secondary btn-sm" style={{ marginBottom:8 }} onClick={() => navigate('/admin/complaints')}><ArrowLeft size={14}/> Back</button>
            <h1>Complaint #{c.id}</h1>
            <p>By {c.user_name} · {formatDate(c.created_at)}</p>
          </div>
          <StatusBadge status={c.status}/>
        </div>
        <div style={{ display:'grid',gridTemplateColumns:'1fr 340px',gap:20 }}>
          <div style={{ display:'flex',flexDirection:'column',gap:16 }}>
            <div className="card">
              <div className="card-title" style={{ marginBottom:14 }}>Complaint Info</div>
              <h3 style={{ fontSize:17,fontWeight:600,marginBottom:8 }}>{c.title}</h3>
              <p style={{ color:'var(--gray-600)',lineHeight:1.7,marginBottom:14 }}>{c.description}</p>
              {c.image_path && <img src={c.image_path} alt="complaint" style={{ maxWidth:'100%',borderRadius:8,border:'1px solid var(--gray-200)',marginBottom:14 }}/>}
              <div style={{ display:'grid',gridTemplateColumns:'1fr 1fr',gap:10 }}>
                {[['Category',c.category],['Priority',<PriorityBadge priority={c.priority}/>],
                  ['Department',<span style={{color:'var(--blue)',fontWeight:600}}>{c.department}</span>],
                  ['Location',c.location||'—'],['City',c.city||'—'],['Status',<StatusBadge status={c.status}/>],
                  ['Submitted', formatDate(c.created_at)],
                  ['Completed', c.completed_at ? <span style={{color:'var(--green-dark)',fontWeight:600}}>{formatDate(c.completed_at)}</span> : <span style={{color:'var(--gray-400)'}}>Not yet</span>],
                ].map(([l,v],i) => (
                  <div key={i} style={{ background:'var(--gray-50)',padding:'10px 12px',borderRadius:8 }}>
                    <div style={{ fontSize:11,color:'var(--gray-400)',marginBottom:4,textTransform:'uppercase' }}>{l}</div>
                    <div style={{ fontSize:13.5,fontWeight:500 }}>{v||'—'}</div>
                  </div>
                ))}
              </div>
            </div>
            {c.prediction && <div className="card"><div className="card-title" style={{ marginBottom:14 }}>ML Prediction</div><PredictionCard prediction={c.prediction}/></div>}
          </div>
          <div style={{ display:'flex',flexDirection:'column',gap:16 }}>
            <div className="card">
              <div className="card-title" style={{ marginBottom:14 }}>Update Status</div>
              <p style={{ fontSize:12.5,color:'var(--gray-500)',marginBottom:12,lineHeight:1.6 }}>
                When you mark as <strong>Closed</strong>, the completion date is automatically recorded and shown to the citizen.
              </p>
              <div className="form-group">
                <select className="form-select" value={newStatus} onChange={e => setNewStatus(e.target.value)}>
                  <option value="">Select new status</option>
                  <option>Open</option><option>In Progress</option><option>Closed</option><option>Rejected</option>
                </select>
              </div>
              <div className="form-group">
                <textarea className="form-textarea" rows={2} placeholder="Add a note…" value={note} onChange={e => setNote(e.target.value)}/>
              </div>
              <button className="btn btn-primary" style={{ width:'100%',justifyContent:'center' }} onClick={doStatus} disabled={saving}>Update Status</button>
            </div>
            <div className="card">
              <div className="card-title" style={{ marginBottom:14 }}>Assign Department</div>
              <div className="form-group">
                <select className="form-select" value={newDept} onChange={e => setNewDept(e.target.value)}>
                  <option value="">Select department</option>
                  {DEPTS.map(d => <option key={d}>{d}</option>)}
                </select>
              </div>
              <div className="form-group">
                <select className="form-select" value={newPri} onChange={e => setNewPri(e.target.value)}>
                  <option value="">Change priority (optional)</option>
                  <option>Low</option><option>Medium</option><option>High</option><option>Critical</option>
                </select>
              </div>
              <button className="btn btn-secondary" style={{ width:'100%',justifyContent:'center' }} onClick={doAssign} disabled={saving}>Assign & Set In Progress</button>
            </div>
            <div className="card">
              <div className="card-title" style={{ marginBottom:14 }}>Status Timeline</div>
              <ComplaintTimeline history={c.history} currentStatus={c.status}/>
            </div>
          </div>
        </div>
      </div></main>
    </div>
  )
}
