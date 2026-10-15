import React, { useState, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { Upload, X } from 'lucide-react'
import Sidebar from '../../components/Sidebar'
import { complaintsAPI, mlPredict } from '../../services/api'
import { useAuth } from '../../context/AuthContext'
import { useToast } from '../../components/Toast'

const URGENCY_OPTIONS = [
  { value: 'Urgent', label: '🔴 Urgent',  desc: 'Immediate danger or serious disruption' },
  { value: 'Normal', label: '🟡 Normal',  desc: 'Needs attention but not critical' },
  { value: 'Low',    label: '🟢 Low',     desc: 'Minor issue, can wait' },
]

export default function ReportIssue() {
  const { user }   = useAuth()
  const navigate   = useNavigate()
  const toast      = useToast()
  const fileRef    = useRef()

  const [description, setDescription] = useState('')
  const [urgency, setUrgency]         = useState('Normal')
  const [location, setLocation]       = useState('')
  const [preview, setPreview]         = useState(null)
  const [imageData, setImageData]     = useState(null)
  const [submitting, setSubmitting]   = useState(false)
  const [done, setDone]               = useState(null)
  const [errors, setErrors]           = useState({})

  const handleImage = e => {
    const file = e.target.files[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = ev => { setPreview(ev.target.result); setImageData(ev.target.result) }
    reader.readAsDataURL(file)
  }

  const handleSubmit = async e => {
    e.preventDefault()
    // Validate all required fields
    const errs = {}
    if (!description.trim())       errs.description = 'Please describe the problem.'
    if (!urgency)                   errs.urgency     = 'Please select urgency level.'
    if (Object.keys(errs).length > 0) { setErrors(errs); return }
    setErrors({})
    setSubmitting(true)
    try {
      // AI auto-predicts everything from description + urgency
      const pred = mlPredict(description, description)
      const complaint = await complaintsAPI.createComplaint({
        title       : description.slice(0, 80),   // first 80 chars as title
        description,
        category    : pred.predicted_category,
        priority    : urgency,                     // user's urgency choice = priority
        department  : pred.predicted_department,   // AI auto-assigned
        location,
        city        : '',
        area        : '',
        imageDataUrl: imageData,
        currentUser : user,
        prediction  : pred,
      })
      setDone(complaint)
      toast.success('Complaint submitted successfully!')
    } catch { toast.error('Submission failed. Please try again.') }
    setSubmitting(false)
  }

  /* ── Success screen ── */
  if (done) return (
    <div className="app-layout"><Sidebar />
      <main className="main-content"><div className="page-content">
        <div className="card" style={{ maxWidth:520,margin:'60px auto',textAlign:'center',padding:44 }}>
          <div style={{ fontSize:60,marginBottom:14 }}>✅</div>
          <h2 style={{ marginBottom:8 }}>Complaint Submitted!</h2>
          <p style={{ color:'var(--gray-500)',marginBottom:4 }}>
            Ticket ID: <strong style={{ color:'var(--green)',fontFamily:'monospace',fontSize:16 }}>#{done.id}</strong>
          </p>
          <div style={{ background:'var(--gray-50)',borderRadius:10,padding:16,margin:'16px 0',textAlign:'left' }}>
            <div style={{ fontSize:12,color:'var(--gray-400)',marginBottom:4 }}>AI AUTO-ASSIGNED TO</div>
            <div style={{ fontSize:16,fontWeight:600,color:'var(--navy)' }}>{done.department}</div>
            <div style={{ fontSize:12,color:'var(--gray-500)',marginTop:4 }}>Category: {done.category}</div>
          </div>
          <p style={{ color:'var(--gray-500)',fontSize:13,marginBottom:24 }}>
            You can track the status and see when it gets resolved in <strong>My Complaints</strong>.
          </p>
          <div style={{ display:'flex',gap:10,justifyContent:'center' }}>
            <button className="btn btn-primary"   onClick={() => navigate('/citizen/complaints')}>Track My Complaint</button>
            <button className="btn btn-secondary" onClick={() => { setDone(null); setDescription(''); setUrgency('Normal'); setLocation(''); setPreview(null); setImageData(null) }}>
              Report Another
            </button>
          </div>
        </div>
      </div></main>
    </div>
  )

  /* ── Form ── */
  return (
    <div className="app-layout"><Sidebar />
      <main className="main-content"><div className="page-content">
        <div className="page-header">
          <div>
            <h1>Report a Problem</h1>
            <p>Describe the issue and select urgency. Our AI will handle the rest.</p>
          </div>
        </div>

        <div style={{ display:'grid',gridTemplateColumns:'1fr 300px',gap:20,alignItems:'start' }}>

          {/* ── FORM ── */}
          <div className="card">
            <form onSubmit={handleSubmit}>

              {/* Description */}
              <div className="form-group">
                <label className="form-label" style={{ fontSize:14,fontWeight:600 }}>
                  What is the problem? *
                </label>
                <textarea
                  className="form-textarea"
                  rows={5}
                  value={description}
                  onChange={e => setDescription(e.target.value)}
                  placeholder="Describe the civic issue clearly — e.g. 'There is a large pothole on Oak Street near the park entrance, causing damage to vehicles.'"
                  required
                  style={{ fontSize:14 }}
                />
                <div className="form-hint">Be specific about location and the problem. More detail = faster resolution.</div>
              </div>

              {/* Urgency */}
              <div className="form-group">
                <label className="form-label" style={{ fontSize:14,fontWeight:600 }}>How urgent is this? *</label>
                <div style={{ display:'flex',flexDirection:'column',gap:8 }}>
                  {URGENCY_OPTIONS.map(opt => (
                    <label key={opt.value} style={{ display:'flex',alignItems:'center',gap:12,padding:'12px 14px',
                      border:`2px solid ${urgency===opt.value ? 'var(--green)' : 'var(--gray-200)'}`,
                      borderRadius:9,cursor:'pointer',background:urgency===opt.value ? '#f0fdf4' : 'white',
                      transition:'all .15s' }}>
                      <input type="radio" name="urgency" value={opt.value} checked={urgency===opt.value}
                        onChange={() => setUrgency(opt.value)} style={{ accentColor:'var(--green)',width:16,height:16 }}/>
                      <div>
                        <div style={{ fontWeight:600,fontSize:14 }}>{opt.label}</div>
                        <div style={{ fontSize:12,color:'var(--gray-500)' }}>{opt.desc}</div>
                      </div>
                    </label>
                  ))}
                </div>
              </div>

              {/* Location */}
              <div className="form-group">
                <label className="form-label" style={{ fontSize:14,fontWeight:600 }}>Location (optional)</label>
                <input className="form-input" value={location} onChange={e => setLocation(e.target.value)}
                  placeholder="e.g. Oak Street near Main Road, Brooklyn" />
              </div>

              {/* Image upload */}
              <div className="form-group">
                <label className="form-label" style={{ fontSize:14,fontWeight:600 }}>Photo of the problem (optional)</label>
                {preview ? (
                  <div style={{ position:'relative',display:'inline-block' }}>
                    <img src={preview} alt="preview" style={{ maxWidth:'100%',maxHeight:180,borderRadius:8,border:'1px solid var(--gray-200)' }}/>
                    <button type="button" onClick={() => { setPreview(null); setImageData(null) }}
                      style={{ position:'absolute',top:4,right:4,background:'#ef4444',color:'white',border:'none',
                               borderRadius:'50%',width:24,height:24,cursor:'pointer',fontSize:14,fontWeight:700 }}>×</button>
                  </div>
                ) : (
                  <div className="upload-zone" onClick={() => fileRef.current.click()}>
                    <Upload size={26} style={{ color:'var(--gray-400)',margin:'0 auto 8px',display:'block' }}/>
                    <p style={{ color:'var(--gray-500)',fontSize:13 }}>Click to upload a photo (JPG, PNG, GIF)</p>
                  </div>
                )}
                <input ref={fileRef} type="file" accept=".jpg,.jpeg,.png,.gif" onChange={handleImage} style={{ display:'none' }}/>
              </div>

              <button className="btn btn-primary" type="submit" disabled={submitting}
                style={{ width:'100%',justifyContent:'center',padding:13,fontSize:15,marginTop:4 }}>
                {submitting ? 'Submitting…' : '🚀 Submit Complaint'}
              </button>
            </form>
          </div>

          {/* ── SIDE INFO ── */}
          <div style={{ display:'flex',flexDirection:'column',gap:14 }}>
            <div className="card" style={{ background:'#f0fdf4',border:'1px solid #bbf7d0' }}>
              <div style={{ fontWeight:600,fontSize:14,color:'var(--green-dark)',marginBottom:10 }}>
                🤖 AI Auto-Assignment
              </div>
              <p style={{ fontSize:13,color:'var(--gray-600)',lineHeight:1.7 }}>
                After you submit, our AI automatically:<br/>
                ✅ Classifies the issue type<br/>
                ✅ Assigns the correct department<br/>
                ✅ Sets priority based on urgency<br/>
                <br/>
                You don't need to choose any department.
              </p>
            </div>

            <div className="card">
              <div style={{ fontWeight:600,fontSize:14,marginBottom:10 }}>📋 What happens next?</div>
              {[
                ['1','Your complaint is received'],
                ['2','AI classifies & assigns department'],
                ['3','Department investigates the issue'],
                ['4','Status updated to In Progress'],
                ['5','Issue resolved — completion date recorded'],
              ].map(([n,t]) => (
                <div key={n} style={{ display:'flex',gap:10,alignItems:'flex-start',padding:'7px 0',
                  borderBottom:+n<5?'1px solid var(--gray-100)':''}}>
                  <div style={{ width:22,height:22,borderRadius:'50%',background:'var(--green-light)',
                    color:'var(--green)',display:'flex',alignItems:'center',justifyContent:'center',
                    fontSize:11,fontWeight:700,flexShrink:0,marginTop:1 }}>{n}</div>
                  <span style={{ fontSize:13,color:'var(--gray-600)' }}>{t}</span>
                </div>
              ))}
            </div>
          </div>

        </div>
      </div></main>
    </div>
  )
}
