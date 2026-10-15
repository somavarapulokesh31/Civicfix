import React, { useState } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'

export default function Login() {
  const [email, setEmail]       = useState('')
  const [password, setPassword] = useState('')
  const [error, setError]       = useState('')
  const [loading, setLoading]   = useState(false)
  const { login } = useAuth()
  const navigate  = useNavigate()

  const handleSubmit = async e => {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      const user = await login(email, password)
      navigate(user.role === 'admin' ? '/admin/dashboard' : '/citizen/dashboard')
    } catch (err) {
      setError(err.message || err.response?.data?.error || 'Login failed. Check credentials.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="auth-page">
      <div className="auth-card">
        <div className="auth-logo">
          <div className="brand-icon" style={{ width:52, height:52, fontSize:22, margin:'0 auto 10px', display:'flex', alignItems:'center', justifyContent:'center', borderRadius:12, background:'var(--green)', color:'white', fontWeight:800 }}>CF</div>
          <h2>Welcome Back</h2>
          <p>Sign in to your CivicFix account</p>
        </div>

        {error && <div className="error-msg">{error}</div>}

        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label className="form-label">Email Address</label>
            <input className="form-input" type="email" value={email}
              onChange={e => setEmail(e.target.value)} placeholder="you@example.com" required />
          </div>
          <div className="form-group">
            <label className="form-label">Password</label>
            <input className="form-input" type="password" value={password}
              onChange={e => setPassword(e.target.value)} placeholder="••••••••" required />
          </div>
          <button className="btn btn-primary" type="submit" disabled={loading}
            style={{ width:'100%', justifyContent:'center', padding:'11px', fontSize:15, marginTop:4 }}>
            {loading ? 'Signing in…' : 'Sign In'}
          </button>
        </form>

        <div className="auth-footer">
          Don't have an account? <Link to="/register">Register here</Link>
        </div>

        {/* Demo credentials box */}
        <div style={{ marginTop:20, padding:'14px', background:'var(--gray-50)', borderRadius:8, border:'1px solid var(--gray-200)', fontSize:12.5 }}>
          <strong>Demo Accounts:</strong><br />
          <span style={{ color:'var(--blue)' }}>Admin:</span> admin@civicfix.com / admin123<br />
          <span style={{ color:'var(--green)' }}>Citizen:</span> citizen@civicfix.com / citizen123
        </div>
      </div>
    </div>
  )
}
