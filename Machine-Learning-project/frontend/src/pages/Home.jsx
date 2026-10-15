import React from 'react'
import { useNavigate } from 'react-router-dom'
import { MapPin, Zap, BarChart2, CheckCircle, Users, Clock, Shield, ArrowRight } from 'lucide-react'
import { useAuth } from '../context/AuthContext'

export default function Home() {
  const navigate = useNavigate()
  const { user } = useAuth()

  const goToDashboard = () => {
    if (user) navigate(user.role === 'admin' ? '/admin/dashboard' : '/citizen/dashboard')
    else navigate('/register')
  }

  return (
    <div>
      {/* NAVBAR */}
      <nav className="navbar">
        <div className="navbar-brand">
          <div className="brand-icon">CF</div>
          <span className="brand-name">CivicFix</span>
        </div>
        <div className="navbar-links">
          <a href="#how" className="navbar-link">How It Works</a>
          <a href="#features" className="navbar-link">Features</a>
          <a href="#stats" className="navbar-link">Statistics</a>
        </div>
        <div className="navbar-actions">
          {user ? (
            <button className="btn btn-primary" onClick={goToDashboard}>
              Go to Dashboard <ArrowRight size={15} />
            </button>
          ) : (
            <>
              <button className="btn btn-secondary" onClick={() => navigate('/login')}>Login</button>
              <button className="btn btn-primary"   onClick={() => navigate('/register')}>Register</button>
            </>
          )}
        </div>
      </nav>

      {/* HERO */}
      <section className="hero">
        <div style={{ maxWidth: 700, margin: '0 auto' }}>
          <h1>Report Problems.<br /><span>Improve Your City.</span></h1>
          <p>CivicFix uses AI-powered technology to classify, prioritize, and manage civic complaints efficiently — connecting citizens with the right departments instantly.</p>
          <div className="hero-btns">
            <button className="btn btn-primary" style={{ fontSize: 16, padding: '13px 28px' }} onClick={goToDashboard}>
              Report an Issue <ArrowRight size={16} />
            </button>
            <button className="btn btn-secondary" style={{ fontSize: 16, padding: '13px 28px', color: 'white', borderColor: 'rgba(255,255,255,.3)', background: 'rgba(255,255,255,.08)' }} onClick={() => document.getElementById('how').scrollIntoView({ behavior: 'smooth' })}>
              See How It Works
            </button>
          </div>
        </div>
      </section>

      {/* STATS */}
      <section className="stats-section" id="stats">
        <div className="stats-inner">
          <div><div className="stat-num">50K+</div><div className="stat-text">Total Complaints</div></div>
          <div><div className="stat-num">87%</div><div className="stat-text">Resolution Rate</div></div>
          <div><div className="stat-num">2.4 days</div><div className="stat-text">Avg. Resolution Time</div></div>
          <div><div className="stat-num">15+</div><div className="stat-text">City Departments</div></div>
        </div>
      </section>

      {/* HOW IT WORKS */}
      <section className="how-section" id="how">
        <div className="section-title">How CivicFix Works</div>
        <div className="section-sub">Four simple steps from complaint to resolution</div>
        <div className="how-steps">
          {[
            { n:1, title:'Report an Issue',       desc:'Citizen enters complaint description and optionally uploads an image of the problem.',   icon:<MapPin size={22} /> },
            { n:2, title:'AI Analyzes It',         desc:'Our machine learning model analyzes the complaint text and image to classify the issue.', icon:<Zap size={22} /> },
            { n:3, title:'Smart Routing',           desc:'The system predicts issue category, priority level, and the responsible department.',     icon:<BarChart2 size={22} /> },
            { n:4, title:'Track Resolution',        desc:'Citizen monitors the complaint status in real-time until it is fully resolved.',           icon:<CheckCircle size={22} /> },
          ].map(s => (
            <div key={s.n} className="how-step">
              <div className="step-num">{s.n}</div>
              <div style={{ color: 'var(--green)', marginBottom: 10 }}>{s.icon}</div>
              <h3 style={{ fontSize: 15, fontWeight: 600, marginBottom: 8 }}>{s.title}</h3>
              <p style={{ fontSize: 13, color: 'var(--gray-500)', lineHeight: 1.6 }}>{s.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* FEATURES */}
      <section style={{ background: 'var(--gray-50)', padding: '70px 32px' }} id="features">
        <div style={{ maxWidth: 1000, margin: '0 auto' }}>
          <div className="section-title">Why CivicFix?</div>
          <div className="section-sub">Built for citizens, powered by AI</div>
          <div className="features-grid" style={{ marginTop: 40 }}>
            {[
              { icon:<Zap size={24}/>,       title:'AI Classification',      desc:'ML model automatically classifies your issue into the right category with confidence scores.' },
              { icon:<Shield size={24}/>,    title:'Smart Priority',          desc:'Automatic priority prediction ensures urgent issues are escalated to Critical or High.' },
              { icon:<Users size={24}/>,     title:'Department Routing',      desc:'Complaints are automatically routed to the correct city department — no manual assignment.' },
              { icon:<Clock size={24}/>,     title:'Real-Time Tracking',      desc:'Track every status update — from Open to In Progress to Closed — with a visual timeline.' },
              { icon:<BarChart2 size={24}/>, title:'Admin Analytics',         desc:'Administrators see live dashboards with charts, statistics, and ML prediction performance.' },
              { icon:<CheckCircle size={24}/>,title:'Dataset Integration',    desc:'Every submitted complaint is automatically added to the training dataset for continuous improvement.' },
            ].map((f, i) => (
              <div key={i} className="feature-card">
                <div className="feature-icon">{f.icon}</div>
                <h3 style={{ fontSize: 15, fontWeight: 600, marginBottom: 8 }}>{f.title}</h3>
                <p style={{ fontSize: 13, color: 'var(--gray-500)', lineHeight: 1.6 }}>{f.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section style={{ background: 'var(--green)', padding: '60px 32px', textAlign: 'center' }}>
        <h2 style={{ color: 'white', fontSize: 28, fontWeight: 700, marginBottom: 12 }}>Ready to Fix Your City?</h2>
        <p style={{ color: 'rgba(255,255,255,.8)', fontSize: 16, marginBottom: 28 }}>Join thousands of citizens making their communities better.</p>
        <button className="btn" style={{ background: 'white', color: 'var(--green)', fontSize: 15, padding: '12px 28px' }} onClick={goToDashboard}>
          Get Started Free <ArrowRight size={15} />
        </button>
      </section>

      {/* FOOTER */}
      <footer style={{ background: 'var(--navy)', color: 'var(--gray-400)', padding: '28px 32px', textAlign: 'center', fontSize: 13 }}>
        © 2024 CivicFix · Better City, Better Life · Built with AI for smarter civic management
      </footer>
    </div>
  )
}
