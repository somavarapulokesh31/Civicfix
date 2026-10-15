import React from 'react'
import { NavLink, useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import { getInitials } from '../utils/helpers'
import {
  LayoutDashboard, FileText, Brain, BarChart2,
  Users, LogOut, PlusCircle, List, MapPin, PieChart
} from 'lucide-react'

const CITIZEN_LINKS = [
  { to: '/citizen/dashboard',   label: 'Dashboard',         icon: <LayoutDashboard size={16} /> },
  { to: '/citizen/report',      label: 'Report Issue',       icon: <PlusCircle size={16} /> },
  { to: '/citizen/complaints',  label: 'My Complaints',      icon: <List size={16} /> },
]

const ADMIN_LINKS = [
  { to: '/admin/dashboard',       label: 'Dashboard',      icon: <LayoutDashboard size={16} /> },
  { to: '/admin/complaints',      label: 'All Complaints', icon: <FileText size={16} /> },
  { to: '/admin/ml',              label: 'ML Predictions', icon: <Brain size={16} /> },
  { to: '/admin/analytics',       label: 'Analytics',      icon: <BarChart2 size={16} /> },
  { to: '/admin/visualizations',  label: 'Visualizations', icon: <PieChart size={16} /> },
  { to: '/admin/users',           label: 'Users',          icon: <Users size={16} /> },
]

export default function Sidebar() {
  const { user, logout } = useAuth()
  const navigate = useNavigate()

  const links = user?.role === 'admin' ? ADMIN_LINKS : CITIZEN_LINKS

  const handleLogout = () => {
    logout()
    navigate('/')
  }

  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <div className="brand-icon">CF</div>
        <div>
          <div className="brand-name">CivicFix</div>
          <div className="brand-tag">Better City, Better Life</div>
        </div>
      </div>

      <nav className="sidebar-nav">
        <div className="nav-section-title">
          {user?.role === 'admin' ? 'Admin Panel' : 'Citizen Portal'}
        </div>
        {links.map(link => (
          <NavLink key={link.to} to={link.to}
            className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
            {link.icon}
            <span>{link.label}</span>
          </NavLink>
        ))}
      </nav>

      <div className="sidebar-footer">
        <div className="sidebar-user">
          <div className="user-avatar">{getInitials(user?.name)}</div>
          <div>
            <div className="user-name">{user?.name}</div>
            <div className="user-role">{user?.role}</div>
          </div>
        </div>
        <button className="btn-logout" onClick={handleLogout}>
          <LogOut size={14} /> Logout
        </button>
      </div>
    </aside>
  )
}
