import React from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { AuthProvider, useAuth } from './context/AuthContext'
import ToastContainer from './components/Toast'
import LoadingSpinner from './components/LoadingSpinner'
import Home             from './pages/Home'
import Login            from './pages/Login'
import Register         from './pages/Register'
import CitizenDashboard from './pages/citizen/CitizenDashboard'
import ReportIssue      from './pages/citizen/ReportIssue'
import MyComplaints     from './pages/citizen/MyComplaints'
import ComplaintDetails from './pages/citizen/ComplaintDetails'
import AdminDashboard   from './pages/admin/AdminDashboard'
import AdminComplaints  from './pages/admin/AdminComplaints'
import AdminComplaintDetails from './pages/admin/AdminComplaintDetails'
import MLPredictions    from './pages/admin/MLPredictions'
import Analytics        from './pages/admin/Analytics'
import AdminUsers       from './pages/admin/AdminUsers'
import Visualizations  from './pages/admin/Visualizations'

function Guard({ children, roles }) {
  const { user, loading } = useAuth()
  if (loading) return <LoadingSpinner />
  if (!user) return <Navigate to="/login" replace />
  if (roles && !roles.includes(user.role)) return <Navigate to="/" replace />
  return children
}

function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <ToastContainer />
        <Routes>
          <Route path="/"         element={<Home />} />
          <Route path="/login"    element={<Login />} />
          <Route path="/register" element={<Register />} />

          <Route path="/citizen/dashboard"       element={<Guard roles={['citizen']}><CitizenDashboard /></Guard>} />
          <Route path="/citizen/report"          element={<Guard roles={['citizen']}><ReportIssue /></Guard>} />
          <Route path="/citizen/complaints"      element={<Guard roles={['citizen']}><MyComplaints /></Guard>} />
          <Route path="/citizen/complaints/:id"  element={<Guard roles={['citizen']}><ComplaintDetails /></Guard>} />

          <Route path="/admin/dashboard"         element={<Guard roles={['admin']}><AdminDashboard /></Guard>} />
          <Route path="/admin/complaints"        element={<Guard roles={['admin']}><AdminComplaints /></Guard>} />
          <Route path="/admin/complaints/:id"    element={<Guard roles={['admin']}><AdminComplaintDetails /></Guard>} />
          <Route path="/admin/ml"                element={<Guard roles={['admin']}><MLPredictions /></Guard>} />
          <Route path="/admin/analytics"         element={<Guard roles={['admin']}><Analytics /></Guard>} />
          <Route path="/admin/users"             element={<Guard roles={['admin']}><AdminUsers /></Guard>} />
          <Route path="/admin/visualizations"   element={<Guard roles={['admin']}><Visualizations /></Guard>} />

          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  )
}

export default App
