import React, { createContext, useContext, useState, useEffect } from 'react'
import { authAPI } from '../services/api'

const AuthContext = createContext(null)

export function AuthProvider({ children }) {
  const [user, setUser]       = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const saved = authAPI.getCurrentUser()
    if (saved) setUser(saved)
    setLoading(false)
  }, [])

  const login = async (email, password) => {
    const u = await authAPI.login(email, password)
    setUser(u)
    return u
  }

  const register = async (name, email, phone, password) => {
    const u = await authAPI.register(name, email, phone, password)
    setUser(u)
    return u
  }

  const logout = () => {
    authAPI.logout()
    setUser(null)
  }

  return (
    <AuthContext.Provider value={{ user, loading, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() { return useContext(AuthContext) }
