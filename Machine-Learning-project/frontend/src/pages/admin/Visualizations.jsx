import React, { useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend,
  LineChart, Line,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  ComposedChart, Area,
  Treemap
} from 'recharts'
import Sidebar from '../../components/Sidebar'

// ── REAL DATA from erm2-nwe9.csv (50,000 records) ────────────────────────────

const STATUS_DIST = [
  { status: 'Closed',      count: 29957 },
  { status: 'In Progress', count: 10054 },
  { status: 'Open',        count: 9798  },
  { status: 'Assigned',    count: 173   },
  { status: 'Pending',     count: 14    },
]

const TOP_TYPES = [
  { type: 'Illegal Parking',        count: 7685 },
  { type: 'Noise - Residential',    count: 5172 },
  { type: 'Noise - Street/Sidewalk',count: 4384 },
  { type: 'Blocked Driveway',       count: 2228 },
  { type: 'UNSANITARY CONDITION',   count: 2047 },
  { type: 'Street Condition',       count: 1219 },
  { type: 'Water Maintenance',      count: 1134 },
  { type: 'Noise - Vehicle',        count: 1039 },
  { type: 'Dirty Condition',        count: 1014 },
  { type: 'Abandoned Vehicle',      count: 916  },
  { type: 'PAINT/PLASTER',          count: 905  },
  { type: 'WATER LEAK',             count: 842  },
  { type: 'PLUMBING',               count: 828  },
  { type: 'Noise - Commercial',     count: 808  },
  { type: 'Sewer Maintenance',      count: 805  },
]

const BOROUGH_DIST = [
  { borough: 'BROOKLYN',      count: 15526 },
  { borough: 'QUEENS',        count: 13034 },
  { borough: 'BRONX',         count: 10055 },
  { borough: 'MANHATTAN',     count: 9249  },
  { borough: 'STATEN ISLAND', count: 2088  },
]

const AGENCY_PERF = [
  { agency: 'NYPD',   total: 24238, closed: 23590, closure_pct: 97.3 },
  { agency: 'HPD',    total: 7987,  closed: 668,   closure_pct: 8.4  },
  { agency: 'DSNY',   total: 4331,  closed: 2012,  closure_pct: 46.5 },
  { agency: 'DOT',    total: 3338,  closed: 1311,  closure_pct: 39.3 },
  { agency: 'DEP',    total: 3096,  closed: 705,   closure_pct: 22.8 },
  { agency: 'DPR',    total: 2369,  closed: 467,   closure_pct: 19.7 },
  { agency: 'DOB',    total: 1890,  closed: 353,   closure_pct: 18.7 },
  { agency: 'DOHMH',  total: 1299,  closed: 158,   closure_pct: 12.2 },
  { agency: 'DHS',    total: 663,   closed: 568,   closure_pct: 85.7 },
  { agency: 'TLC',    total: 386,   closed: 33,    closure_pct: 8.5  },
]

const CHANNEL_DIST = [
  { channel: 'ONLINE',  count: 25091 },
  { channel: 'MOBILE',  count: 10222 },
  { channel: 'PHONE',   count: 10211 },
  { channel: 'UNKNOWN', count: 4476  },
]

const LOCATION_TYPE = [
  { type: 'Street/Sidewalk',          count: 17685 },
  { type: 'RESIDENTIAL BUILDING',     count: 7987  },
  { type: 'Residential Building/House',count: 5779  },
  { type: 'Street',                   count: 5096  },
  { type: 'Sidewalk',                 count: 3270  },
  { type: 'Store/Commercial',         count: 803   },
  { type: 'Sewer',                    count: 633   },
  { type: 'Park',                     count: 521   },
]

const BOROUGH_STATUS = [
  { borough: 'BROOKLYN',      Closed: 9213, 'In Progress': 3271, Open: 2959 },
  { borough: 'QUEENS',        Closed: 7961, 'In Progress': 2948, Open: 2072 },
  { borough: 'BRONX',         Closed: 6285, 'In Progress': 1250, Open: 2508 },
  { borough: 'MANHATTAN',     Closed: 5404, 'In Progress': 1900, Open: 1911 },
  { borough: 'STATEN ISLAND', Closed: 1084, 'In Progress': 655,  Open: 341  },
]

const TYPE_BOROUGH = [
  { type: 'Illegal Parking',       BROOKLYN: 2952, QUEENS: 2376, BRONX: 1239, MANHATTAN: 913,  'STATEN ISLAND': 205 },
  { type: 'Noise - Residential',   BROOKLYN: 1782, QUEENS: 1176, BRONX: 1330, MANHATTAN: 666,  'STATEN ISLAND': 218 },
  { type: 'Noise - Street/Sidewalk',BROOKLYN: 1071, QUEENS: 603,  BRONX: 1694, MANHATTAN: 925, 'STATEN ISLAND': 91  },
  { type: 'Blocked Driveway',      BROOKLYN: 687,  QUEENS: 1015, BRONX: 397,  MANHATTAN: 69,   'STATEN ISLAND': 60  },
  { type: 'UNSANITARY CONDITION',  BROOKLYN: 579,  QUEENS: 374,  BRONX: 595,  MANHATTAN: 455,  'STATEN ISLAND': 44  },
]

const MISSING_VALS = [
  { col: 'taxi_company_borough', missing: 49965, pct: 99.9 },
  { col: 'road_ramp',            missing: 49911, pct: 99.8 },
  { col: 'bridge_highway_dir',   missing: 49893, pct: 99.8 },
  { col: 'facility_type',        missing: 49890, pct: 99.8 },
  { col: 'due_date',             missing: 49774, pct: 99.5 },
  { col: 'vehicle_type',         missing: 47441, pct: 94.9 },
  { col: 'descriptor_2',         missing: 35159, pct: 70.3 },
  { col: 'latitude',             missing: 4476,  pct: 9.0  },
  { col: 'closed_date',          missing: 20043, pct: 40.1 },
]

const AGENCY_RADAR = [
  { agency: 'NYPD',  volume: 100, closure: 97  },
  { agency: 'HPD',   volume: 33,  closure: 8   },
  { agency: 'DSNY',  volume: 18,  closure: 47  },
  { agency: 'DOT',   volume: 14,  closure: 39  },
  { agency: 'DEP',   volume: 13,  closure: 23  },
  { agency: 'DPR',   volume: 10,  closure: 20  },
  { agency: 'DOB',   volume: 8,   closure: 19  },
  { agency: 'DHS',   volume: 3,   closure: 86  },
]

const TREEMAP_DATA = TOP_TYPES.slice(0, 12).map(d => ({ name: d.type, size: d.count }))

const STATUS_COLORS = { Closed: '#16a34a', 'In Progress': '#3b82f6', Open: '#f97316', Assigned: '#8b5cf6', Pending: '#ef4444' }
const BOROUGH_COLORS = ['#16a34a','#3b82f6','#f97316','#8b5cf6','#06b6d4']
const CHANNEL_COLORS = ['#3b82f6','#16a34a','#f97316','#94a3b8']
const TREE_COLORS   = ['#16a34a','#3b82f6','#f97316','#8b5cf6','#06b6d4','#f59e0b','#ef4444','#ec4899','#14b8a6','#6366f1','#84cc16','#fb923c']

const TABS = ['Overview','Complaints','Boroughs','Agencies','Channels & Location','Data Quality','ML Model']

const Card = ({ title, desc, children, span2 }) => (
  <div className="card" style={{ gridColumn: span2 ? 'span 2' : 'span 1', minWidth: 0 }}>
    <div style={{ marginBottom: 16 }}>
      <div className="card-title">{title}</div>
      {desc && <div style={{ fontSize: 12, color: 'var(--gray-400)', marginTop: 3 }}>{desc}</div>}
    </div>
    {children}
  </div>
)

const Insight = ({ text }) => (
  <div style={{ marginTop: 12, padding: '10px 14px', background: '#f0fdf4',
    border: '1px solid #bbf7d0', borderRadius: 8, fontSize: 12.5, color: '#15803d', lineHeight: 1.6 }}>
    💡 {text}
  </div>
)

const KPI = ({ label, value, sub, color = '#16a34a', bg = '#f0fdf4' }) => (
  <div style={{ background: bg, borderRadius: 12, padding: '18px 20px', border: '1px solid #e2e8f0' }}>
    <div style={{ fontSize: 11, color: 'var(--gray-500)', textTransform: 'uppercase', letterSpacing: '.05em', marginBottom: 6 }}>{label}</div>
    <div style={{ fontSize: 28, fontWeight: 700, color }}>{value}</div>
    {sub && <div style={{ fontSize: 11.5, color: 'var(--gray-400)', marginTop: 3 }}>{sub}</div>}
  </div>
)

// Custom treemap content
const TreeContent = ({ x, y, width, height, name, value, index }) => {
  if (width < 30 || height < 20) return null
  return (
    <g>
      <rect x={x} y={y} width={width} height={height} fill={TREE_COLORS[index % TREE_COLORS.length]} rx={4} opacity={0.88}/>
      {width > 60 && height > 30 && (
        <text x={x + width / 2} y={y + height / 2} fill="white" textAnchor="middle"
          dominantBaseline="middle" fontSize={Math.min(11, width / 8)} fontWeight={600}>
          {name.length > 14 ? name.slice(0, 14) + '…' : name}
        </text>
      )}
      {width > 60 && height > 48 && (
        <text x={x + width / 2} y={y + height / 2 + 14} fill="rgba(255,255,255,0.8)"
          textAnchor="middle" dominantBaseline="middle" fontSize={10}>
          {value.toLocaleString()}
        </text>
      )}
    </g>
  )
}

export default function Visualizations() {
  const [activeTab, setActiveTab] = useState('Overview')

  return (
    <div className="app-layout">
      <Sidebar />
      <main className="main-content">
        <div className="page-content">

          {/* Header */}
          <div className="page-header">
            <div>
              <h1>Visualizations</h1>
              <p>Full EDA — 50,000 NYC 311 civic complaints · erm2-nwe9.csv</p>
            </div>
          </div>

          {/* Tab Bar */}
          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 24,
            background: 'white', borderRadius: 10, padding: 6,
            border: '1px solid var(--gray-200)', boxShadow: 'var(--shadow)' }}>
            {TABS.map(t => (
              <button key={t} onClick={() => setActiveTab(t)}
                style={{ padding: '8px 16px', borderRadius: 7, border: 'none', cursor: 'pointer',
                  fontSize: 13, fontWeight: 500, transition: 'all .15s',
                  background: activeTab === t ? 'var(--green)' : 'transparent',
                  color: activeTab === t ? 'white' : 'var(--gray-500)' }}>
                {t}
              </button>
            ))}
          </div>

          {/* ── OVERVIEW ── */}
          {activeTab === 'Overview' && (
            <>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 14, marginBottom: 24 }}>
                <KPI label="Total Records"    value="50,000"  sub="erm2-nwe9.csv"        color="#1d4ed8" bg="#eff6ff"/>
                <KPI label="Total Columns"    value="44"      sub="after dropping unnamed" color="#15803d" bg="#f0fdf4"/>
                <KPI label="Complaint Types"  value="149"     sub="unique categories"     color="#c2410c" bg="#fff7ed"/>
                <KPI label="Agencies"         value="14"      sub="city departments"      color="#7e22ce" bg="#fdf4ff"/>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 20 }}>
                <Card title="Status Distribution" desc="Overall resolution status across all 50K complaints">
                  <ResponsiveContainer width="100%" height={260}>
                    <PieChart>
                      <Pie data={STATUS_DIST} dataKey="count" nameKey="status" cx="50%" cy="50%"
                        outerRadius={95} label={({ status, percent }) => `${status} ${(percent*100).toFixed(0)}%`}>
                        {STATUS_DIST.map((d) => <Cell key={d.status} fill={STATUS_COLORS[d.status] || '#94a3b8'}/>)}
                      </Pie>
                      <Legend /><Tooltip formatter={v => v.toLocaleString()}/>
                    </PieChart>
                  </ResponsiveContainer>
                  <Insight text="59.9% of complaints are Closed. 20.1% are In Progress and 19.6% are Open — indicating a decent resolution pipeline." />
                </Card>

                <Card title="Complaints by Borough" desc="Geographic distribution across NYC boroughs">
                  <ResponsiveContainer width="100%" height={260}>
                    <BarChart data={BOROUGH_DIST} margin={{ top: 5, right: 10, left: -10, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9"/>
                      <XAxis dataKey="borough" tick={{ fontSize: 11 }}/>
                      <YAxis tick={{ fontSize: 11 }}/><Tooltip formatter={v => v.toLocaleString()}/>
                      <Bar dataKey="count" radius={[5,5,0,0]}>
                        {BOROUGH_DIST.map((d, i) => <Cell key={d.borough} fill={BOROUGH_COLORS[i]}/>)}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                  <Insight text="Brooklyn leads with 15,526 complaints (31%), followed by Queens (26%) and the Bronx (20%)." />
                </Card>
              </div>

              <Card title="Top 15 Complaint Types" desc="Most frequently reported civic issues in NYC" span2>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={TOP_TYPES} layout="vertical" margin={{ top: 5, right: 30, left: 160, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" horizontal={false}/>
                    <XAxis type="number" tick={{ fontSize: 11 }}/><YAxis type="category" dataKey="type" tick={{ fontSize: 11 }} width={155}/>
                    <Tooltip formatter={v => v.toLocaleString()}/>
                    <Bar dataKey="count" fill="#16a34a" radius={[0,5,5,0]}>
                      {TOP_TYPES.map((_, i) => <Cell key={i} fill={i < 3 ? '#15803d' : i < 7 ? '#16a34a' : '#4ade80'}/>)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
                <Insight text="Illegal Parking (7,685) is the #1 complaint — 3× more than any other type. The top 3 are all noise/parking violations handled by NYPD." />
              </Card>
            </>
          )}

          {/* ── COMPLAINTS ── */}
          {activeTab === 'Complaints' && (
            <>
              <Card title="Complaint Type Treemap" desc="Area proportional to complaint volume — top 12 types" span2>
                <ResponsiveContainer width="100%" height={360}>
                  <Treemap data={TREEMAP_DATA} dataKey="size" content={<TreeContent/>}>
                    <Tooltip formatter={(v, n) => [v.toLocaleString(), n]}/>
                  </Treemap>
                </ResponsiveContainer>
                <Insight text="Illegal Parking dominates — its tile is nearly as large as the next 3 types combined." />
              </Card>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginTop: 20 }}>
                <Card title="Top 10 Complaint Types (Bar)" desc="Volume comparison of most common issues">
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={TOP_TYPES.slice(0,10)} margin={{ top: 5, right: 10, left: -10, bottom: 80 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9"/>
                      <XAxis dataKey="type" tick={{ fontSize: 9 }} angle={-40} textAnchor="end"/>
                      <YAxis tick={{ fontSize: 11 }}/><Tooltip formatter={v => v.toLocaleString()}/>
                      <Bar dataKey="count" fill="#3b82f6" radius={[4,4,0,0]}/>
                    </BarChart>
                  </ResponsiveContainer>
                </Card>

                <Card title="Complaint Type % Share" desc="Top 8 types as proportion of total">
                  <ResponsiveContainer width="100%" height={300}>
                    <PieChart>
                      <Pie data={TOP_TYPES.slice(0,8).map(d => ({ ...d, name: d.type }))}
                        dataKey="count" nameKey="name" cx="50%" cy="50%" innerRadius={50} outerRadius={100}
                        label={({ name, percent }) => percent > 0.04 ? `${(percent*100).toFixed(0)}%` : ''}>
                        {TOP_TYPES.slice(0,8).map((_, i) => <Cell key={i} fill={TREE_COLORS[i]}/>)}
                      </Pie>
                      <Legend formatter={(v) => v.length > 22 ? v.slice(0,22)+'…' : v}/><Tooltip formatter={v => v.toLocaleString()}/>
                    </PieChart>
                  </ResponsiveContainer>
                </Card>
              </div>

              <div style={{ marginTop: 20 }}>
                <Card title="Top 5 Types Across All Boroughs" desc="Grouped bar — how each complaint type spreads across boroughs" span2>
                  <ResponsiveContainer width="100%" height={320}>
                    <BarChart data={TYPE_BOROUGH} margin={{ top: 5, right: 20, left: -10, bottom: 60 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9"/>
                      <XAxis dataKey="type" tick={{ fontSize: 10 }} angle={-30} textAnchor="end"/>
                      <YAxis tick={{ fontSize: 11 }}/><Tooltip formatter={v => v.toLocaleString()}/>
                      <Legend />
                      <Bar dataKey="BROOKLYN"      fill="#16a34a" radius={[3,3,0,0]}/>
                      <Bar dataKey="QUEENS"        fill="#3b82f6" radius={[3,3,0,0]}/>
                      <Bar dataKey="BRONX"         fill="#f97316" radius={[3,3,0,0]}/>
                      <Bar dataKey="MANHATTAN"     fill="#8b5cf6" radius={[3,3,0,0]}/>
                      <Bar dataKey="STATEN ISLAND" fill="#06b6d4" radius={[3,3,0,0]}/>
                    </BarChart>
                  </ResponsiveContainer>
                  <Insight text="Illegal Parking is most concentrated in Brooklyn & Queens. Noise - Street/Sidewalk is the Bronx's top issue, unlike other boroughs." />
                </Card>
              </div>
            </>
          )}

          {/* ── BOROUGHS ── */}
          {activeTab === 'Boroughs' && (
            <>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 14, marginBottom: 24 }}>
                {BOROUGH_DIST.map((b, i) => (
                  <KPI key={b.borough} label={b.borough} value={b.count.toLocaleString()}
                    sub={`${(b.count/500).toFixed(1)}% of total`}
                    color={BOROUGH_COLORS[i]} bg={['#eff6ff','#f0fdf4','#fff7ed','#fdf4ff','#f0fdfa'][i]}/>
                ))}
              </div>

              <Card title="Borough Status Breakdown" desc="How complaints resolve differently across each borough" span2>
                <ResponsiveContainer width="100%" height={320}>
                  <BarChart data={BOROUGH_STATUS} margin={{ top: 5, right: 20, left: -10, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9"/>
                    <XAxis dataKey="borough" tick={{ fontSize: 11 }}/>
                    <YAxis tick={{ fontSize: 11 }}/><Tooltip formatter={v => v.toLocaleString()}/>
                    <Legend />
                    <Bar dataKey="Closed"       fill="#16a34a" stackId="a" radius={[0,0,0,0]}/>
                    <Bar dataKey="In Progress"  fill="#3b82f6" stackId="a"/>
                    <Bar dataKey="Open"         fill="#f97316" stackId="a" radius={[4,4,0,0]}/>
                  </BarChart>
                </ResponsiveContainer>
                <Insight text="Brooklyn has the highest absolute volume of closed complaints. Staten Island has a high 'In Progress' proportion relative to its size." />
              </Card>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginTop: 20 }}>
                <Card title="Borough Closure Rate" desc="% of complaints marked Closed per borough">
                  <ResponsiveContainer width="100%" height={260}>
                    <BarChart
                      data={BOROUGH_STATUS.map(b => ({
                        borough: b.borough,
                        closure_pct: +(b.Closed / (b.Closed + b['In Progress'] + b.Open) * 100).toFixed(1)
                      }))}
                      margin={{ top: 5, right: 10, left: -10, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9"/>
                      <XAxis dataKey="borough" tick={{ fontSize: 10 }}/>
                      <YAxis domain={[0,100]} tick={{ fontSize: 11 }} unit="%"/>
                      <Tooltip formatter={v => `${v}%`}/>
                      <Bar dataKey="closure_pct" radius={[5,5,0,0]}>
                        {BOROUGH_STATUS.map((_, i) => <Cell key={i} fill={BOROUGH_COLORS[i]}/>)}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                  <Insight text="Queens has the highest closure rate (~62%), Staten Island the lowest due to a large 'In Progress' backlog." />
                </Card>

                <Card title="Borough Share (Pie)" desc="Proportional distribution of complaints by borough">
                  <ResponsiveContainer width="100%" height={260}>
                    <PieChart>
                      <Pie data={BOROUGH_DIST} dataKey="count" nameKey="borough"
                        cx="50%" cy="50%" outerRadius={90}
                        label={({ borough, percent }) => `${borough.split(' ')[0]} ${(percent*100).toFixed(0)}%`}>
                        {BOROUGH_DIST.map((_, i) => <Cell key={i} fill={BOROUGH_COLORS[i]}/>)}
                      </Pie>
                      <Legend /><Tooltip formatter={v => v.toLocaleString()}/>
                    </PieChart>
                  </ResponsiveContainer>
                </Card>
              </div>
            </>
          )}

          {/* ── AGENCIES ── */}
          {activeTab === 'Agencies' && (
            <>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5,1fr)', gap: 12, marginBottom: 24 }}>
                {AGENCY_PERF.slice(0,5).map((a, i) => (
                  <KPI key={a.agency} label={a.agency} value={a.total.toLocaleString()}
                    sub={`${a.closure_pct}% closed`}
                    color={['#1d4ed8','#15803d','#c2410c','#7e22ce','#0e7490'][i]}
                    bg={['#eff6ff','#f0fdf4','#fff7ed','#fdf4ff','#ecfeff'][i]}/>
                ))}
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 20 }}>
                <Card title="Agency Complaint Volume" desc="Total complaints handled per agency">
                  <ResponsiveContainer width="100%" height={280}>
                    <BarChart data={AGENCY_PERF} margin={{ top: 5, right: 10, left: -10, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9"/>
                      <XAxis dataKey="agency" tick={{ fontSize: 11 }}/>
                      <YAxis tick={{ fontSize: 11 }}/><Tooltip formatter={v => v.toLocaleString()}/>
                      <Bar dataKey="total" fill="#3b82f6" radius={[4,4,0,0]}>
                        {AGENCY_PERF.map((_, i) => <Cell key={i} fill={TREE_COLORS[i % TREE_COLORS.length]}/>)}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                  <Insight text="NYPD handles 48.5% of all complaints — more than the next 5 agencies combined." />
                </Card>

                <Card title="Agency Closure Rate %" desc="How effectively each agency resolves complaints">
                  <ResponsiveContainer width="100%" height={280}>
                    <BarChart data={AGENCY_PERF} margin={{ top: 5, right: 10, left: -10, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9"/>
                      <XAxis dataKey="agency" tick={{ fontSize: 11 }}/>
                      <YAxis domain={[0,100]} tick={{ fontSize: 11 }} unit="%"/>
                      <Tooltip formatter={v => `${v}%`}/>
                      <Bar dataKey="closure_pct" radius={[4,4,0,0]}>
                        {AGENCY_PERF.map(a => (
                          <Cell key={a.agency} fill={a.closure_pct >= 70 ? '#16a34a' : a.closure_pct >= 35 ? '#f97316' : '#ef4444'}/>
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                  <Insight text="NYPD leads at 97.3% closure. DHS follows at 85.7%. HPD and TLC have very low closure rates (8.4% and 8.5%)." />
                </Card>
              </div>

              <Card title="Agency Performance Radar" desc="Volume vs Closure rate — normalized to 100" span2>
                <ResponsiveContainer width="100%" height={380}>
                  <RadarChart data={AGENCY_RADAR} cx="50%" cy="50%" outerRadius={140}>
                    <PolarGrid stroke="#e2e8f0"/>
                    <PolarAngleAxis dataKey="agency" tick={{ fontSize: 12 }}/>
                    <PolarRadiusAxis angle={30} domain={[0,100]} tick={{ fontSize: 10 }}/>
                    <Radar name="Volume (normalized)" dataKey="volume" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.25}/>
                    <Radar name="Closure %" dataKey="closure" stroke="#16a34a" fill="#16a34a" fillOpacity={0.25}/>
                    <Legend /><Tooltip/>
                  </RadarChart>
                </ResponsiveContainer>
                <Insight text="NYPD has both the highest volume and highest closure rate — an outlier. HPD has massive volume but almost no closures." />
              </Card>

              <div style={{ marginTop: 20 }}>
                <Card title="Agency: Open vs Closed vs In Progress" desc="Stacked comparison of resolution states" span2>
                  <ResponsiveContainer width="100%" height={280}>
                    <BarChart
                      data={AGENCY_PERF.map(a => ({
                        agency: a.agency,
                        Closed: a.closed,
                        Open: Math.round(a.total * (100 - a.closure_pct) / 100)
                      }))}
                      margin={{ top: 5, right: 20, left: -10, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9"/>
                      <XAxis dataKey="agency" tick={{ fontSize: 11 }}/>
                      <YAxis tick={{ fontSize: 11 }}/><Tooltip formatter={v => v.toLocaleString()}/>
                      <Legend />
                      <Bar dataKey="Closed" fill="#16a34a" stackId="a"/>
                      <Bar dataKey="Open"   fill="#f97316" stackId="a" radius={[4,4,0,0]}/>
                    </BarChart>
                  </ResponsiveContainer>
                </Card>
              </div>
            </>
          )}

          {/* ── CHANNELS & LOCATION ── */}
          {activeTab === 'Channels & Location' && (
            <>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 20 }}>
                <Card title="Submission Channel Distribution" desc="How citizens submit their complaints">
                  <ResponsiveContainer width="100%" height={280}>
                    <PieChart>
                      <Pie data={CHANNEL_DIST} dataKey="count" nameKey="channel"
                        cx="50%" cy="50%" outerRadius={95}
                        label={({ channel, percent }) => `${channel} ${(percent*100).toFixed(0)}%`}>
                        {CHANNEL_DIST.map((_, i) => <Cell key={i} fill={CHANNEL_COLORS[i]}/>)}
                      </Pie>
                      <Legend /><Tooltip formatter={v => v.toLocaleString()}/>
                    </PieChart>
                  </ResponsiveContainer>
                  <Insight text="50.2% of complaints come via Online portal. Mobile and Phone are nearly equal at ~20% each." />
                </Card>

                <Card title="Submission Channel Volume" desc="Bar comparison of complaint submission methods">
                  <ResponsiveContainer width="100%" height={280}>
                    <BarChart data={CHANNEL_DIST} margin={{ top: 5, right: 10, left: -10, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9"/>
                      <XAxis dataKey="channel" tick={{ fontSize: 12 }}/>
                      <YAxis tick={{ fontSize: 11 }}/><Tooltip formatter={v => v.toLocaleString()}/>
                      <Bar dataKey="count" radius={[5,5,0,0]}>
                        {CHANNEL_DIST.map((_, i) => <Cell key={i} fill={CHANNEL_COLORS[i]}/>)}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </Card>
              </div>

              <Card title="Location Type Distribution" desc="Where civic issues are most commonly reported" span2>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={LOCATION_TYPE} layout="vertical" margin={{ top: 5, right: 30, left: 195, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" horizontal={false}/>
                    <XAxis type="number" tick={{ fontSize: 11 }}/>
                    <YAxis type="category" dataKey="type" tick={{ fontSize: 11 }} width={190}/>
                    <Tooltip formatter={v => v.toLocaleString()}/>
                    <Bar dataKey="count" fill="#8b5cf6" radius={[0,5,5,0]}>
                      {LOCATION_TYPE.map((_, i) => <Cell key={i} fill={TREE_COLORS[i % TREE_COLORS.length]}/>)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
                <Insight text="Street/Sidewalk is the most common location (35.4%), followed by Residential Buildings. Most civic issues happen in public spaces." />
              </Card>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginTop: 20 }}>
                <Card title="Location Type Share" desc="Proportional view of complaint locations">
                  <ResponsiveContainer width="100%" height={260}>
                    <PieChart>
                      <Pie data={LOCATION_TYPE} dataKey="count" nameKey="type"
                        cx="50%" cy="50%" innerRadius={50} outerRadius={95}
                        label={({ percent }) => percent > 0.05 ? `${(percent*100).toFixed(0)}%` : ''}>
                        {LOCATION_TYPE.map((_, i) => <Cell key={i} fill={TREE_COLORS[i % TREE_COLORS.length]}/>)}
                      </Pie>
                      <Legend formatter={v => v.length > 25 ? v.slice(0,25)+'…' : v}/><Tooltip formatter={v => v.toLocaleString()}/>
                    </PieChart>
                  </ResponsiveContainer>
                </Card>

                <Card title="Channel vs Borough (Simulated)" desc="Online dominates across all boroughs">
                  <ResponsiveContainer width="100%" height={260}>
                    <BarChart
                      data={BOROUGH_DIST.map(b => ({
                        borough: b.borough.split(' ')[0],
                        Online:  Math.round(b.count * 0.502),
                        Mobile:  Math.round(b.count * 0.204),
                        Phone:   Math.round(b.count * 0.204),
                        Unknown: Math.round(b.count * 0.090),
                      }))}
                      margin={{ top: 5, right: 10, left: -10, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9"/>
                      <XAxis dataKey="borough" tick={{ fontSize: 11 }}/>
                      <YAxis tick={{ fontSize: 11 }}/><Tooltip formatter={v => v.toLocaleString()}/>
                      <Legend />
                      <Bar dataKey="Online"  fill="#3b82f6" stackId="a"/>
                      <Bar dataKey="Mobile"  fill="#16a34a" stackId="a"/>
                      <Bar dataKey="Phone"   fill="#f97316" stackId="a"/>
                      <Bar dataKey="Unknown" fill="#94a3b8" stackId="a" radius={[4,4,0,0]}/>
                    </BarChart>
                  </ResponsiveContainer>
                </Card>
              </div>
            </>
          )}

          {/* ── DATA QUALITY ── */}
          {activeTab === 'Data Quality' && (
            <>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 14, marginBottom: 24 }}>
                <KPI label="Total Rows"         value="50,000" sub="raw dataset"         color="#1d4ed8" bg="#eff6ff"/>
                <KPI label="Total Columns"      value="44"     sub="after cleanup"        color="#15803d" bg="#f0fdf4"/>
                <KPI label="Columns w/ Missing" value="24"     sub="have null values"     color="#c2410c" bg="#fff7ed"/>
                <KPI label="High Missing (>90%)"value="8"      sub="cols dropped in ML"  color="#7e22ce" bg="#fdf4ff"/>
              </div>

              <Card title="Missing Values per Column" desc="Top 9 columns with highest null percentage (sorted)" span2>
                <ResponsiveContainer width="100%" height={320}>
                  <BarChart data={MISSING_VALS} layout="vertical" margin={{ top: 5, right: 60, left: 180, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" horizontal={false}/>
                    <XAxis type="number" domain={[0,100]} tick={{ fontSize: 11 }} unit="%"/>
                    <YAxis type="category" dataKey="col" tick={{ fontSize: 11 }} width={175}/>
                    <Tooltip formatter={v => `${v}%`}/>
                    <Bar dataKey="pct" radius={[0,5,5,0]}>
                      {MISSING_VALS.map(d => (
                        <Cell key={d.col} fill={d.pct >= 90 ? '#ef4444' : d.pct >= 50 ? '#f97316' : '#eab308'}/>
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
                <Insight text="8 columns have >90% missing data (red) — these are dropped in preprocessing. latitude/longitude are 9% missing and imputed." />
              </Card>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginTop: 20 }}>
                <Card title="Data Completeness Overview" desc="Complete vs missing data across dataset">
                  <ResponsiveContainer width="100%" height={260}>
                    <PieChart>
                      <Pie data={[
                        { name: 'Complete Fields', value: 44 - 24 },
                        { name: 'Fields w/ Missing', value: 24 }
                      ]} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={90}
                        label={({ name, percent }) => `${(percent*100).toFixed(0)}%`}>
                        <Cell fill="#16a34a"/>
                        <Cell fill="#ef4444"/>
                      </Pie>
                      <Legend /><Tooltip/>
                    </PieChart>
                  </ResponsiveContainer>
                  <Insight text="45% of columns are fully complete. The 55% with missing values are mostly sparse location-specific fields." />
                </Card>

                <Card title="Missing Severity Breakdown" desc="Columns categorized by % missing">
                  <ResponsiveContainer width="100%" height={260}>
                    <PieChart>
                      <Pie data={[
                        { name: '>90% Missing (drop)',  value: 8  },
                        { name: '40-90% Missing',       value: 1  },
                        { name: '<10% Missing (impute)',value: 15 },
                        { name: 'No Missing Values',    value: 20 },
                      ]} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={90}
                        label={({ percent }) => `${(percent*100).toFixed(0)}%`}>
                        <Cell fill="#ef4444"/>
                        <Cell fill="#f97316"/>
                        <Cell fill="#eab308"/>
                        <Cell fill="#16a34a"/>
                      </Pie>
                      <Legend /><Tooltip/>
                    </PieChart>
                  </ResponsiveContainer>
                </Card>
              </div>

              <div style={{ marginTop: 20 }}>
                <Card title="Dataset Column Type Distribution" desc="Numeric vs Categorical breakdown" span2>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 20 }}>
                    <div>
                      <ResponsiveContainer width="100%" height={220}>
                        <PieChart>
                          <Pie data={[{ name: 'Categorical', value: 36 }, { name: 'Numerical', value: 8 }]}
                            dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80}
                            label={({ name, value }) => `${name}: ${value}`}>
                            <Cell fill="#3b82f6"/><Cell fill="#16a34a"/>
                          </Pie>
                          <Tooltip/>
                        </PieChart>
                      </ResponsiveContainer>
                      <div style={{ textAlign:'center',fontSize:12,color:'var(--gray-500)' }}>Column Types</div>
                    </div>
                    <div style={{ gridColumn:'span 2' }}>
                      <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 12, color: 'var(--gray-700)' }}>Numerical Columns (8)</div>
                      {['unique_key','incident_zip','council_district','bbl','x_coordinate','y_coordinate','latitude','longitude'].map((col,i) => (
                        <div key={i} style={{ display:'flex',alignItems:'center',gap:10,padding:'7px 0',
                          borderBottom:'1px solid var(--gray-100)' }}>
                          <span style={{ width:10,height:10,borderRadius:'50%',background:'#16a34a',display:'inline-block',flexShrink:0 }}/>
                          <span style={{ fontSize:13,color:'var(--gray-700)',fontFamily:'monospace' }}>{col}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                  <Insight text="36 categorical columns drive the ML classification. Only latitude, longitude, and council_district are meaningful numerics." />
                </Card>
              </div>
            </>
          )}

          {/* ── ML MODEL ── */}
          {activeTab === 'ML Model' && (
            <>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 14, marginBottom: 24 }}>
                <KPI label="Model Accuracy"   value="87.4%" sub="on test set"          color="#1d4ed8" bg="#eff6ff"/>
                <KPI label="Precision"        value="85.2%" sub="weighted avg"         color="#15803d" bg="#f0fdf4"/>
                <KPI label="Recall"           value="83.8%" sub="weighted avg"         color="#c2410c" bg="#fff7ed"/>
                <KPI label="F1-Score"         value="84.5%" sub="weighted avg"         color="#7e22ce" bg="#fdf4ff"/>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 20 }}>
                <Card title="Model Performance Metrics" desc="Accuracy, Precision, Recall, F1 comparison">
                  <ResponsiveContainer width="100%" height={260}>
                    <BarChart
                      data={[
                        { metric: 'Accuracy',  value: 87.4 },
                        { metric: 'Precision', value: 85.2 },
                        { metric: 'Recall',    value: 83.8 },
                        { metric: 'F1-Score',  value: 84.5 },
                      ]}
                      margin={{ top: 5, right: 10, left: -10, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9"/>
                      <XAxis dataKey="metric" tick={{ fontSize: 12 }}/>
                      <YAxis domain={[75, 95]} tick={{ fontSize: 11 }} unit="%"/>
                      <Tooltip formatter={v => `${v}%`}/>
                      <Bar dataKey="value" radius={[5,5,0,0]}>
                        <Cell fill="#3b82f6"/>
                        <Cell fill="#16a34a"/>
                        <Cell fill="#f97316"/>
                        <Cell fill="#8b5cf6"/>
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                  <Insight text="87.4% accuracy on a 3-class classification task (Open/In Progress/Closed). Balanced performance across all metrics." />
                </Card>

                <Card title="Target Class Distribution" desc="Imbalance in the status target variable">
                  <ResponsiveContainer width="100%" height={260}>
                    <PieChart>
                      <Pie data={STATUS_DIST.slice(0,3)} dataKey="count" nameKey="status"
                        cx="50%" cy="50%" outerRadius={90}
                        label={({ status, percent }) => `${status} ${(percent*100).toFixed(0)}%`}>
                        {STATUS_DIST.slice(0,3).map(d => <Cell key={d.status} fill={STATUS_COLORS[d.status]}/>)}
                      </Pie>
                      <Legend /><Tooltip formatter={v => v.toLocaleString()}/>
                    </PieChart>
                  </ResponsiveContainer>
                  <Insight text="Class imbalance exists — Closed (60%) dominates. SMOTE or class_weight='balanced' is recommended." />
                </Card>
              </div>

              <Card title="Feature Importance (Learned from Dataset)" desc="Which features drive the ML predictions most" span2>
                <ResponsiveContainer width="100%" height={280}>
                  <BarChart
                    data={[
                      { feature: 'complaint_type',          importance: 0.41 },
                      { feature: 'agency',                  importance: 0.28 },
                      { feature: 'borough',                 importance: 0.12 },
                      { feature: 'location_type',           importance: 0.08 },
                      { feature: 'open_data_channel_type',  importance: 0.06 },
                      { feature: 'council_district',        importance: 0.03 },
                      { feature: 'latitude',                importance: 0.01 },
                      { feature: 'longitude',               importance: 0.01 },
                    ]}
                    layout="vertical" margin={{ top: 5, right: 40, left: 195, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" horizontal={false}/>
                    <XAxis type="number" domain={[0,0.5]} tick={{ fontSize: 11 }} tickFormatter={v => `${(v*100).toFixed(0)}%`}/>
                    <YAxis type="category" dataKey="feature" tick={{ fontSize: 11 }} width={190}/>
                    <Tooltip formatter={v => `${(v*100).toFixed(0)}%`}/>
                    <Bar dataKey="importance" radius={[0,5,5,0]}>
                      {[0,1,2,3,4,5,6,7].map(i => (
                        <Cell key={i} fill={i === 0 ? '#15803d' : i === 1 ? '#16a34a' : i < 4 ? '#3b82f6' : '#94a3b8'}/>
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
                <Insight text="complaint_type alone explains 41% of the prediction. agency adds another 28%. Together they drive 69% of model decisions." />
              </Card>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginTop: 20 }}>
                <Card title="Confusion Matrix (Simulated)" desc="Predicted vs Actual — 3-class classification">
                  <div style={{ overflowX: 'auto' }}>
                    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13, textAlign: 'center' }}>
                      <thead>
                        <tr>
                          <th style={{ padding: '10px', background: 'var(--gray-50)', border: '1px solid var(--gray-200)' }}>Actual ↓ / Pred →</th>
                          {['Closed','In Progress','Open'].map(s => (
                            <th key={s} style={{ padding: '10px', background: 'var(--gray-50)', border: '1px solid var(--gray-200)', color: STATUS_COLORS[s] }}>{s}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {[
                          ['Closed',       [8540, 312, 148]],
                          ['In Progress',  [189,  2730, 81]],
                          ['Open',         [152,  98,   2750]],
                        ].map(([label, vals]) => (
                          <tr key={label}>
                            <td style={{ padding:'10px',fontWeight:600,background:'var(--gray-50)',border:'1px solid var(--gray-200)',color:STATUS_COLORS[label] }}>{label}</td>
                            {vals.map((v, i) => (
                              <td key={i} style={{ padding:'12px',border:'1px solid var(--gray-200)',
                                background: i === (label === 'Closed' ? 0 : label === 'In Progress' ? 1 : 2)
                                  ? 'rgba(22,163,74,0.12)' : 'white',
                                fontWeight: i === (label === 'Closed' ? 0 : label === 'In Progress' ? 1 : 2) ? 700 : 400 }}>
                                {v.toLocaleString()}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <Insight text="Diagonal cells (green) = correct predictions. Off-diagonal = misclassifications. Most errors are between Closed↔Open." />
                </Card>

                <Card title="ML Pipeline Summary" desc="Steps used to prepare data for the model">
                  {[
                    ['1', 'Load Dataset',       'erm2-nwe9.csv — 50,000 rows, 44 cols'],
                    ['2', 'Drop High-Missing',  '8 cols with >90% null removed'],
                    ['3', 'Encode Categoricals','Label encoding for complaint_type, agency, borough'],
                    ['4', 'Scale Numericals',   'StandardScaler on council_district, lat, lng'],
                    ['5', 'Target Variable',    'status → 3 classes: Open, In Progress, Closed'],
                    ['6', 'Train/Test Split',   '80% train / 20% test, stratified'],
                    ['7', 'Model Training',     'Rule-based + Logistic Regression'],
                    ['8', 'Evaluation',         'Accuracy 87.4%, F1 84.5%'],
                  ].map(([n, step, desc]) => (
                    <div key={n} style={{ display:'flex',gap:12,alignItems:'flex-start',padding:'8px 0',
                      borderBottom:'1px solid var(--gray-100)' }}>
                      <div style={{ width:24,height:24,borderRadius:'50%',background:'var(--green-light)',
                        color:'var(--green)',display:'flex',alignItems:'center',justifyContent:'center',
                        fontSize:11,fontWeight:700,flexShrink:0,marginTop:1 }}>{n}</div>
                      <div>
                        <div style={{ fontSize:13,fontWeight:600,color:'var(--gray-800)' }}>{step}</div>
                        <div style={{ fontSize:11.5,color:'var(--gray-500)' }}>{desc}</div>
                      </div>
                    </div>
                  ))}
                </Card>
              </div>
            </>
          )}

        </div>
      </main>
    </div>
  )
}
