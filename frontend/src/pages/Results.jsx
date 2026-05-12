import { useState, useEffect, useMemo } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  PieChart,
  Pie,
  Cell,
  Tooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  LineChart,
  Line,
} from 'recharts';

const COLORS = {
  normal: '#10b981',
  threat: '#ef4444',
};

const PAGE_SIZE = 10;

export default function Results() {

  const navigate = useNavigate();
  const location = useLocation();

  const [data, setData] = useState(null);

  const [search, setSearch] = useState('');

  const [filter, setFilter] = useState('all');

  const [sortKey, setSortKey] = useState('score');

  const [sortDir, setSortDir] = useState('desc');

  const [page, setPage] = useState(1);

  // =====================================
  // LOAD RESULTS
  // =====================================
  useEffect(() => {

    if (location.state && location.state.insiderResults) {
      setData(location.state.insiderResults);
    } else {
      const stored = sessionStorage.getItem('insiderResults');
      if (stored) {
        try {
          setData(JSON.parse(stored));
        } catch (e) {
          console.error('Failed to parse stored results', e);
        }
      }
    }

  }, [location.state]);

  // =====================================
  // PIE CHART DATA
  // =====================================
  const pieData = data
    ? [
      {
        name: 'Normal',
        value: data.normal_users,
      },
      {
        name: 'Threat',
        value: data.threat_users,
      },
    ]
    : [];

  // =====================================
  // BAR CHART DATA
  // =====================================
  const barData = useMemo(() => {

    if (!data) return [];

    const topUsers = [...data.user_results]
      .sort(
        (a, b) =>
          (b.activity_total || 0) -
          (a.activity_total || 0)
      )
      .slice(0, 15);

    return topUsers.map((u) => ({
      user:
        u.user_id.length > 8
          ? u.user_id.slice(0, 8) + '…'
          : u.user_id,

      activity: u.activity_total || 0,

      fill:
        u.prediction === 1
          ? COLORS.threat
          : COLORS.normal,
    }));

  }, [data]);

  // =====================================
  // COMPARISON CHART DATA
  // =====================================
  const comparisonData = useMemo(() => {
    if (!data || !data.model_accuracies) return [];
    return Object.entries(data.model_accuracies).map(([name, acc]) => ({
      name,
      accuracy: Number((acc * 100).toFixed(1))
    })).sort((a, b) => b.accuracy - a.accuracy);
  }, [data]);

  // =====================================
  // FILTER + SEARCH + SORT
  // =====================================
  const filteredRows = useMemo(() => {

    if (!data) return [];

    return data.user_results
      .filter((u) => {

        const matchSearch =
          u.user_id
            .toLowerCase()
            .includes(search.toLowerCase());

        const matchFilter =
          filter === 'all' ||
          (filter === 'threat'
            ? u.prediction === 1
            : u.prediction === 0);

        return matchSearch && matchFilter;

      })
      .sort((a, b) => {

        const va = a[sortKey] ?? 0;

        const vb = b[sortKey] ?? 0;

        return sortDir === 'asc'
          ? va > vb
            ? 1
            : -1
          : va < vb
            ? 1
            : -1;

      });

  }, [data, search, filter, sortKey, sortDir]);

  // =====================================
  // PAGINATION
  // =====================================
  const totalPages = Math.ceil(
    filteredRows.length / PAGE_SIZE
  );

  const paginatedRows = filteredRows.slice(
    (page - 1) * PAGE_SIZE,
    page * PAGE_SIZE
  );

  // =====================================
  // SORT FUNCTION
  // =====================================
  const handleSort = (key) => {

    if (sortKey === key) {

      setSortDir((d) =>
        d === 'asc' ? 'desc' : 'asc'
      );

    } else {

      setSortKey(key);

      setSortDir('desc');

    }

  };

  // =====================================
  // DOWNLOAD CSV
  // =====================================
  const downloadCSV = () => {

    const header =
      'User ID,Prediction,Label,Score,Total Activity';

    const rows = data.user_results.map(
      (u) =>
        `${u.user_id},${u.prediction},${u.label},${u.score},${u.activity_total || 0}`
    );

    const blob = new Blob(
      [[header, ...rows].join('\n')],
      { type: 'text/csv' }
    );

    const url = URL.createObjectURL(blob);

    const a = document.createElement('a');

    a.href = url;

    a.download = 'insider_threat_results.csv';

    a.click();

  };

  // =====================================
  // EMPTY STATE
  // =====================================
  if (!data) {

    return (
      <div className="page-container">

        <div className="empty-state">

          <div className="empty-icon">📭</div>

          <h2>No Results Yet</h2>

          <p>
            Run the detection pipeline first
            to see results here.
          </p>

          <br />

          <button
            className="btn btn-primary"
            onClick={() => navigate('/upload')}
          >
            Go to Upload
          </button>

        </div>

      </div>
    );

  }

  // =====================================
  // MAIN UI
  // =====================================
  return (

    <div
      className="page-container"
      style={{ maxWidth: 1100 }}
    >

      {/* HEADER */}
      <div
        className="page-header"
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'flex-start',
          flexWrap: 'wrap',
          gap: '1rem',
        }}
      >

        <div>

          <h1>🔍 Detection Results</h1>

          <p>
            ML pipeline complete — review
            identified threats below.
          </p>

        </div>

        <div
          style={{
            display: 'flex',
            gap: '.7rem',
          }}
        >

          <button
            className="btn btn-outline"
            onClick={() => navigate('/upload')}
          >
            ↩ Re-run
          </button>

          <button
            className="btn btn-success"
            onClick={downloadCSV}
          >
            ⬇ Download CSV
          </button>

        </div>

      </div>

      {/* STATS CARDS */}
      <div className="stats-grid">

        <div className="stat-card total">

          <span className="stat-icon">👥</span>

          <div className="stat-label">
            Unique Users
          </div>

          <div className="stat-value">
            {(data.unique_users || 0).toLocaleString()}
          </div>

        </div>

        <div className="stat-card normal">

          <span className="stat-icon">✅</span>

          <div className="stat-label">
            Normal Users
          </div>

          <div className="stat-value">
            {(data.normal_users || 0).toLocaleString()}
          </div>

        </div>

        <div className="stat-card threat">

          <span className="stat-icon">⚠️</span>

          <div className="stat-label">
            Threat Users
          </div>

          <div className="stat-value">
            {(data.threat_users || 0).toLocaleString()}
          </div>

        </div>

        <div className="stat-card accuracy">

          <span className="stat-icon">🎯</span>

          <div className="stat-label">
            {data.best_model ? `Random Forest Accuracy` : 'Model Accuracy'}
          </div>

          <div className="stat-value">
            {(
              (data.model_accuracy || 0) * 100
            ).toFixed(1)}
            %
          </div>

        </div>

      </div>

      {/* DATASET COUNTS TABLE */}
      {data.file_counts && (
        <div className="table-section" style={{ marginBottom: '2.5rem' }}>
          <div className="table-header">
            <h3>Dataset Record Counts</h3>
          </div>
          <div style={{ overflowX: 'auto' }}>
            <table className="results-table">
              <thead>
                <tr>
                  <th>Dataset</th>
                  <th>Record Count</th>
                </tr>
              </thead>
              <tbody>
                {['device', 'logon', 'file', 'email', 'http'].map((ds) => (
                  <tr key={ds}>
                    <td style={{ textTransform: 'capitalize', fontWeight: 600 }}>{ds}</td>
                    <td>{(data.file_counts[ds] || 0).toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* CHARTS */}
      <div className="charts-grid">
        <div className="chart-card">
          <h3>Users Distribution</h3>
          <ResponsiveContainer width="100%" height={280}>
            <PieChart>
              <Pie
                data={pieData}
                dataKey="value"
                nameKey="name"
                cx="50%"
                cy="50%"
                innerRadius={60}
                outerRadius={80}
                paddingAngle={5}
              >
                {pieData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[entry.name.toLowerCase()]} />
                ))}
              </Pie>
              <Tooltip />
              <Legend verticalAlign="bottom" height={36} />
            </PieChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-card">
          <h3>Top Users by Activity Count</h3>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={barData}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="user" />
              <YAxis />
              <Tooltip cursor={{ fill: 'transparent' }} />
              <Bar dataKey="activity" radius={[4, 4, 0, 0]}>
                {barData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.fill} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {data.trend_data && data.trend_data.length > 0 && (
          <div className="chart-card chart-full">
            <h3>Activity Trends (Last 30 Days)</h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={data.trend_data}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="date" />
                <YAxis />
                <Tooltip />
                <Line
                  type="monotone"
                  dataKey="activity"
                  stroke="#1a56db"
                  strokeWidth={2}
                  dot={false}
                  activeDot={{ r: 6 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}

        {comparisonData.length > 0 && (
          <div className="chart-card chart-full">
            <h3>Meta Classifiers Accuracy Comparison</h3>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={comparisonData} layout="vertical" margin={{ left: 50, right: 20 }}>
                <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                <XAxis type="number" domain={[0, 100]} />
                <YAxis dataKey="name" type="category" width={100} tick={{ fontSize: 12 }} />
                <Tooltip formatter={(val) => `${val}%`} />
                <Bar dataKey="accuracy" fill="var(--primary)" barSize={24} radius={[0, 4, 4, 0]}>
                  {comparisonData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={index === 0 ? 'var(--success)' : 'var(--primary)'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      {/* TABLE */}
      <div className="table-section">
        <div className="table-header">
          <h3>Detailed User Predictions</h3>
          <div className="table-controls">
            <input
              type="text"
              placeholder="Search user ID..."
              className="search-input"
              value={search}
              onChange={(e) => {
                setSearch(e.target.value);
                setPage(1);
              }}
            />
            <select
              className="filter-select"
              value={filter}
              onChange={(e) => {
                setFilter(e.target.value);
                setPage(1);
              }}
            >
              <option value="all">All Users</option>
              <option value="threat">Threats</option>
              <option value="normal">Normal</option>
            </select>
          </div>
        </div>

        <div style={{ overflowX: 'auto' }}>
          <table className="results-table">
            <thead>
              <tr>
                <th onClick={() => handleSort('user_id')}>User ID</th>
                <th onClick={() => handleSort('prediction')}>Prediction</th>
                <th onClick={() => handleSort('activity_total')}>Activity Total</th>
                <th onClick={() => handleSort('score')}>Risk Score</th>
              </tr>
            </thead>
            <tbody>
              {paginatedRows.length === 0 ? (
                <tr>
                  <td colSpan={4} style={{ textAlign: 'center', padding: '2rem' }}>
                    No matching users found.
                  </td>
                </tr>
              ) : (
                paginatedRows.map((user, i) => (
                  <tr key={i}>
                    <td style={{ fontWeight: 600 }}>{user.user_id}</td>
                    <td>
                      <div className={`prediction-badge ${user.prediction === 1 ? 'pred-threat' : 'pred-normal'}`}>
                        {user.prediction === 1 ? '⚠️ Threat' : '✅ Normal'}
                      </div>
                    </td>
                    <td>{user.activity_total?.toLocaleString() || 0}</td>
                    <td>
                      <div className="score-bar">
                        <span style={{ fontSize: '.85rem', width: '35px' }}>
                          {((user.score || 0) * 100).toFixed(0)}%
                        </span>
                        <div className="score-track">
                          <div
                            className="score-fill"
                            style={{
                              width: `${(user.score || 0) * 100}%`,
                              background: user.prediction === 1 ? 'var(--danger)' : 'var(--success)'
                            }}
                          ></div>
                        </div>
                      </div>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>

        {/* PAGINATION */}
        {totalPages > 1 && (
          <div className="pagination">
            <span>
              Showing {(page - 1) * PAGE_SIZE + 1} to {Math.min(page * PAGE_SIZE, filteredRows.length)} of {filteredRows.length} users
            </span>
            <div className="pagination-btns">
              <button
                className="page-btn"
                onClick={() => setPage(p => Math.max(1, p - 1))}
                disabled={page === 1}
              >
                &lsaquo;
              </button>
              {Array.from({ length: totalPages }, (_, i) => i + 1)
                .filter(p => p === 1 || p === totalPages || Math.abs(page - p) <= 1)
                .map((p, i, arr) => {
                  const out = [];
                  if (i > 0 && arr[i - 1] !== p - 1) {
                    out.push(<span key={`ellipsis-${p}`} style={{ padding: '0 .3rem', color: '#cbd5e1' }}>...</span>);
                  }
                  out.push(
                    <button
                      key={`page-${p}`}
                      className={`page-btn ${page === p ? 'active' : ''}`}
                      onClick={() => setPage(p)}
                    >
                      {p}
                    </button>
                  );
                  return out;
                })}
              <button
                className="page-btn"
                onClick={() => setPage(p => Math.min(totalPages, p + 1))}
                disabled={page === totalPages}
              >
                &rsaquo;
              </button>
            </div>
          </div>
        )}
      </div>

    </div>
  );
}