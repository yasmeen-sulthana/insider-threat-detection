import { Link } from 'react-router-dom';

const features = [
  { icon: '🔍', color: 'rgba(26,86,219,.12)', label: 'Multi-Source Analysis', desc: 'Fuses device, logon, file, email & HTTP logs into a unified behavioural profile.' },
  { icon: '🤖', color: 'rgba(6,182,212,.12)',  label: 'Deep Learning Pipeline', desc: 'Autoencoder + BiLSTM models extract hidden anomaly signals from time-series data.' },
  { icon: '🌲', color: 'rgba(16,185,129,.12)', label: 'Random Forest Classifier', desc: 'Meta-features are classified by a trained Random Forest for high-accuracy predictions.' },
  { icon: '📊', color: 'rgba(245,158,11,.12)', label: 'Interactive Visualizations', desc: 'Pie, bar & line charts instantly reveal the threat landscape across your organisation.' },
];

export default function Home() {
  return (
    <div className="home-hero">
      <div className="hero-bg" />

      <span className="hero-badge">🔒 ML-Powered Security Intelligence</span>

      <h1 className="hero-title">
        Insider Threat<br />Detection System
      </h1>

      <p className="hero-sub">
        Upload your organisation's activity logs and let our machine-learning pipeline
        identify anomalous user behaviour — before it becomes a breach.
      </p>

      <div className="hero-actions">
        <Link to="/upload" className="btn btn-primary btn-lg">
          🚀 Upload Dataset
        </Link>
        <Link to="/results" className="btn btn-outline btn-lg">
          📊 View Results
        </Link>
      </div>

      <div className="hero-features">
        {features.map((f) => (
          <div className="feature-card" key={f.label}>
            <div className="feature-icon" style={{ background: f.color }}>
              {f.icon}
            </div>
            <h3>{f.label}</h3>
            <p>{f.desc}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
