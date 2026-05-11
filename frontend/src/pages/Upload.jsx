import { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { runModel, getProgress } from '../api/api';

const FILE_SLOTS = [
  { key: 'device', label: 'device.csv', icon: '💻', desc: 'Device usage logs' },
  { key: 'logon', label: 'logon.csv', icon: '🔑', desc: 'Login/logoff events' },
  { key: 'file', label: 'file.csv', icon: '📁', desc: 'File access records' },
  { key: 'email', label: 'email.csv', icon: '📧', desc: 'Email activity data' },
  { key: 'http', label: 'http.csv', icon: '🌐', desc: 'Web browsing history' },
];

const STEPS = [
  'Loading datasets',
  'Engineering features',
  'Training Autoencoder',
  'Training BiLSTM',
  'Classifying users',
  'Aggregating results',
];

export default function Upload() {

  const navigate = useNavigate();

  const [files, setFiles] = useState({});

  const [running, setRunning] = useState(false);

  const [progress, setProgress] = useState({
    step: 0,
    total: 6,
    message: 'Idle',
  });

  const [error, setError] = useState('');

  const [drag, setDrag] = useState('');

  const pollRef = useRef(null);

  // =========================
  // HANDLE FILE SELECT
  // =========================
  const handleFileChange = (key, e) => {

    const file = e.target.files[0];

    if (!file) return;

    setFiles((prev) => ({
      ...prev,
      [key]: file,
    }));

    setError('');
  };

  // =========================
  // HANDLE DRAG DROP
  // =========================
  const handleDrop = (key, e) => {

    e.preventDefault();

    setDrag('');

    const file = e.dataTransfer.files[0];

    if (!file) return;

    setFiles((prev) => ({
      ...prev,
      [key]: file,
    }));
  };

  // =========================
  // POLLING
  // =========================
  const startPolling = () => {

    pollRef.current = setInterval(async () => {

      try {

        const p = await getProgress();

        setProgress(p);

        if (p.step >= p.total) {

          clearInterval(pollRef.current);

        }

      } catch (_) { }

    }, 1200);
  };

  // =========================
  // RUN MODEL
  // =========================
  const handleRunModel = async () => {

    setRunning(true);

    setError('');

    setProgress({
      step: 0,
      total: 6,
      message: 'Starting pipeline...',
    });

    startPolling();

    try {

      const data = await runModel();

      clearInterval(pollRef.current);

      setProgress({
        step: 6,
        total: 6,
        message: 'Done!',
      });

      sessionStorage.setItem(
        'insiderResults',
        JSON.stringify(data)
      );

      setTimeout(() => navigate('/results'), 600);

    } catch (e) {

      clearInterval(pollRef.current);

      setError(
        'Model run failed: ' +
        (e.response?.data?.error || e.message)
      );

      setRunning(false);
    }
  };

  const readyCount = Object.values(files).filter(Boolean).length;

  const pct = running
    ? Math.round((progress.step / progress.total) * 100)
    : 0;

  return (

    <div className="page-container">

      <div className="page-header">

        <h1>📂 Upload Datasets</h1>

        <p>
          Select CSV activity log files and run the insider threat detection pipeline.
        </p>

      </div>

      {error && (
        <div className="error-banner">
          ⚠️ {error}
        </div>
      )}

      {/* ===================== */}
      {/* PROGRESS */}
      {/* ===================== */}
      {running && (

        <div className="progress-container">

          <div className="progress-header">

            <span>{progress.message}</span>

            <span>{pct}%</span>

          </div>

          <div className="progress-bar-track">

            <div
              className="progress-bar-fill"
              style={{ width: `${pct}%` }}
            />

          </div>

          <div className="progress-steps">

            {STEPS.map((s, i) => (

              <span
                key={s}
                className={`progress-step ${i < progress.step
                  ? 'done'
                  : i === progress.step - 1
                    ? 'active'
                    : ''
                  }`}
              >
                {i < progress.step ? '✓' : i + 1} {s}
              </span>

            ))}

          </div>

        </div>

      )}

      {/* ===================== */}
      {/* FILE GRID */}
      {/* ===================== */}
      <div className="file-grid">

        {FILE_SLOTS.map(({ key, label, icon, desc }) => {

          const fileName = files[key]?.name;

          return (

            <div
              key={key}
              className={`file-drop-zone ${drag === key ? 'drag-over' : ''
                }`}
              onDragOver={(e) => {
                e.preventDefault();
                setDrag(key);
              }}
              onDragLeave={() => setDrag('')}
              onDrop={(e) => handleDrop(key, e)}
            >

              <input
                type="file"
                accept=".csv"
                id={`file-${key}`}
                onChange={(e) =>
                  handleFileChange(key, e)
                }
              />

              <div className="file-drop-icon">
                {icon}
              </div>

              <div className="file-label">
                {label}
              </div>

              <div className="file-hint">
                {desc}
              </div>

              {fileName && (
                <div className="file-name">
                  📎 {fileName}
                </div>
              )}

            </div>
          );
        })}
      </div>

      {/* ===================== */}
      {/* SUMMARY */}
      {/* ===================== */}
      <div className="upload-summary">

        <span>
          Files selected:
          <strong> {readyCount} </strong>/ 5
        </span>

      </div>

      {/* ===================== */}
      {/* BUTTON */}
      {/* ===================== */}
      <div
        className="upload-actions"
        style={{ marginTop: '1.5rem' }}
      >

        <button
          className="btn btn-success btn-lg"
          onClick={handleRunModel}
          disabled={running}
        >

          {running ? (
            <>
              <span className="spinner" />
              Running Pipeline...
            </>
          ) : (
            '▶ Run Detection'
          )}

        </button>

      </div>

    </div>
  );
}