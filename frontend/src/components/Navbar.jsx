import { Link, useLocation } from 'react-router-dom';

export default function Navbar() {
  const { pathname } = useLocation();
  return (
    <nav className="navbar">
      <Link to="/" className="navbar-brand">
        <span className="brand-icon">🛡</span>
        InsiderWatch
      </Link>
      <ul className="navbar-links">
        <li><Link to="/"        className={pathname === '/'        ? 'active' : ''}>Home</Link></li>
        <li><Link to="/upload"  className={pathname === '/upload'  ? 'active' : ''}>Upload</Link></li>
        <li><Link to="/results" className={pathname === '/results' ? 'active' : ''}>Results</Link></li>
      </ul>
    </nav>
  );
}
