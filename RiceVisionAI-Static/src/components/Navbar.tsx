import { NavLink } from 'react-router-dom';
import { Leaf, BarChart2, Grid, Upload, BookOpen, Activity } from 'lucide-react';

const navItems = [
  { to: '/', label: 'Overview', icon: Activity, end: true },
  { to: '/comparison', label: 'Model Comparison', icon: BarChart2 },
  { to: '/classes', label: 'Disease Classes', icon: Grid },
  { to: '/predict', label: 'Predict', icon: Upload },
  { to: '/methodology', label: 'Methodology', icon: BookOpen },
];

export default function Navbar() {
  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-white border-b border-slate-200 shadow-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 flex items-center h-16 gap-8">
        {/* Logo */}
        <NavLink to="/" className="flex items-center gap-2 shrink-0">
          <div className="w-8 h-8 rounded-lg bg-green-700 flex items-center justify-center">
            <Leaf className="w-4 h-4 text-white" />
          </div>
          <span className="font-bold text-green-900 text-lg tracking-tight">RiceVision AI</span>
        </NavLink>

        {/* Nav links */}
        <div className="flex items-center gap-1 overflow-x-auto hide-scrollbar">
          {navItems.map(({ to, label, icon: Icon, end }) => (
            <NavLink
              key={to}
              to={to}
              end={end}
              className={({ isActive }) =>
                `flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium transition-colors whitespace-nowrap ${
                  isActive
                    ? 'bg-green-50 text-green-800'
                    : 'text-slate-600 hover:text-green-800 hover:bg-slate-50'
                }`
              }
            >
              <Icon className="w-4 h-4" />
              {label}
            </NavLink>
          ))}
        </div>

        {/* Research badge */}
        <div className="ml-auto shrink-0">
          <span className="text-xs bg-green-100 text-green-800 px-2.5 py-1 rounded-full font-medium">
            Research Prototype
          </span>
        </div>
      </div>
    </nav>
  );
}

