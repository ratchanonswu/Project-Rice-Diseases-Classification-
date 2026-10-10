import { Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Overview from './pages/Overview';
import Comparison from './pages/Comparison';
import Classes from './pages/Classes';
import Predict from './pages/Predict';
import Methodology from './pages/Methodology';

export default function App() {
  return (
    <div className="min-h-screen bg-slate-50">
      <Navbar />
      <main className="pt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 py-8">
          <Routes>
            <Route path="/" element={<Overview />} />
            <Route path="/comparison" element={<Comparison />} />
            <Route path="/classes" element={<Classes />} />
            <Route path="/predict" element={<Predict />} />
            <Route path="/methodology" element={<Methodology />} />
          </Routes>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-200 bg-white mt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 py-6 flex flex-col sm:flex-row items-center justify-between gap-2">
          <p className="text-xs text-slate-400">
            RiceVision AI · Rice Leaf Disease Classification Research · Clean Split v1
          </p>
          <p className="text-xs text-slate-400">
            Educational research prototype · Not for diagnostic use
          </p>
        </div>
      </footer>
    </div>
  );
}
