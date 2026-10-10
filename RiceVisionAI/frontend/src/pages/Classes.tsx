import { useEffect, useState } from 'react';
import { fetchClasses } from '../api';
import type { ClassInfo } from '../types';
import { LoadingSpinner, ErrorBanner, SectionHeader } from '../components/UI';
import { Leaf, ImageOff } from 'lucide-react';

const CLASS_COLORS: Record<string, string> = {
  'Bacterial Leaf Blight': 'border-yellow-300 bg-yellow-50',
  'Brown Spot': 'border-amber-300 bg-amber-50',
  'Healthy Rice Leaf': 'border-green-300 bg-green-50',
  'Leaf Blast': 'border-red-300 bg-red-50',
  'Leaf Scald': 'border-orange-300 bg-orange-50',
  'Sheath Blight': 'border-purple-300 bg-purple-50',
};

const CLASS_BADGE: Record<string, string> = {
  'Bacterial Leaf Blight': 'bg-yellow-100 text-yellow-800',
  'Brown Spot': 'bg-amber-100 text-amber-800',
  'Healthy Rice Leaf': 'bg-green-100 text-green-800',
  'Leaf Blast': 'bg-red-100 text-red-800',
  'Leaf Scald': 'bg-orange-100 text-orange-800',
  'Sheath Blight': 'bg-purple-100 text-purple-800',
};

export default function Classes() {
  const [classes, setClasses] = useState<ClassInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<ClassInfo | null>(null);

  useEffect(() => {
    fetchClasses()
      .then((data) => { setClasses(data); setSelected(data[0] ?? null); })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <LoadingSpinner label="Loading disease classes…" />;
  if (error) return <ErrorBanner message={error} />;

  return (
    <div className="space-y-6">
      <SectionHeader
        title="Disease Classes"
        subtitle="Six categories classified in this study · descriptions are general educational references"
      />

      <div className="flex flex-col lg:flex-row gap-6">
        {/* Sidebar list */}
        <div className="lg:w-64 shrink-0 space-y-2">
          {classes.map((cls) => (
            <button
              key={cls.name}
              onClick={() => setSelected(cls)}
              className={`w-full text-left px-4 py-3 rounded-xl border transition-all flex items-center gap-3 ${
                selected?.name === cls.name
                  ? `${CLASS_COLORS[cls.name] || 'border-green-300 bg-green-50'} shadow-sm`
                  : 'bg-white border-slate-200 hover:border-slate-300'
              }`}
            >
              <span className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold shrink-0 ${CLASS_BADGE[cls.name] || 'bg-slate-100 text-slate-600'}`}>
                {cls.index}
              </span>
              <span className="text-sm font-medium text-slate-700 leading-tight">{cls.name}</span>
            </button>
          ))}
        </div>

        {/* Detail panel */}
        {selected && (
          <div className="flex-1 bg-white border border-slate-200 rounded-2xl overflow-hidden">
            {/* Image */}
            <div className="aspect-video bg-slate-100 relative flex items-center justify-center overflow-hidden">
              {selected.image ? (
                <img
                  src={selected.image}
                  alt={selected.name}
                  className="w-full h-full object-cover"
                />
              ) : (
                <div className="flex flex-col items-center gap-2 text-slate-400">
                  <ImageOff className="w-10 h-10" />
                  <span className="text-sm">No representative image available</span>
                </div>
              )}
              <div className="absolute bottom-3 left-3">
                <span className={`px-3 py-1 rounded-full text-sm font-semibold ${CLASS_BADGE[selected.name] || 'bg-slate-100 text-slate-700'}`}>
                  Class {selected.index} · {selected.name}
                </span>
              </div>
            </div>

            {/* Content */}
            <div className="p-6">
              <div className="flex items-center gap-3 mb-4">
                <div className={`w-10 h-10 rounded-xl flex items-center justify-center ${CLASS_COLORS[selected.name] || 'bg-green-50'}`}>
                  <Leaf className="w-5 h-5 text-green-700" />
                </div>
                <div>
                  <h2 className="font-bold text-slate-800 text-xl">{selected.name}</h2>
                  <p className="text-xs text-slate-400">Index {selected.index} of 5 (0-based)</p>
                </div>
              </div>

              <p className="text-slate-600 leading-relaxed mb-6">{selected.description}</p>

              <div className="rounded-lg bg-slate-50 border border-slate-200 px-4 py-3 text-xs text-slate-500">
                ℹ Descriptions are general educational references compiled for this research prototype. 
                They are not sourced from the training dataset labels or model outputs.
              </div>
            </div>
          </div>
        )}
      </div>

      {/* All classes grid view */}
      <div>
        <h3 className="font-semibold text-slate-700 mb-4 mt-6">All Classes at a Glance</h3>
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
          {classes.map((cls) => (
            <button
              key={cls.name}
              onClick={() => setSelected(cls)}
              className="bg-white border border-slate-200 rounded-xl overflow-hidden hover:border-green-300 hover:shadow-md transition-all text-left group"
            >
              <div className="aspect-video bg-slate-100 overflow-hidden">
                {cls.image ? (
                  <img src={cls.image} alt={cls.name} className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300" />
                ) : (
                  <div className="w-full h-full flex items-center justify-center text-slate-300">
                    <ImageOff className="w-8 h-8" />
                  </div>
                )}
              </div>
              <div className="p-3">
                <span className={`text-xs font-medium px-2 py-0.5 rounded ${CLASS_BADGE[cls.name] || 'bg-slate-100 text-slate-600'}`}>
                  Class {cls.index}
                </span>
                <p className="mt-1.5 text-sm font-semibold text-slate-700 leading-tight">{cls.name}</p>
              </div>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

