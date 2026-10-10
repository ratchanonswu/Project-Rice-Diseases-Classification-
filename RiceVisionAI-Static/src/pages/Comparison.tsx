import { useEffect, useState } from 'react';
import {
  RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, Legend,
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
} from 'recharts';
import { fetchModels, fetchMetrics } from '../api';
import type { ModelsResponse, MetricsResponse } from '../types';
import { StatCard, LoadingSpinner, ErrorBanner, SectionHeader, Badge } from '../components/UI';

const CLASS_ABBR: Record<string, string> = {
  'Bacterial Leaf Blight': 'BLB',
  'Brown Spot': 'BS',
  'Healthy Rice Leaf': 'HRL',
  'Leaf Blast': 'LB',
  'Leaf Scald': 'LS',
  'Sheath Blight': 'SB',
};

const PCT = (v: number) => `${(v * 100).toFixed(2)}%`;
const PCT1 = (v: number) => `${(v * 100).toFixed(1)}%`;

export default function Comparison() {
  const [models, setModels] = useState<ModelsResponse | null>(null);
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'overview' | 'per-class' | 'confusion'>('overview');

  useEffect(() => {
    Promise.all([fetchModels(), fetchMetrics()])
      .then(([m, me]) => { setModels(m); setMetrics(me); })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <LoadingSpinner label="Loading comparison data…" />;
  if (error) return <ErrorBanner message={error} />;

  const rn = models!.resnet50;
  const mv = models!.mobilenetv2;

  // Summary comparison rows
  const compRows = [
    { label: 'Test Accuracy', rn: rn.metrics.accuracy, mv: mv.metrics.accuracy },
    { label: 'Macro Precision', rn: rn.metrics.precision_macro, mv: mv.metrics.precision_macro },
    { label: 'Macro Recall', rn: rn.metrics.recall_macro, mv: mv.metrics.recall_macro },
    { label: 'Macro F1', rn: rn.metrics.f1_score_macro, mv: mv.metrics.f1_score_macro },
    { label: 'ROC-AUC (macro)', rn: rn.metrics.roc_auc_macro, mv: mv.metrics.roc_auc_macro },
  ];

  const effRows = [
    { label: 'Model Size (MB)', rn: rn.efficiency.model_file_size_mb, mv: mv.efficiency.model_file_size_mb, fmt: (v: number) => `${v} MB`, lowerBetter: true },
    { label: 'Total Parameters', rn: rn.efficiency.total_parameters, mv: mv.efficiency.total_parameters, fmt: (v: number) => v.toLocaleString(), lowerBetter: true },
    { label: 'Training Time (min)', rn: rn.efficiency.training_time_minutes, mv: mv.efficiency.training_time_minutes, fmt: (v: number) => `${v.toFixed(2)} min`, lowerBetter: true },
    { label: 'Avg Inference (ms)', rn: rn.efficiency.inference_latency.average_inference_time_ms, mv: mv.efficiency.inference_latency.average_inference_time_ms, fmt: (v: number) => `${v.toFixed(1)} ms`, lowerBetter: true },
    { label: 'Throughput (FPS)', rn: rn.efficiency.inference_latency.throughput_fps, mv: mv.efficiency.inference_latency.throughput_fps, fmt: (v: number) => `${v.toFixed(2)} fps`, lowerBetter: false },
  ];

  // Per-class bar chart data
  const perClassData = metrics!.class_names.map((cls) => {
    const rnCls = metrics!.resnet50.per_class.find((r) => r.class === cls);
    const mvCls = metrics!.mobilenetv2.per_class.find((r) => r.class === cls);
    return {
      class: CLASS_ABBR[cls] || cls,
      fullName: cls,
      ResNet50: rnCls ? +(rnCls.f1_score * 100).toFixed(1) : 0,
      MobileNetV2: mvCls ? +(mvCls.f1_score * 100).toFixed(1) : 0,
    };
  });

  // Radar chart data
  const radarData = compRows.map(({ label, rn: r, mv: m }) => ({
    metric: label.replace('Macro ', '').replace(' (macro)', ''),
    ResNet50: +(r * 100).toFixed(2),
    MobileNetV2: +(m * 100).toFixed(2),
  }));

  // Confusion matrix renderer
  const ConfusionMatrix = ({ model }: { model: 'resnet50' | 'mobilenetv2' }) => {
    const cm = metrics![model].confusion_matrix;
    const classNames = metrics!.class_names;
    if (!cm || Object.keys(cm).length === 0) return <p className="text-slate-400 text-sm">No confusion matrix data.</p>;

    const maxVal = Math.max(...classNames.flatMap((r) => classNames.map((c) => cm[r]?.[c] ?? 0)));

    return (
      <div className="overflow-x-auto">
        <table className="text-xs border-collapse">
          <thead>
            <tr>
              <th className="p-1 text-slate-400 font-normal text-right pr-2">Actual ↓ / Predicted →</th>
              {classNames.map((c) => (
                <th key={c} className="p-1 text-slate-600 font-medium text-center" style={{ minWidth: 54 }}>
                  {CLASS_ABBR[c]}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {classNames.map((actual) => (
              <tr key={actual}>
                <td className="pr-2 py-0.5 text-slate-600 font-medium text-right whitespace-nowrap">{CLASS_ABBR[actual]}</td>
                {classNames.map((predicted) => {
                  const val = cm[actual]?.[predicted] ?? 0;
                  const isDiag = actual === predicted;
                  const opacity = maxVal > 0 ? val / maxVal : 0;
                  const bg = isDiag
                    ? `rgba(22, 163, 74, ${0.15 + opacity * 0.7})`
                    : val > 0 ? `rgba(239, 68, 68, ${0.05 + opacity * 0.5})` : 'transparent';
                  return (
                    <td
                      key={predicted}
                      className="p-1 text-center font-mono border border-slate-100 rounded"
                      style={{ backgroundColor: bg, minWidth: 54 }}
                      title={`${actual} → ${predicted}: ${val}`}
                    >
                      <span className={isDiag ? 'font-bold text-green-800' : val > 0 ? 'text-red-700' : 'text-slate-300'}>
                        {val}
                      </span>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
        <p className="text-xs text-slate-400 mt-2">
          Abbreviations: BLB=Bacterial Leaf Blight, BS=Brown Spot, HRL=Healthy Rice Leaf, LB=Leaf Blast, LS=Leaf Scald, SB=Sheath Blight
        </p>
      </div>
    );
  };

  return (
    <div className="space-y-8">
      <SectionHeader
        title="Model Comparison"
        subtitle="ResNet50 vs MobileNetV2 · Clean Split v1 · 570 test images"
      />

      {/* Tab switcher */}
      <div className="flex gap-1 bg-slate-100 rounded-lg p-1 w-fit">
        {(['overview', 'per-class', 'confusion'] as const).map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`px-4 py-1.5 rounded-md text-sm font-medium transition-colors ${
              activeTab === tab ? 'bg-white text-green-800 shadow-sm' : 'text-slate-600 hover:text-slate-800'
            }`}
          >
            {tab === 'overview' ? 'Overview' : tab === 'per-class' ? 'Per-Class F1' : 'Confusion Matrix'}
          </button>
        ))}
      </div>

      {activeTab === 'overview' && (
        <div className="space-y-8">
          {/* Top stat cards */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <StatCard label="ResNet50 Accuracy" value={PCT1(rn.metrics.accuracy)} sub="Clean Split v1" accent />
            <StatCard label="MobileNetV2 Accuracy" value={PCT1(mv.metrics.accuracy)} sub="Clean Split v1" />
            <StatCard label="ResNet50 F1 (macro)" value={PCT1(rn.metrics.f1_score_macro)} />
            <StatCard label="MobileNetV2 F1 (macro)" value={PCT1(mv.metrics.f1_score_macro)} />
          </div>

          {/* Accuracy & metrics table */}
          <div className="bg-white border border-slate-200 rounded-xl overflow-hidden">
            <div className="px-5 py-4 border-b border-slate-100">
              <h3 className="font-semibold text-slate-800">Classification Metrics</h3>
              <p className="text-xs text-slate-400 mt-0.5">Higher is better for all metrics</p>
            </div>
            <table className="w-full text-sm">
              <thead className="bg-slate-50">
                <tr>
                  <th className="text-left px-5 py-3 font-medium text-slate-600">Metric</th>
                  <th className="text-right px-5 py-3 font-medium text-green-700">ResNet50</th>
                  <th className="text-right px-5 py-3 font-medium text-blue-700">MobileNetV2</th>
                  <th className="text-right px-5 py-3 font-medium text-slate-500">Δ</th>
                </tr>
              </thead>
              <tbody>
                {compRows.map(({ label, rn: r, mv: m }, i) => {
                  const diff = r - m;
                  return (
                    <tr key={label} className={i % 2 === 0 ? 'bg-white' : 'bg-slate-50'}>
                      <td className="px-5 py-3 text-slate-700">{label}</td>
                      <td className="px-5 py-3 text-right font-semibold text-green-700">{PCT(r)}</td>
                      <td className="px-5 py-3 text-right font-semibold text-blue-700">{PCT(m)}</td>
                      <td className={`px-5 py-3 text-right text-xs font-medium ${diff > 0 ? 'text-green-600' : 'text-red-500'}`}>
                        {diff > 0 ? '+' : ''}{(diff * 100).toFixed(2)}pp
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {/* Efficiency table */}
          <div className="bg-white border border-slate-200 rounded-xl overflow-hidden">
            <div className="px-5 py-4 border-b border-slate-100">
              <h3 className="font-semibold text-slate-800">Efficiency & Resources</h3>
              <p className="text-xs text-slate-400 mt-0.5">Measured on CPU (AMD64)</p>
            </div>
            <table className="w-full text-sm">
              <thead className="bg-slate-50">
                <tr>
                  <th className="text-left px-5 py-3 font-medium text-slate-600">Metric</th>
                  <th className="text-right px-5 py-3 font-medium text-green-700">ResNet50</th>
                  <th className="text-right px-5 py-3 font-medium text-blue-700">MobileNetV2</th>
                  <th className="text-right px-5 py-3 font-medium text-slate-500">Winner</th>
                </tr>
              </thead>
              <tbody>
                {effRows.map(({ label, rn: r, mv: m, fmt, lowerBetter }, i) => {
                  const rnWins = lowerBetter ? r < m : r > m;
                  return (
                    <tr key={label} className={i % 2 === 0 ? 'bg-white' : 'bg-slate-50'}>
                      <td className="px-5 py-3 text-slate-700">{label}</td>
                      <td className="px-5 py-3 text-right font-semibold text-green-700">{fmt(r)}</td>
                      <td className="px-5 py-3 text-right font-semibold text-blue-700">{fmt(m)}</td>
                      <td className="px-5 py-3 text-right">
                        <Badge color={rnWins ? 'green' : 'blue'}>{rnWins ? 'ResNet50' : 'MobileNetV2'}</Badge>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {/* Radar chart */}
          <div className="bg-white border border-slate-200 rounded-xl p-5">
            <h3 className="font-semibold text-slate-800 mb-4">Metric Radar</h3>
            <ResponsiveContainer width="100%" height={300}>
              <RadarChart data={radarData}>
                <PolarGrid stroke="#e2e8f0" />
                <PolarAngleAxis dataKey="metric" tick={{ fontSize: 12, fill: '#475569' }} />
                <PolarRadiusAxis angle={30} domain={[75, 100]} tick={{ fontSize: 10, fill: '#94a3b8' }} />
                <Radar name="ResNet50" dataKey="ResNet50" stroke="#15803d" fill="#15803d" fillOpacity={0.15} />
                <Radar name="MobileNetV2" dataKey="MobileNetV2" stroke="#2563eb" fill="#2563eb" fillOpacity={0.1} />
                <Legend />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {activeTab === 'per-class' && (
        <div className="space-y-6">
          <div className="bg-white border border-slate-200 rounded-xl p-5">
            <h3 className="font-semibold text-slate-800 mb-4">F1-Score by Class</h3>
            <ResponsiveContainer width="100%" height={320}>
              <BarChart data={perClassData} margin={{ top: 4, right: 16, left: 0, bottom: 4 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis dataKey="class" tick={{ fontSize: 12, fill: '#475569' }} />
                <YAxis domain={[50, 100]} tick={{ fontSize: 11, fill: '#94a3b8' }} tickFormatter={(v) => `${v}%`} />
                <Tooltip formatter={(v) => `${Number(v).toFixed(1)}%`} labelFormatter={(l, p) => (p?.[0] as any)?.payload?.fullName || l} />
                <Legend />
                <Bar dataKey="ResNet50" fill="#15803d" radius={[3, 3, 0, 0]} />
                <Bar dataKey="MobileNetV2" fill="#2563eb" radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Per-class detail table */}
          <div className="bg-white border border-slate-200 rounded-xl overflow-hidden">
            <div className="px-5 py-4 border-b border-slate-100">
              <h3 className="font-semibold text-slate-800">Per-Class Detailed Metrics</h3>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="bg-slate-50">
                  <tr>
                    <th className="text-left px-4 py-3 font-medium text-slate-600">Class</th>
                    <th className="text-right px-4 py-3 font-medium text-green-700">RN50 P</th>
                    <th className="text-right px-4 py-3 font-medium text-green-700">RN50 R</th>
                    <th className="text-right px-4 py-3 font-medium text-green-700">RN50 F1</th>
                    <th className="text-right px-4 py-3 font-medium text-blue-700">MV2 P</th>
                    <th className="text-right px-4 py-3 font-medium text-blue-700">MV2 R</th>
                    <th className="text-right px-4 py-3 font-medium text-blue-700">MV2 F1</th>
                    <th className="text-right px-4 py-3 font-medium text-slate-500">Support</th>
                  </tr>
                </thead>
                <tbody>
                  {metrics!.class_names.map((cls, i) => {
                    const r = metrics!.resnet50.per_class.find((x) => x.class === cls);
                    const m = metrics!.mobilenetv2.per_class.find((x) => x.class === cls);
                    return (
                      <tr key={cls} className={i % 2 === 0 ? 'bg-white' : 'bg-slate-50'}>
                        <td className="px-4 py-2.5 font-medium text-slate-700 whitespace-nowrap">{cls}</td>
                        <td className="px-4 py-2.5 text-right text-green-700">{r ? PCT1(r.precision) : '—'}</td>
                        <td className="px-4 py-2.5 text-right text-green-700">{r ? PCT1(r.recall) : '—'}</td>
                        <td className="px-4 py-2.5 text-right font-semibold text-green-700">{r ? PCT1(r.f1_score) : '—'}</td>
                        <td className="px-4 py-2.5 text-right text-blue-700">{m ? PCT1(m.precision) : '—'}</td>
                        <td className="px-4 py-2.5 text-right text-blue-700">{m ? PCT1(m.recall) : '—'}</td>
                        <td className="px-4 py-2.5 text-right font-semibold text-blue-700">{m ? PCT1(m.f1_score) : '—'}</td>
                        <td className="px-4 py-2.5 text-right text-slate-500">{r?.support ?? m?.support ?? '—'}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'confusion' && (
        <div className="space-y-6">
          {(['resnet50', 'mobilenetv2'] as const).map((model) => (
            <div key={model} className="bg-white border border-slate-200 rounded-xl p-6">
              <h3 className="font-semibold text-slate-800 mb-4">
                {model === 'resnet50' ? 'ResNet50' : 'MobileNetV2'} Confusion Matrix
              </h3>
              <ConfusionMatrix model={model} />
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

