import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { BarChart2, Upload, Database, Shield, Zap, ArrowRight, CheckCircle, AlertCircle } from 'lucide-react';
import { fetchHealth, fetchModels } from '../api';
import type { HealthResponse, ModelsResponse } from '../types';
import { StatCard, LoadingSpinner, ErrorBanner } from '../components/UI';

const CLASS_NAMES = [
  'Bacterial Leaf Blight',
  'Brown Spot',
  'Healthy Rice Leaf',
  'Leaf Blast',
  'Leaf Scald',
  'Sheath Blight',
];

export default function Overview() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [models, setModels] = useState<ModelsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([fetchHealth(), fetchModels()])
      .then(([h, m]) => {
        setHealth(h);
        setModels(m);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <LoadingSpinner label="Loading dashboard…" />;
  if (error) return <ErrorBanner message={`Could not reach backend: ${error}`} />;

  const rn = models?.resnet50;
  const mv = models?.mobilenetv2;

  return (
    <div className="space-y-10">
      {/* Hero */}
      <div className="rounded-2xl bg-gradient-to-br from-green-900 to-green-700 p-8 text-white">
        <div className="max-w-2xl">
          <div className="flex items-center gap-2 mb-4">
            <span className="text-green-300 text-sm font-medium uppercase tracking-widest">University Research</span>
          </div>
          <h1 className="text-3xl font-bold mb-3 leading-tight">
            Rice Leaf Disease Classification
          </h1>
          <p className="text-green-100 text-base leading-relaxed mb-6">
            A controlled comparison of <strong>ResNet50</strong> and <strong>MobileNetV2</strong> on a
            leakage-free, deduplicated dataset of 3,798 rice leaf images across six disease classes.
          </p>
          <div className="flex flex-wrap gap-3">
            <Link
              to="/predict"
              className="flex items-center gap-2 px-4 py-2 bg-white text-green-800 rounded-lg text-sm font-semibold hover:bg-green-50 transition-colors"
            >
              <Upload className="w-4 h-4" />
              Try Prediction
            </Link>
            <Link
              to="/comparison"
              className="flex items-center gap-2 px-4 py-2 bg-green-800 text-white rounded-lg text-sm font-semibold hover:bg-green-950 transition-colors"
            >
              <BarChart2 className="w-4 h-4" />
              Compare Models
            </Link>
          </div>
        </div>
      </div>

      {/* Backend status */}
      <div className="flex items-center gap-3 bg-white border border-slate-200 rounded-xl px-4 py-3">
        {health?.status === 'ok' ? (
          <CheckCircle className="w-5 h-5 text-green-600 shrink-0" />
        ) : (
          <AlertCircle className="w-5 h-5 text-orange-500 shrink-0" />
        )}
        <div className="flex flex-wrap gap-4 text-sm">
          <span className="text-slate-600">
            Backend: <span className={health?.status === 'ok' ? 'text-green-700 font-medium' : 'text-orange-600 font-medium'}>
              {health?.status === 'ok' ? 'Connected' : 'Unavailable'}
            </span>
          </span>
          <span className="text-slate-600">
            ResNet50 checkpoint:{' '}
            <span className={health?.resnet50_available ? 'text-green-700 font-medium' : 'text-red-600 font-medium'}>
              {health?.resnet50_available ? 'Found' : 'Missing'}
            </span>
          </span>
          <span className="text-slate-600">
            MobileNetV2 checkpoint:{' '}
            <span className={health?.mobilenetv2_available ? 'text-green-700 font-medium' : 'text-red-600 font-medium'}>
              {health?.mobilenetv2_available ? 'Found' : 'Missing'}
            </span>
          </span>
        </div>
      </div>

      {/* Key stats */}
      <div>
        <h2 className="text-lg font-bold text-slate-800 mb-4">Dataset — Clean Split v1</h2>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          <StatCard label="Total Images" value="3,798" sub="After deduplication" />
          <StatCard label="Training" value="2,847" sub="74.96%" />
          <StatCard label="Validation" value="381" sub="10.03%" />
          <StatCard label="Test" value="570" sub="15.01%" />
        </div>
      </div>

      {/* Model comparison summary */}
      <div>
        <h2 className="text-lg font-bold text-slate-800 mb-4">Model Results — Test Set</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {/* ResNet50 */}
          <div className="bg-white border border-slate-200 rounded-xl p-6">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h3 className="font-bold text-slate-800 text-lg">ResNet50</h3>
                <p className="text-xs text-slate-500">Transfer learning · frozen backbone</p>
              </div>
              <span className="text-2xl font-bold text-green-700">
                {rn ? `${(rn.metrics.accuracy * 100).toFixed(1)}%` : '—'}
              </span>
            </div>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between text-slate-600">
                <span>Macro F1</span>
                <span className="font-medium">{rn ? `${(rn.metrics.f1_score_macro * 100).toFixed(2)}%` : '—'}</span>
              </div>
              <div className="flex justify-between text-slate-600">
                <span>ROC-AUC</span>
                <span className="font-medium">{rn ? `${(rn.metrics.roc_auc_macro * 100).toFixed(2)}%` : '—'}</span>
              </div>
              <div className="flex justify-between text-slate-600">
                <span>Model Size</span>
                <span className="font-medium">{rn ? `${rn.efficiency.model_file_size_mb} MB` : '—'}</span>
              </div>
              <div className="flex justify-between text-slate-600">
                <span>Avg Inference</span>
                <span className="font-medium">
                  {rn ? `${rn.efficiency.inference_latency.average_inference_time_ms.toFixed(0)} ms` : '—'}
                </span>
              </div>
            </div>
          </div>

          {/* MobileNetV2 */}
          <div className="bg-white border border-slate-200 rounded-xl p-6">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h3 className="font-bold text-slate-800 text-lg">MobileNetV2</h3>
                <p className="text-xs text-slate-500">Transfer learning · frozen backbone</p>
              </div>
              <span className="text-2xl font-bold text-blue-700">
                {mv ? `${(mv.metrics.accuracy * 100).toFixed(1)}%` : '—'}
              </span>
            </div>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between text-slate-600">
                <span>Macro F1</span>
                <span className="font-medium">{mv ? `${(mv.metrics.f1_score_macro * 100).toFixed(2)}%` : '—'}</span>
              </div>
              <div className="flex justify-between text-slate-600">
                <span>ROC-AUC</span>
                <span className="font-medium">{mv ? `${(mv.metrics.roc_auc_macro * 100).toFixed(2)}%` : '—'}</span>
              </div>
              <div className="flex justify-between text-slate-600">
                <span>Model Size</span>
                <span className="font-medium">{mv ? `${mv.efficiency.model_file_size_mb} MB` : '—'}</span>
              </div>
              <div className="flex justify-between text-slate-600">
                <span>Avg Inference</span>
                <span className="font-medium">
                  {mv ? `${mv.efficiency.inference_latency.average_inference_time_ms.toFixed(0)} ms` : '—'}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* 6 classes grid */}
      <div>
        <h2 className="text-lg font-bold text-slate-800 mb-4">Six Classification Classes</h2>
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
          {CLASS_NAMES.map((name, i) => (
            <Link
              key={name}
              to="/classes"
              className="flex items-center gap-3 bg-white border border-slate-200 rounded-xl p-4 hover:border-green-300 hover:shadow-sm transition-all group"
            >
              <div className="w-8 h-8 rounded-full bg-green-100 text-green-800 flex items-center justify-center text-sm font-bold shrink-0">
                {i}
              </div>
              <span className="text-sm font-medium text-slate-700 group-hover:text-green-800 leading-tight">{name}</span>
            </Link>
          ))}
        </div>
      </div>

      {/* Feature highlights */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        {[
          {
            icon: Shield,
            title: 'Leakage-Safe Splits',
            desc: 'Exact and near-duplicate detection performed before splitting. 31 duplicate groups excluded.',
            color: 'text-green-700',
            bg: 'bg-green-50',
          },
          {
            icon: Database,
            title: 'Reproducible Pipeline',
            desc: 'All split manifests, audit logs, and contact sheets preserved for full traceability.',
            color: 'text-blue-700',
            bg: 'bg-blue-50',
          },
          {
            icon: Zap,
            title: 'Live Inference',
            desc: 'Upload any rice leaf image and receive real predictions from both trained models.',
            color: 'text-orange-600',
            bg: 'bg-orange-50',
          },
        ].map(({ icon: Icon, title, desc, color, bg }) => (
          <div key={title} className="bg-white border border-slate-200 rounded-xl p-5">
            <div className={`w-9 h-9 rounded-lg ${bg} flex items-center justify-center mb-3`}>
              <Icon className={`w-5 h-5 ${color}`} />
            </div>
            <h3 className="font-semibold text-slate-800 mb-1.5">{title}</h3>
            <p className="text-sm text-slate-500 leading-relaxed">{desc}</p>
          </div>
        ))}
      </div>

      {/* CTA */}
      <div className="flex justify-end">
        <Link
          to="/comparison"
          className="flex items-center gap-2 text-sm font-medium text-green-700 hover:text-green-900 transition-colors"
        >
          See full model comparison <ArrowRight className="w-4 h-4" />
        </Link>
      </div>
    </div>
  );
}

