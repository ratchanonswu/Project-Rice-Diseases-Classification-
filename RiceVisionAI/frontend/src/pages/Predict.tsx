import { useState, useCallback, useRef } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { predictImage } from '../api';
import type { PredictResponse } from '../types';
import { SectionHeader, ErrorBanner } from '../components/UI';
import { Upload, ImageIcon, Loader2, AlertTriangle, CheckCircle, X } from 'lucide-react';

type ModelKey = 'resnet50' | 'mobilenetv2';

const MODEL_LABELS: Record<ModelKey, string> = {
  resnet50: 'ResNet50',
  mobilenetv2: 'MobileNetV2',
};

const CLASS_COLORS_MAP: Record<string, string> = {
  'Bacterial Leaf Blight': '#ca8a04',
  'Brown Spot': '#d97706',
  'Healthy Rice Leaf': '#16a34a',
  'Leaf Blast': '#dc2626',
  'Leaf Scald': '#ea580c',
  'Sheath Blight': '#7c3aed',
};

export default function Predict() {
  const [model, setModel] = useState<ModelKey>('resnet50');
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback((f: File) => {
    if (!f.type.startsWith('image/')) {
      setError('Please select a JPEG or PNG image file.');
      return;
    }
    if (f.size > 10 * 1024 * 1024) {
      setError('Image must be smaller than 10 MB.');
      return;
    }
    setFile(f);
    setError(null);
    setResult(null);
    const url = URL.createObjectURL(f);
    setPreview(url);
  }, []);

  const onDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  }, [handleFile]);

  const onInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) handleFile(f);
  };

  const clearImage = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    if (inputRef.current) inputRef.current.value = '';
  };

  const handlePredict = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await predictImage(file, model);
      setResult(res);
    } catch (e: any) {
      setError(e.message || 'Prediction failed');
    } finally {
      setLoading(false);
    }
  };

  // Sort scores for chart
  const chartData = result
    ? Object.entries(result.scores)
        .map(([cls, score]) => ({ class: cls, score: +(score * 100).toFixed(1) }))
        .sort((a, b) => b.score - a.score)
    : [];

  return (
    <div className="space-y-6">
      <SectionHeader
        title="Image Prediction"
        subtitle="Upload a rice leaf image and receive a real-time classification from the selected model"
      />

      {/* Model selector */}
      <div className="flex gap-3">
        {(['resnet50', 'mobilenetv2'] as ModelKey[]).map((m) => (
          <button
            key={m}
            onClick={() => { setModel(m); setResult(null); }}
            className={`flex-1 py-3 px-4 rounded-xl border text-sm font-semibold transition-all ${
              model === m
                ? 'bg-green-800 text-white border-green-800 shadow-sm'
                : 'bg-white text-slate-600 border-slate-200 hover:border-green-300'
            }`}
          >
            {MODEL_LABELS[m]}
            {m === 'resnet50' && <span className="ml-2 text-xs opacity-70">88.6% acc</span>}
            {m === 'mobilenetv2' && <span className="ml-2 text-xs opacity-70">81.6% acc</span>}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Upload panel */}
        <div className="space-y-4">
          {/* Drop zone */}
          {!preview ? (
            <div
              onDrop={onDrop}
              onDragOver={(e) => e.preventDefault()}
              onClick={() => inputRef.current?.click()}
              className="border-2 border-dashed border-slate-300 hover:border-green-400 rounded-2xl aspect-square flex flex-col items-center justify-center gap-4 cursor-pointer bg-slate-50 hover:bg-green-50 transition-all group"
            >
              <div className="w-14 h-14 rounded-full bg-white border border-slate-200 flex items-center justify-center group-hover:border-green-300 transition-colors shadow-sm">
                <Upload className="w-7 h-7 text-slate-400 group-hover:text-green-600" />
              </div>
              <div className="text-center">
                <p className="text-slate-600 font-medium text-sm">Drop an image here</p>
                <p className="text-slate-400 text-xs mt-1">or click to browse · JPEG / PNG · max 10 MB</p>
              </div>
              <input
                ref={inputRef}
                type="file"
                accept="image/jpeg,image/jpg,image/png"
                className="hidden"
                onChange={onInputChange}
              />
            </div>
          ) : (
            <div className="relative rounded-2xl overflow-hidden border border-slate-200 aspect-square bg-black">
              <img src={preview} alt="Preview" className="w-full h-full object-contain" />
              <button
                onClick={clearImage}
                className="absolute top-3 right-3 w-7 h-7 bg-white rounded-full shadow-md flex items-center justify-center hover:bg-red-50 transition-colors"
              >
                <X className="w-4 h-4 text-slate-600" />
              </button>
              {file && (
                <div className="absolute bottom-3 left-3 bg-black/60 backdrop-blur-sm text-white text-xs px-3 py-1.5 rounded-lg">
                  {file.name} · {(file.size / 1024).toFixed(0)} KB
                </div>
              )}
            </div>
          )}

          {/* Error */}
          {error && <ErrorBanner message={error} />}

          {/* Predict button */}
          <button
            onClick={handlePredict}
            disabled={!file || loading}
            className="w-full py-3 rounded-xl bg-green-800 text-white font-semibold text-sm hover:bg-green-900 disabled:opacity-40 disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-2"
          >
            {loading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" /> Running {MODEL_LABELS[model]}…
              </>
            ) : (
              <>
                <ImageIcon className="w-4 h-4" /> Classify with {MODEL_LABELS[model]}
              </>
            )}
          </button>

          <input
            ref={inputRef}
            type="file"
            accept="image/jpeg,image/jpg,image/png"
            className="hidden"
            onChange={onInputChange}
          />
        </div>

        {/* Results panel */}
        <div className="space-y-4">
          {!result && !loading && (
            <div className="h-full flex flex-col items-center justify-center gap-3 py-16 text-center">
              <div className="w-16 h-16 rounded-full bg-slate-100 flex items-center justify-center">
                <ImageIcon className="w-8 h-8 text-slate-300" />
              </div>
              <p className="text-slate-400 text-sm">Upload an image and click Classify to see results here.</p>
            </div>
          )}

          {loading && (
            <div className="h-full flex flex-col items-center justify-center gap-3 py-16">
              <Loader2 className="w-10 h-10 text-green-600 animate-spin" />
              <p className="text-slate-500 text-sm">Running inference on {MODEL_LABELS[model]}…</p>
              <p className="text-slate-400 text-xs">(First prediction may take a moment while the model loads)</p>
            </div>
          )}

          {result && (
            <div className="space-y-4">
              {/* Top prediction */}
              <div className="bg-white border border-green-200 rounded-2xl p-5">
                <div className="flex items-start gap-3">
                  <CheckCircle className="w-6 h-6 text-green-600 mt-0.5 shrink-0" />
                  <div className="flex-1 min-w-0">
                    <p className="text-xs text-slate-500 font-medium uppercase tracking-wide mb-1">Predicted Class</p>
                    <p className="text-xl font-bold text-green-800 leading-tight">{result.predicted_class}</p>
                    <div className="flex items-center gap-3 mt-2">
                      <span className="text-sm text-slate-600">
                        Confidence: <strong className="text-green-700">{(result.confidence * 100).toFixed(1)}%</strong>
                      </span>
                      <span className="text-sm text-slate-400">·</span>
                      <span className="text-sm text-slate-600">
                        Latency: <strong>{result.inference_ms} ms</strong>
                      </span>
                    </div>
                  </div>
                </div>

                {/* Confidence bar */}
                <div className="mt-4">
                  <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-green-600 rounded-full transition-all duration-500"
                      style={{ width: `${result.confidence * 100}%` }}
                    />
                  </div>
                </div>
              </div>

              {/* Score distribution */}
              <div className="bg-white border border-slate-200 rounded-2xl p-5">
                <h3 className="font-semibold text-slate-800 mb-4 text-sm">Score Distribution</h3>
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={chartData} layout="vertical" margin={{ left: 8, right: 16, top: 4, bottom: 4 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" horizontal={false} />
                    <XAxis type="number" domain={[0, 100]} tick={{ fontSize: 11, fill: '#94a3b8' }} tickFormatter={(v) => `${v}%`} />
                    <YAxis
                      type="category"
                      dataKey="class"
                      tick={{ fontSize: 10, fill: '#475569' }}
                      width={130}
                    />
                    <Tooltip formatter={(v) => `${Number(v).toFixed(1)}%`} />
                    <Bar dataKey="score" radius={[0, 3, 3, 0]}>
                      {chartData.map((entry) => (
                        <Cell
                          key={entry.class}
                          fill={entry.class === result.predicted_class ? CLASS_COLORS_MAP[entry.class] || '#16a34a' : '#cbd5e1'}
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* Disclaimer */}
              <div className="flex gap-2 rounded-xl bg-amber-50 border border-amber-200 px-4 py-3 text-xs text-amber-800">
                <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
                <p>{result.disclaimer}</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

