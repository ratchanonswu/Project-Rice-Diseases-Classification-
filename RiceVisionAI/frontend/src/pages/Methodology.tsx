import React, { type ElementType, type ReactNode } from 'react';
import { SectionHeader, Badge } from '../components/UI';
import { Database, Filter, Layers, FlaskConical, BarChart2, GitBranch, Shield, Cpu } from 'lucide-react';

interface Step {
  icon: ElementType;
  title: string;
  badge?: string;
  badgeColor?: 'green' | 'blue' | 'orange' | 'slate';
  content: ReactNode;
}

const steps: Step[] = [
  {
    icon: Database,
    title: 'Dataset Collection',
    badge: '3,829 images',
    badgeColor: 'slate',
    content: (
      <div className="space-y-2 text-sm text-slate-600 leading-relaxed">
        <p>
          The dataset comprises 3,829 JPEG images of rice leaves across six classes: Bacterial Leaf Blight, 
          Brown Spot, Healthy Rice Leaf, Leaf Blast, Leaf Scald, and Sheath Blight. Images were sourced 
          from the existing project dataset with class folders used as authoritative labels.
        </p>
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 mt-3">
          {[
            ['Bacterial Leaf Blight', '~630'],
            ['Brown Spot', '~570'],
            ['Healthy Rice Leaf', '~700'],
            ['Leaf Blast', '~630'],
            ['Leaf Scald', '~640'],
            ['Sheath Blight', '~659'],
          ].map(([cls, count]) => (
            <div key={cls} className="bg-slate-50 border border-slate-200 rounded-lg px-3 py-2">
              <p className="text-xs font-medium text-slate-700">{cls}</p>
              <p className="text-xs text-slate-400">{count} images</p>
            </div>
          ))}
        </div>
      </div>
    ),
  },
  {
    icon: Filter,
    title: 'Deduplication & Audit',
    badge: '31 groups removed',
    badgeColor: 'orange',
    content: (
      <div className="space-y-3 text-sm text-slate-600 leading-relaxed">
        <p>
          An automated audit pipeline detected exact duplicates using MD5 file hashing and near-duplicates 
          using perceptual hashing (pHash with Hamming distance ≤ 10). All findings were recorded in 
          structured manifests.
        </p>
        <ul className="space-y-1.5 list-disc list-inside text-slate-500">
          <li>31 exact-duplicate groups identified; one canonical copy kept per group</li>
          <li>125 confirmed near-duplicate pairs identified for manual review</li>
          <li>Contact sheets generated for visual verification of near-duplicate pairs</li>
          <li>Near-duplicates were reviewed but not excluded from the clean dataset</li>
          <li>All exclusions documented in <code className="text-xs bg-slate-100 px-1 rounded">excluded_images.csv</code></li>
        </ul>
        <p className="text-xs text-slate-400 italic">
          Clean dataset: 3,798 images after removing 31 exact-duplicate redundant copies.
        </p>
      </div>
    ),
  },
  {
    icon: GitBranch,
    title: 'Leakage-Safe Splitting',
    badge: '75 / 10 / 15 %',
    badgeColor: 'green',
    content: (
      <div className="space-y-3 text-sm text-slate-600 leading-relaxed">
        <p>
          Images were split into train, validation, and test sets using stratified random sampling 
          with a fixed random seed for reproducibility. Before splitting, duplicate group membership 
          was verified to ensure no image from the same group crosses split boundaries.
        </p>
        <div className="grid grid-cols-3 gap-3">
          {[
            { split: 'Train', count: 2847, pct: '74.96%' },
            { split: 'Validation', count: 381, pct: '10.03%' },
            { split: 'Test', count: 570, pct: '15.01%' },
          ].map(({ split, count, pct }) => (
            <div key={split} className="bg-green-50 border border-green-200 rounded-xl px-4 py-3 text-center">
              <p className="text-sm font-bold text-green-800">{count}</p>
              <p className="text-xs text-green-600 font-medium">{split}</p>
              <p className="text-xs text-green-500">{pct}</p>
            </div>
          ))}
        </div>
        <p className="text-xs text-slate-400">
          Split manifests: <code className="bg-slate-100 px-1 rounded">clean_train.csv</code>, 
          <code className="bg-slate-100 px-1 rounded ml-1">clean_validation.csv</code>, 
          <code className="bg-slate-100 px-1 rounded ml-1">clean_test.csv</code>
        </p>
      </div>
    ),
  },
  {
    icon: Layers,
    title: 'Model Architecture',
    badge: 'Transfer learning',
    badgeColor: 'blue',
    content: (
      <div className="space-y-4 text-sm text-slate-600 leading-relaxed">
        <p>
          Both models use ImageNet-pretrained backbones with frozen base weights and fine-tuned 
          classification heads, following a transfer learning protocol.
        </p>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div className="bg-slate-50 border border-slate-200 rounded-xl p-4">
            <h4 className="font-semibold text-slate-800 mb-2">ResNet50</h4>
            <ul className="space-y-1 text-xs text-slate-500">
              <li>• 24.1M total parameters</li>
              <li>• 526K trainable parameters</li>
              <li>• Input: (224, 224, 3)</li>
              <li>• Preprocessing: zero-centered BGR ImageNet means</li>
              <li>• Head: GlobalAveragePool → Dense → Softmax</li>
              <li>• Model size: 96.64 MB</li>
            </ul>
          </div>
          <div className="bg-slate-50 border border-slate-200 rounded-xl p-4">
            <h4 className="font-semibold text-slate-800 mb-2">MobileNetV2</h4>
            <ul className="space-y-1 text-xs text-slate-500">
              <li>• 2.6M total parameters</li>
              <li>• 329K trainable parameters</li>
              <li>• Input: (224, 224, 3)</li>
              <li>• Preprocessing: scaled to [−1, 1]</li>
              <li>• Head: GlobalAveragePool → Dense → Softmax</li>
              <li>• Model size: 12.96 MB</li>
            </ul>
          </div>
        </div>
      </div>
    ),
  },
  {
    icon: FlaskConical,
    title: 'Training Protocol',
    badge: 'Clean Split v1',
    badgeColor: 'green',
    content: (
      <div className="space-y-3 text-sm text-slate-600 leading-relaxed">
        <p>
          Both models were retrained from scratch on the clean dataset using the same train/validation 
          split. Training configurations were kept consistent where applicable for a fair comparison.
        </p>
        <div className="overflow-x-auto">
          <table className="w-full text-xs border border-slate-200 rounded-lg overflow-hidden">
            <thead className="bg-slate-50">
              <tr>
                <th className="text-left px-3 py-2 text-slate-600">Setting</th>
                <th className="text-right px-3 py-2 text-green-700">ResNet50</th>
                <th className="text-right px-3 py-2 text-blue-700">MobileNetV2</th>
              </tr>
            </thead>
            <tbody>
              {[
                ['Image size', '224×224', '224×224'],
                ['Batch size', '32', '32'],
                ['Optimizer', 'Adam', 'Adam'],
                ['Learning rate', '1e-4', '1e-4'],
                ['Early stopping', 'val_loss, patience=10', 'val_loss, patience=10'],
                ['Best epoch', '25', '23'],
                ['Training time', '33.69 min', '11.65 min'],
                ['Hardware', 'CPU (AMD64)', 'CPU (AMD64)'],
              ].map(([setting, rn, mv], i) => (
                <tr key={setting} className={i % 2 === 0 ? 'bg-white' : 'bg-slate-50'}>
                  <td className="px-3 py-2 text-slate-600">{setting}</td>
                  <td className="px-3 py-2 text-right text-green-700 font-medium">{rn}</td>
                  <td className="px-3 py-2 text-right text-blue-700 font-medium">{mv}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="text-xs text-slate-400">
          Checkpoint selection criterion: lowest validation loss. Test set evaluated only once after 
          training and checkpoint selection were finalized.
        </p>
      </div>
    ),
  },
  {
    icon: BarChart2,
    title: 'Evaluation',
    badge: '570 test images',
    badgeColor: 'slate',
    content: (
      <div className="space-y-2 text-sm text-slate-600 leading-relaxed">
        <p>
          Final evaluation was performed on the held-out test set (570 images). Metrics reported include 
          accuracy, macro-averaged precision/recall/F1, weighted averages, ROC-AUC, per-class F1-scores, 
          and confusion matrices.
        </p>
        <ul className="space-y-1.5 list-disc list-inside text-slate-500">
          <li>ResNet50: 88.60% accuracy, 88.39% macro F1, 98.93% ROC-AUC</li>
          <li>MobileNetV2: 81.58% accuracy, 81.32% macro F1, 97.09% ROC-AUC</li>
          <li>Both models struggle most with Leaf Blast (easily confused with other classes)</li>
          <li>Healthy Rice Leaf achieves highest F1 in both models (≥94%)</li>
        </ul>
      </div>
    ),
  },
  {
    icon: Shield,
    title: 'Integrity Measures',
    content: (
      <div className="space-y-2 text-sm text-slate-600 leading-relaxed">
        <ul className="space-y-2 list-disc list-inside text-slate-500">
          <li>Original dataset images were never deleted, renamed, or overwritten</li>
          <li>Historical baseline results (original split) preserved separately</li>
          <li>Test data used only for final evaluation — not for hyperparameter tuning</li>
          <li>All preprocessing exactly matches training code (verified via source inspection)</li>
          <li>No test leakage: all duplicate groups confined to a single split</li>
          <li>Duplicate group manifests and audit reports stored for traceability</li>
        </ul>
      </div>
    ),
  },
  {
    icon: Cpu,
    title: 'Inference & Deployment',
    content: (
      <div className="space-y-2 text-sm text-slate-600 leading-relaxed">
        <p>
          This web application loads the trained Keras models lazily at first request and caches them in 
          memory. Predictions use the exact same preprocessing pipeline as training.
        </p>
        <ul className="space-y-1.5 list-disc list-inside text-slate-500">
          <li>Backend: FastAPI + TensorFlow/Keras on CPU</li>
          <li>Frontend: React + Vite + TypeScript + Tailwind CSS + Recharts</li>
          <li>Images are processed in-memory; file paths are never exposed to the browser</li>
          <li>Representative class images are served as base64-encoded thumbnails</li>
          <li>ResNet50 avg inference: ~197 ms · MobileNetV2 avg inference: ~105 ms (CPU, profiled)</li>
        </ul>
        <p className="text-xs text-amber-700 bg-amber-50 border border-amber-200 px-3 py-2 rounded-lg mt-3">
          ⚠ This is an educational research prototype. Predictions are model outputs and may not generalize 
          to real-world conditions. Not intended for agricultural diagnostic use.
        </p>
      </div>
    ),
  },
];

export default function Methodology() {
  return (
    <div className="space-y-6">
      <SectionHeader
        title="Methodology"
        subtitle="End-to-end pipeline: data audit → deduplication → leakage-safe splits → training → evaluation"
      />

      {/* Pipeline overview */}
      <div className="bg-white border border-slate-200 rounded-2xl p-6">
        <h3 className="font-semibold text-slate-800 mb-4">Pipeline Overview</h3>
        <div className="flex flex-wrap gap-2 items-center text-xs text-slate-600">
          {['Raw Dataset', 'MD5 Hashing', 'pHash Dedup', 'Manual Review', 'Split Manifests', 'Train ResNet50', 'Train MobileNetV2', 'Evaluation', 'Dashboard'].map((step, i, arr) => (
            <React.Fragment key={step}>
              <span className="bg-slate-100 border border-slate-200 px-3 py-1.5 rounded-lg font-medium">{step}</span>
              {i < arr.length - 1 && <span className="text-slate-300">→</span>}
            </React.Fragment>
          ))}
        </div>
      </div>

      {/* Steps */}
      <div className="space-y-4">
        {steps.map(({ icon: Icon, title, badge, badgeColor, content }, i) => (
          <details key={title} className="group bg-white border border-slate-200 rounded-2xl overflow-hidden" open={i < 3}>
            <summary className="flex items-center gap-4 px-6 py-4 cursor-pointer hover:bg-slate-50 transition-colors list-none">
              <div className="w-9 h-9 rounded-xl bg-green-50 border border-green-200 flex items-center justify-center shrink-0">
                <Icon className="w-5 h-5 text-green-700" />
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-3 flex-wrap">
                  <span className="font-semibold text-slate-800">{title}</span>
                  {badge && <Badge color={badgeColor}>{badge}</Badge>}
                </div>
              </div>
              <span className="text-slate-400 group-open:rotate-180 transition-transform text-lg leading-none">›</span>
            </summary>
            <div className="px-6 pb-5 pt-1 border-t border-slate-100">
              {content}
            </div>
          </details>
        ))}
      </div>

      {/* File map */}
      <div className="bg-white border border-slate-200 rounded-2xl p-6">
        <h3 className="font-semibold text-slate-800 mb-4">Key Artifact Locations</h3>
        <div className="space-y-2">
          {[
            ['Clean train manifest', 'DatasetAudit/manifests/clean_train.csv'],
            ['Clean validation manifest', 'DatasetAudit/manifests/clean_validation.csv'],
            ['Clean test manifest', 'DatasetAudit/manifests/clean_test.csv'],
            ['Duplicate groups', 'DatasetAudit/manifests/duplicate_groups.csv'],
            ['Audit report', 'DatasetAudit/reports/audit_report.md'],
            ['ResNet50 checkpoint', 'ResNet50/models/clean_split_v1/resnet50.keras'],
            ['MobileNetV2 checkpoint', 'MobileNetV2/models/clean_split_v1/mobilenetv2.keras'],
            ['ResNet50 metrics', 'ResNet50/outputs/clean_split_v1/metrics.json'],
            ['MobileNetV2 metrics', 'MobileNetV2/outputs/clean_split_v1/metrics.json'],
          ].map(([label, path]) => (
            <div key={label} className="flex items-start gap-3 text-sm">
              <span className="text-slate-500 shrink-0 w-52">{label}</span>
              <code className="text-xs bg-slate-100 text-slate-700 px-2 py-0.5 rounded break-all">{path}</code>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

