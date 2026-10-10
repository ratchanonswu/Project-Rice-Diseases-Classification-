# RiceVision AI — Static Research Showcase

This is the **zero-server, static replica** of the RiceVision AI research dashboard.

It bundles all the benchmark metrics, confusion matrices, class references, and evaluation results directly into static assets. **No Python backend or TensorFlow server is required.**

---

## 🌟 What Works in this Static Showcase

- **Overview Dashboard**: Full dataset statistics and model performance summary.
- **Model Comparison**: Interactive radar charts, per-class F1 comparisons, resource efficiency table, and both 6×6 confusion matrices.
- **Disease Classes**: High-resolution representative images and symptom descriptions for all 6 rice leaf disease categories.
- **Methodology**: Detailed audit, deduplication, and training protocol documentation.
- **Interactive Prediction Inspector**: Visitors can select sample benchmark test leaves (Healthy, Brown Spot, Leaf Blast) and inspect how both ResNet50 and MobileNetV2 predicted the probabilities and confidence scores.

---

## 🚀 How to Deploy in 2 Minutes (100% Free)

### Option 1: Deploy to Vercel (Recommended)

1. Push your repository to **GitHub**.
2. Go to **[Vercel.com](https://vercel.com/)** and log in.
3. Click **Add New > Project** and import your GitHub repository.
4. Set the **Root Directory** to:
   ```
   RiceVisionAI-Static
   ```
5. Leave the default build settings:
   - Framework: `Vite`
   - Build Command: `npm run build`
   - Output Directory: `dist`
6. Click **Deploy**.

Your website will be live at `https://your-project.vercel.app` with zero hosting cost and instantaneous load times.

---

### Option 2: Deploy to Netlify

1. Go to **[Netlify.com](https://www.netlify.com/)**.
2. Click **Add new site > Import an existing project** (or drag and drop the `dist/` folder).
3. Set **Base directory**: `RiceVisionAI-Static`
4. Set **Build command**: `npm run build`
5. Set **Publish directory**: `RiceVisionAI-Static/dist`
6. Click **Deploy Site**.

---

## 💻 Local Testing

To test this static build locally before deploying:

```powershell
cd "RiceVisionAI-Static"
npm install
npm run build
npm run preview
```

Or simply double-click `start_static.bat` to preview at `http://localhost:4173`.

