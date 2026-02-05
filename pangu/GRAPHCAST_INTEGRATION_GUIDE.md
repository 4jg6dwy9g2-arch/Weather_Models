# GraphCast Integration Guide

## What's New

Your Flask app now supports **two AI weather models**:

1. **PanguWeather** (Huawei) - Already working
2. **GraphCast** (Google DeepMind) - New! Requires additional setup

## What Was Added

### 1. Model Selection in UI
- **Run Forecast page** now has a dropdown to choose between PanguWeather and GraphCast
- Model information updates dynamically when you switch models
- Both models use the same ECMWF Open Data source

### 2. Backend Support
- `run_forecast()` function now accepts a `model` parameter
- API endpoint `/api/forecast/run` accepts model selection
- Run database stores which model was used for each forecast

### 3. Dashboard Updates
- Each forecast run shows a colored badge indicating which model was used
- PanguWeather: Blue badge
- GraphCast: Green badge
- Latest forecast chart title shows the model name

## GraphCast Setup Required

⚠️ **GraphCast needs additional dependencies that aren't installed yet.**

### What GraphCast Needs

GraphCast is built on Google's JAX framework (similar to PyTorch/TensorFlow) and requires:

1. **JAX** - Machine learning framework
2. **GraphCast model files** - The actual neural network weights
3. **Haiku** - Neural network library for JAX
4. **Additional scientific libraries**

### Installation Options

#### Option 1: CPU-Only (Slower, but easier)

```bash
cd ~/Documents/ML_Weather_Models

# Install JAX for CPU
pip install jax jaxlib

# Install additional GraphCast dependencies
pip install dm-haiku chex jraph

# Download GraphCast model assets
ai-models --download-assets graphcast
```

**Note**: CPU-based GraphCast will be **much slower** than PanguWeather. Expect 10-30 minutes per forecast vs. 5-10 minutes for PanguWeather.

#### Option 2: GPU (Faster, but complex)

If you have an NVIDIA GPU with CUDA:

```bash
# Install JAX with CUDA support
pip install "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install GraphCast dependencies
pip install dm-haiku chex jraph

# Download model assets
ai-models --download-assets graphcast
```

This requires:
- NVIDIA GPU with CUDA 11.x
- CUDA Toolkit installed
- cuDNN libraries
- GPU drivers

#### Option 3: Skip GraphCast for Now

You can keep using PanguWeather! The app works perfectly with just PanguWeather:
- Simply select "PanguWeather" when running forecasts
- All existing features continue to work
- You can add GraphCast later if desired

## Testing GraphCast

Once installed, test it with:

```bash
# Quick test (6-hour forecast)
ai-models graphcast \
  --input ecmwf-open-data \
  --date 20260127 \
  --time 12 \
  --lead-time 6 \
  --path test_graphcast.grib
```

If this works, GraphCast is ready to use in the Flask app!

## Using the Integrated App

1. Start the Flask app:
   ```bash
   cd ~/Documents/ML_Weather_Models
   python pangu_app.py
   ```

2. Go to **Run Forecast** page

3. Select your model:
   - **PanguWeather**: Fast, proven, already working
   - **GraphCast**: Cutting-edge, may be more accurate, needs setup

4. Configure your forecast and run

5. View results on Dashboard (with model badge)

## Model Comparison

| Feature | PanguWeather | GraphCast |
|---------|-------------|-----------|
| Developer | Huawei | Google DeepMind |
| Setup | ✅ Ready | ⚠️ Needs JAX |
| Speed (CPU) | ~5-10 min | ~10-30 min |
| Speed (GPU) | N/A | ~2-5 min |
| Resolution | 0.25° | 0.25° |
| Max Lead Time | 7 days | 10+ days |
| Accuracy | Excellent | Excellent+ |
| Dependencies | Minimal | Complex |

## Troubleshooting

### "No module named 'jax'"
You need to install JAX first (see Installation Options above).

### "You need to install Graphcast from git"
This means additional dependencies are missing. Install dm-haiku, chex, and jraph.

### GraphCast is very slow
You're running on CPU. Consider:
- Using PanguWeather instead
- Setting up GPU acceleration
- Running shorter lead times (6-24 hours)

### Model assets not found
Run: `ai-models --download-assets graphcast`

## References

- [ECMWF ai-models-graphcast GitHub](https://github.com/ecmwf-lab/ai-models-graphcast)
- [JAX Installation Guide](https://docs.jax.dev/en/latest/installation.html)
- [GraphCast Paper](https://arxiv.org/abs/2212.12794)

## Next Steps

**Recommended approach:**
1. Keep using PanguWeather for now (it's working great!)
2. Try installing GraphCast with CPU-only setup when you have time
3. Compare the two models on the same initial conditions
4. Decide if GraphCast's complexity is worth it for your use case

The Flask app is ready for both models - you just need to complete the GraphCast installation when you're ready!
