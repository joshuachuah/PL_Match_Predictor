# 🚀 FPL Predictor Deployment Guide

## 📋 Pre-Deployment Checklist

### 1. Test Locally First
```bash
# Install dependencies
pip install -r requirements.txt

# Test the app
python run.py

# Verify cache creation
ls cache/  # Should see model files after first run
```

### 2. Verify App Structure
```
your-app/
├── app.py                 # Main Flask app
├── requirements.txt       # Dependencies
├── Procfile              # Process configuration
├── runtime.txt           # Python version
├── railway.toml          # Railway configuration
├── src/                  # Source code
├── templates/            # HTML templates
├── static/              # CSS/JS files
└── cache/               # Created automatically
```

## 🌐 Hosting Options

### Option A: Railway (Recommended)

**Why Railway?**
- ✅ Free tier with 500 hours/month
- ✅ Persistent storage for your cache
- ✅ Auto-deploy from GitHub
- ✅ Simple setup

**Steps:**
1. Push your code to GitHub
2. Go to [Railway](https://railway.app)
3. Sign in with GitHub
4. Click "New Project" → "Deploy from GitHub repo"
5. Select your repository
6. Railway auto-detects and deploys!

**Environment Variables** (Set in Railway dashboard):
```
ENABLE_SCHEDULER=true
CACHE_DIRECTORY=cache
LOG_LEVEL=INFO
```

### Option B: Render

**Steps:**
1. Push code to GitHub
2. Go to [Render](https://render.com)
3. Create new "Web Service"
4. Connect your GitHub repo
5. Use these settings:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app --bind 0.0.0.0:$PORT`

### Option C: Heroku

**Steps:**
1. Install [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli)
2. Create Heroku app:
```bash
heroku create your-fpl-predictor
heroku config:set ENABLE_SCHEDULER=true
git push heroku main
```

**Note:** Heroku's free tier doesn't have persistent storage. Your cache will reset on each restart.

### Option D: PythonAnywhere

**Steps:**
1. Upload your files to PythonAnywhere
2. Create a web app with Flask
3. Set the source directory and static files
4. Configure WSGI file to point to your app

## 🔧 Production Configuration

### Environment Variables
Set these in your hosting platform:

| Variable | Value | Description |
|----------|--------|-------------|
| `PORT` | (auto-set) | Server port |
| `FLASK_ENV` | `production` | Flask environment |
| `ENABLE_SCHEDULER` | `true` | Enable auto-retraining |
| `CACHE_DIRECTORY` | `cache` | Cache folder |
| `LOG_LEVEL` | `INFO` | Logging level |

### Performance Optimization
Your app is already optimized for production:

- ✅ **Model Caching** - No retraining on restart
- ✅ **Data Caching** - Reduced API calls
- ✅ **Automatic Scheduling** - Weekly updates
- ✅ **Gunicorn** - Production WSGI server
- ✅ **Logging** - Comprehensive monitoring

## 📊 Monitoring Your Deployed App

### Health Check Endpoints
```bash
# Check app status
curl https://your-app.railway.app/api/status

# View cache information
curl https://your-app.railway.app/api/cache/info

# Trigger manual retraining
curl -X POST https://your-app.railway.app/api/retrain
```

### Log Monitoring
Check your hosting platform's logs for:
- ✅ `Model loaded from cache` (fast startup)
- ✅ `Automatic retraining scheduler started`
- ✅ `Training scheduler started`

## 🛠️ Troubleshooting

### Common Issues

**1. Build Fails**
- Check `requirements.txt` has all dependencies
- Verify Python version in `runtime.txt`

**2. Slow First Load**
- Normal! First request trains the model (~30-60s)
- Subsequent requests are fast

**3. Memory Issues**
- Use fewer workers in `Procfile`: `--workers 1`
- Optimize in hosting platform settings

**4. Scheduler Not Working**
- Check `ENABLE_SCHEDULER=true` is set
- Verify logs show "scheduler started"

### Performance Tips
- **First deployment:** ~60 seconds to train model
- **Subsequent restarts:** ~5 seconds (uses cache)
- **Weekly updates:** Automatic, no downtime
- **Memory usage:** ~200-500MB typical

## 🎯 Next Steps

After deployment:
1. **Test predictions** - Visit `/api/predict`
2. **Monitor logs** - Check for errors
3. **Set up monitoring** - Use your platform's tools
4. **Configure alerts** - For failed retraining
5. **Scale if needed** - Add more workers

## 🔗 Useful Links

- [Railway Docs](https://docs.railway.app/)
- [Render Docs](https://render.com/docs)
- [Heroku Python](https://devcenter.heroku.com/articles/getting-started-with-python)
- [Flask Production](https://flask.palletsprojects.com/en/2.3.x/deploying/)

Your app is now production-ready with caching, scheduling, and automatic updates! 🚀
