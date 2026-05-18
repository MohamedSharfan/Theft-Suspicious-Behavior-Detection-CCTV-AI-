## Python AI pipeline

```bash
pip install -r requirements.txt
```

## CCTV AI dashboard

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000` to view the CCTV AI dashboard.

For local backend testing:

```bash
uvicorn src.api.server:app --host 0.0.0.0 --port 8000
```

For Render:

```bash
pip install -r requirements.txt
uvicorn src.api.server:app --host 0.0.0.0 --port 10000
```

Set `CLOUD=true` on Render so the backend does not try to open `cv2.VideoCapture`.
Set `NEXT_PUBLIC_API_BASE` in the frontend to your backend URL, for example `https://your-render-url.onrender.com`.
