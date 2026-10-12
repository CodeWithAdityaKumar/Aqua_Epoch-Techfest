Haan, next phase baad mein karenge. Abhi project ko local machine par use karne ke steps:

## 1. Project folder open karo

Terminal mein project root par jao:

```bash
cd Aqua_Epoch
```

Check karo ki ye files/folders available hain:

```text
artifacts/phase3/
backend/
frontend/
ml/
scripts/
requirements.txt
```

## 2. Python virtual environment banao

Python **3.12 recommended** hai, kyunki promoted models isi runtime par verify hue hain.

Linux/macOS:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Agar `python3.12` command available nahi hai:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

## 3. Dependencies install karo

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Promoted models ke exact runtime versions install karo:

```bash
python -m pip install -r artifacts/phase3/requirements-serving.txt
```

Verify:

```bash
python -m pip check
```

## 4. Models aur frontend check karo

Project root se run karo:

```bash
python -m backend.demo --check-only
```

Expected output:

```text
Demo preflight passed: all four models and public frontend files are ready.
```

Agar runtime mismatch error aaye, exact model runtime install karo:

```bash
python -m pip install -r artifacts/phase3/requirements-serving.txt
```

## 5. Local application start karo

```bash
python -m backend.demo
```

Browser mein open karo:

```text
http://127.0.0.1:8000/
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

Models/readiness:

```text
http://127.0.0.1:8000/health
http://127.0.0.1:8000/models
```

Server stop karne ke liye:

```text
Ctrl + C
```

## Alternative launcher

Agar shell use karna ho:

```bash
PYTHON_BIN="$PWD/.venv/bin/python" bash scripts/serve_demo.sh
```

Check-only mode:

```bash
PYTHON_BIN="$PWD/.venv/bin/python" \
bash scripts/serve_demo.sh --check-only
```

## Demo kaise use karna hai

1. Page open karo.
2. `Refresh metadata` press karo agar models loading mein stuck hon.
3. Pehle **Quick screening** try kar sakte ho:
   - Turbidity
   - Hardness
   - TDS
4. Better demonstration ke liye **Full profile** use karo.
5. `Use synthetic training-median example` button sirf workflow demonstration ke liye hai.
6. Result mein:
   - `assessment`
   - `probability_safe`
   - decision threshold
   - warnings
   - pH reliability
   read karo.
7. Quick mode ko reliable safety verdict ke roop mein present mat karna.
8. Local history save karni ho to checkbox enable karo. Default mein history save nahi hoti.

History default location:

```text
.local/predictions.sqlite3
```

Ye local demo database hai; isko public network par expose mat karna.

## Quick API test

Server start hone ke baad:

```bash
curl -X POST http://127.0.0.1:8000/classify/quick \
  -H "Content-Type: application/json" \
  -d '{"Turbidity": 3, "Hardness": 150, "TDS": 1200}'
```

Combined quick prediction:

```bash
curl -X POST http://127.0.0.1:8000/predict/quick \
  -H "Content-Type: application/json" \
  -d '{"Turbidity": 3, "Hardness": 150, "TDS": 1200}'
```

## Important

- Runtime local-only hai: `127.0.0.1`
- Public hosting abhi configured nahi hai
- `0.0.0.0` par bind karke expose mat karo
- Ye competition demo hai, water-safety certification nahi
- Frontend run karne ke liye Node server ki zarurat nahi hai; FastAPI frontend serve karta hai
- `npm install`/`npm test` sirf frontend tests ke liye hai, normal app run ke liye nahi

Normal daily command sequence:

```bash
cd Aqua_Epoch
source .venv/bin/activate
python -m backend.demo --check-only
python -m backend.demo
```

Phir browser mein:

```text
http://127.0.0.1:8000/
```