# Analysis step

## 1) Create a TMDb API key and export it for this shell:
**macOS/Linux:**  
```bash
export TMDB_API_KEY=YOUR_KEY
```

**Windows PowerShell:**  
```powershell
setx TMDB_API_KEY YOUR_KEY
```
Then reopen terminal.

## 2) Install dependencies
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 3) Run the script
```bash
python color_analysis.py --csv input/movies.csv --rows 5 --cols 10 --k 5 --out ../frontend/public/posters.json
```
