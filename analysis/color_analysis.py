import os, io, json, math, argparse, time
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

import requests
import numpy as np
from PIL import Image
import cv2
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import pandas as pd

TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMG = "https://image.tmdb.org/t/p/"  # e.g. w500

@dataclass
class PosterRecord:
    title: str
    year: Optional[int]
    tmdb_id: Optional[int]
    poster_url: Optional[str]
    avg_rgb: List[int]
    lab_avg: List[float]
    palette: List[Dict[str, Any]]
    hue_histogram: List[float]
    signature: List[float]
    pca2d: List[float]
    grid_pos: Dict[str, int]


def fetch_tmdb_movie(title: str, year: Optional[int]) -> Optional[Dict[str, Any]]:
    if not TMDB_API_KEY:
        raise RuntimeError("TMDB_API_KEY not set")
    params = {"api_key": TMDB_API_KEY, "query": title}
    if year:
        params["year"] = int(year)
    r = requests.get(f"{TMDB_BASE}/search/movie", params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    return (data.get("results") or [None])[0]


def fetch_tmdb_movie_by_id(tmdb_id: int) -> Optional[Dict[str, Any]]:
    if not TMDB_API_KEY:
        raise RuntimeError("TMDB_API_KEY not set")
    r = requests.get(f"{TMDB_BASE}/movie/{tmdb_id}", params={"api_key": TMDB_API_KEY}, timeout=20)
    if r.status_code != 200:
        return None
    return r.json()


def tmdb_poster_url(poster_path: Optional[str], size: str = "w500") -> Optional[str]:
    if not poster_path:
        return None
    return f"{TMDB_IMG}{size}{poster_path}"


def download_image(url: str) -> Image.Image:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    img = Image.open(io.BytesIO(r.content)).convert("RGB")
    return img


def pil_to_cv(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def analyze_image(img_rgb: np.ndarray, k: int = 5) -> Tuple[List[int], List[float], List[Dict[str, Any]], List[float]]:
    # Resize for speed/consistency
    h, w = img_rgb.shape[:2]
    max_dim = max(h, w)
    scale = 512 / max_dim if max_dim > 512 else 1.0
    if scale != 1.0:
        img_rgb = cv2.resize(img_rgb, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

    # Average RGB
    avg_rgb = img_rgb.reshape(-1, 3).mean(axis=0)
    avg_rgb = np.clip(avg_rgb, 0, 255).astype(np.uint8).tolist()

    # Hue histogram (12 bins), weighted by saturation*value
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(np.float32)  # 0..179 (OpenCV)
    sat = hsv[:, :, 1].astype(np.float32) / 255.0
    val = hsv[:, :, 2].astype(np.float32) / 255.0
    weights = sat * val
    bins = 12
    h_deg = hue * 2.0  # to 0..360
    idx = np.floor(h_deg / (360.0 / bins)).astype(np.int32)
    idx = np.clip(idx, 0, bins - 1)
    hist = np.bincount(idx.flatten(), weights=weights.flatten(), minlength=bins)
    hist = (hist / (hist.sum() + 1e-8)).tolist()

    # Average LAB for similarity
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2LAB).reshape(-1, 3)
    lab_avg = lab.mean(axis=0).astype(float).tolist()

    # Palette via KMeans on LAB (perceptual)
    km = KMeans(n_clusters=k, n_init=4, random_state=42)
    labels = km.fit_predict(lab)
    centers_lab = km.cluster_centers_
    counts = np.bincount(labels, minlength=k).astype(float)
    weights_palette = counts / counts.sum()

    # Convert palette centers to RGB for display
    centers_lab_img = centers_lab.reshape(-1, 1, 3).astype(np.float32)
    centers_rgb = cv2.cvtColor(centers_lab_img, cv2.COLOR_LAB2BGR).reshape(-1, 3)
    centers_rgb = np.clip(centers_rgb, 0, 255).astype(np.uint8)

    order = np.argsort(-weights_palette)  # largest first
    palette = []
    for i in order:
        rgb = centers_rgb[i].tolist()[::-1][::-1]  # keep BGR→BGR (we display via CSS rgb())
        # Convert BGR to RGB for clarity
        b, g, r = centers_rgb[i].tolist()
        palette.append({
            "rgb": [int(r), int(g), int(b)],
            "weight": float(weights_palette[i])
        })

    return avg_rgb, lab_avg, palette, hist


def load_movies_csv(path: str) -> List[Dict[str, Any]]:
    df = pd.read_csv(path)
    out = []
    for _, row in df.iterrows():
        tmdb_id = int(row["TMDB_ID"]) if not pd.isna(row.get("TMDB_ID")) else None
        title = None if pd.isna(row.get("Title")) else str(row.get("Title")).strip()
        year = None
        if not pd.isna(row.get("Year")):
            try:
                year = int(row.get("Year"))
            except Exception:
                year = None
        out.append({"tmdb_id": tmdb_id, "title": title, "year": year})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--rows", type=int, default=5)
    ap.add_argument("--cols", type=int, default=10)
    ap.add_argument("--k", type=int, default=5, help="palette size per poster")
    ap.add_argument("--size", default="w500", help="TMDb poster size, e.g. w342, w500, original")
    ap.add_argument("--out", default="../frontend/public/posters.json")
    args = ap.parse_args()

    movies = load_movies_csv(args.csv)
    records: List[PosterRecord] = []

    # 1) Resolve posters
    posters_imgs = []  # (index, cv2 BGR image)
    meta_rows = []     # [(title, year, tmdb_id, poster_url)]

    for m in movies:
        tmdb_id = m.get("tmdb_id")
        title = m.get("title")
        year = m.get("year")
        info = None
        if tmdb_id:
            info = fetch_tmdb_movie_by_id(tmdb_id)
        else:
            if not title:
                print("Skipping row with neither TMDB_ID nor Title")
                continue
            info = fetch_tmdb_movie(title, year)
        if not info:
            print(f"Could not resolve: {title} ({year}) / {tmdb_id}")
            continue
        resolved_title = info.get("title") or info.get("name") or title
        release = (info.get("release_date") or "")[:4]
        try:
            resolved_year = int(release) if release else year
        except Exception:
            resolved_year = year
        poster_path = info.get("poster_path")
        url = tmdb_poster_url(poster_path, size=args.size)
        if not url:
            print(f"No poster for: {resolved_title}")
            continue
        try:
            pil = download_image(url)
            bgr = pil_to_cv(pil)
        except Exception as e:
            print(f"Download failed for {resolved_title}: {e}")
            continue
        posters_imgs.append(bgr)
        meta_rows.append((resolved_title, resolved_year, info.get("id"), url))

    if not posters_imgs:
        raise RuntimeError("No posters resolved. Check your CSV and TMDB_API_KEY.")

    # 2) Per-image analysis
    all_signatures = []
    tmp_records = []
    for img_bgr, meta in zip(posters_imgs, meta_rows):
        title, year, tmdb_id, url = meta
        avg_rgb, lab_avg, palette, hist = analyze_image(img_bgr, k=args.k)
        signature = hist  # 12D hue signature (you can concatenate more features later)
        all_signatures.append(signature)
        tmp_records.append({
            "title": title,
            "year": year,
            "tmdb_id": tmdb_id,
            "poster_url": url,
            "avg_rgb": avg_rgb,
            "lab_avg": lab_avg,
            "palette": palette,
            "hue_histogram": hist,
            "signature": signature
        })

    # 3) Global 2D embedding (PCA on signatures)
    sig = np.array(all_signatures)
    if sig.shape[0] >= 2:
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(sig)
    else:
        coords = np.zeros((sig.shape[0], 2))
    # normalize to [0,1]
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    span = (maxs - mins)
    span[span == 0] = 1.0
    coords01 = (coords - mins) / span

    # 4) Assign to an R×C grid by minimizing total distance
    R, C = int(args.rows), int(args.cols)
    N = R * C
    if len(tmp_records) != N:
        print(f"Warning: you have {len(tmp_records)} posters but grid is {R}x{C}={N}.")
        if len(tmp_records) < N:
            # pad coords with copies of nearest to keep assignment stable
            pad = N - len(tmp_records)
            coords01 = np.vstack([coords01, coords01[:pad]])
            tmp_records += tmp_records[:pad]
        else:
            # truncate extras
            coords01 = coords01[:N]
            tmp_records = tmp_records[:N]

    grid_points = np.array([(r / (R - 1 if R > 1 else 1), c / (C - 1 if C > 1 else 1)) for r in range(R) for c in range(C)])
    cost = cdist(coords01, grid_points, metric="sqeuclidean")
    row_ind, col_ind = linear_sum_assignment(cost)

    # Build final records with grid positions
    records: List[PosterRecord] = []
    for i, rec in enumerate(tmp_records):
        gi = int(col_ind[i])
        gr, gc = divmod(gi, C)
        p = PosterRecord(
            title=rec["title"],
            year=rec["year"],
            tmdb_id=int(rec["tmdb_id"]) if rec["tmdb_id"] is not None else None,
            poster_url=rec["poster_url"],
            avg_rgb=[int(x) for x in rec["avg_rgb"]],
            lab_avg=[float(x) for x in rec["lab_avg"]],
            palette=[{"rgb": [int(c) for c in p["rgb"]], "weight": float(p["weight"]) } for p in rec["palette"]],
            hue_histogram=[float(x) for x in rec["hue_histogram"]],
            signature=[float(x) for x in rec["signature"]],
            pca2d=[float(coords01[i,0]), float(coords01[i,1])],
            grid_pos={"row": int(gr), "col": int(gc)}
        )
        records.append(p)

    # 5) Output JSON
    out = {
        "generated_at": int(time.time()),
        "grid": {"rows": R, "cols": C},
        "items": [asdict(r) for r in records]
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Wrote {args.out} with {len(records)} items.")

if __name__ == "__main__":
    main()
