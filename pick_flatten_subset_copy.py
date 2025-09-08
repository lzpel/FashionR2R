#!/usr/bin/env python3
import argparse, os, random, shutil, sys
from pathlib import Path

DEF_EXTS = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]

def is_image_file(name: str, exts: set) -> bool:
    name = name.lower()
    return any(name.endswith(ext) for ext in exts)

def count_images_in_dir(d: Path, exts: set) -> int:
    cnt = 0
    with os.scandir(d) as it:
        for e in it:
            if e.is_file() and is_image_file(e.name, exts):
                cnt += 1
    return cnt

def reservoir_sample_dir(d: Path, k: int, exts: set) -> list[Path]:
    if k <= 0:
        return []
    sample: list[Path] = []
    seen = 0
    with os.scandir(d) as it:
        for e in it:
            if not e.is_file() or not is_image_file(e.name, exts):
                continue
            seen += 1
            p = Path(e.path)
            if len(sample) < k:
                sample.append(p)
            else:
                j = random.randrange(seen)  # [0, seen-1]
                if j < k:
                    sample[j] = p
    return sample

def safe_dest_path(dst_dir: Path, category: str, src_name: str) -> Path:
    # 保存名: <カテゴリ>__<元ファイル名> （衝突時は連番）
    base = f"{category}__{src_name}"
    out = dst_dir / base
    if not out.exists():
        return out
    # 連番付与
    stem, dot, ext = base.rpartition('.')
    if dot == "":
        stem, ext = base, ""
    i = 1
    while True:
        cand = dst_dir / (f"{stem}_{i}{('.' + ext) if ext else ''}")
        if not cand.exists():
            return cand
        i += 1

def compute_quota_per_dir(counts: dict[str, int], total: int) -> dict[str, int]:
    cats = list(counts.keys())
    D = len(cats)
    if D == 0:
        return {}
    base = max(total // D, 0)
    q = {c: min(base, counts[c]) for c in cats}
    allocated = sum(q.values())
    rem = total - allocated
    if rem <= 0:
        return q
    target = base + 1
    while rem > 0:
        progressed = False
        for c in cats:
            if q[c] < target and q[c] < counts[c]:
                q[c] += 1
                rem -= 1
                progressed = True
                if rem == 0:
                    break
        if not progressed:
            target += 1
            if all(q[c] >= counts[c] for c in cats):
                break
    return q

def main():
    ap = argparse.ArgumentParser(
        description="Two-level tree (cat/image) からカテゴリ均等に近いランダム抽出を行い、平坦フォルダにコピーします。"
    )
    ap.add_argument("--src", required=True, help="ソース 2階層ルート（例: synfashion_release）")
    ap.add_argument("--dst", required=True, help="出力先 平坦ディレクトリ（例: flatten_subset）")
    ap.add_argument("--total", type=int, default=1000, help="抽出する合計枚数（デフォルト 1000）")
    ap.add_argument("--seed", type=int, default=None, help="乱数シード（任意）")
    ap.add_argument("--ext", nargs="*", default=DEF_EXTS,
                    help=f"対象拡張子（デフォルト: {', '.join(DEF_EXTS)})")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    exts = set(e.lower() if e.startswith(".") else "." + e.lower() for e in args.ext)

    if args.seed is not None:
        random.seed(args.seed)

    if not src.is_dir():
        print(f"[ERR] src not found or not a dir: {src}", file=sys.stderr); sys.exit(1)
    dst.mkdir(parents=True, exist_ok=True)

    # 直下のカテゴリディレクトリ収集
    cats_dirs: list[Path] = []
    with os.scandir(src) as it:
        for e in it:
            if e.is_dir():
                cats_dirs.append(Path(e.path))
    if not cats_dirs:
        print(f"[ERR] no category subdirectories under {src}", file=sys.stderr); sys.exit(1)

    # 各カテゴリの画像枚数を数える
    counts: dict[str, int] = {}
    total_available = 0
    for d in cats_dirs:
        n = count_images_in_dir(d, exts)
        if n > 0:
            counts[d.name] = n
            total_available += n
    if not counts:
        print(f"[ERR] no images found with extensions {sorted(exts)}", file=sys.stderr); sys.exit(1)

    target_total = min(args.total, total_available)
    if target_total < args.total:
        print(f"[WARN] requested {args.total} but only {total_available} available; sampling {target_total}")

    # 均等割当を計算
    quotas = compute_quota_per_dir(counts, target_total)

    # カテゴリごとにリザーバサンプリング→コピー
    picked = 0
    for d in cats_dirs:
        cat = d.name
        k = quotas.get(cat, 0)
        if k <= 0:
            continue
        sample = reservoir_sample_dir(d, k, exts)
        if len(sample) < k:
            print(f"[WARN] {cat}: expected {k}, got {len(sample)}", file=sys.stderr)
        for src_path in sample:
            dst_path = safe_dest_path(dst, cat, src_path.name)
            shutil.copy2(src_path, dst_path)
        picked += len(sample)

    print(f"[OK] sampled {picked} images into: {dst}")

if __name__ == "__main__":
    main()
