# Shared snippet — paste at top of Kaggle cells OR import after cloning repo

REPO_URL = "https://github.com/mersh01/hybrid-lightweight-xray-detection.git"
REPO_DIR = Path("/kaggle/working/hybrid-lightweight-xray-detection")


def ensure_repo():
    import subprocess
    if not (REPO_DIR / "models" / "hybrid_rare_ft_epoch20.pt").exists():
        if REPO_DIR.exists():
            import shutil
            shutil.rmtree(REPO_DIR, ignore_errors=True)
        print("Cloning GitHub repo for weights + modules...")
        subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)],
            check=True,
        )
    return REPO_DIR


def search_roots(*extra):
    roots = [Path("/kaggle/input")]
    roots.extend(extra)
    repo = ensure_repo()
    roots.append(repo)
    return roots


def list_pts(roots, min_mb=5):
    out = []
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*.pt"):
            if p.is_file() and p.stat().st_size >= min_mb * 1_000_000:
                out.append(p)
    return sorted(set(out), key=lambda p: -p.stat().st_size)


def find_long_and_ft(roots):
    pts = list_pts(roots)
    ft = None
    long = None

    for p in pts:
        n = p.name.lower()
        if n in ("hybrid_rare_ft_epoch20.pt", "epoch20.pt") or "epoch20" in n:
            ft = p
            break
    if ft is None:
        big = [p for p in pts if p.stat().st_size > 20_000_000]
        if big:
            ft = big[0]

    for p in pts:
        if p == ft:
            continue
        n = p.name.lower()
        if n == "hybrid_fdd_long_best.pt":
            long = p
            break
    if long is None:
        for p in pts:
            if p == ft:
                continue
            if "epoch20" in p.name.lower() or "rare" in str(p).lower():
                continue
            if 8_000_000 < p.stat().st_size < 20_000_000:
                long = p
                break
    if long is None:
        for p in pts:
            if p != ft and "best" in p.name.lower():
                long = p
                break

    if long is None or ft is None:
        lines = "\n".join(f"  {p} ({p.stat().st_size/1e6:.1f} MB)" for p in pts[:25])
        raise AssertionError(
            "Missing checkpoint(s).\n"
            "Option A — add Kaggle Input: dataset from GitHub repo hybrid-lightweight-xray-detection\n"
            "Option B — upload models/hybrid_fdd_long_best.pt (~13 MB) + hybrid_rare_ft_epoch20.pt (~26 MB)\n"
            "Option C — re-run this cell (it auto-clones the repo into /kaggle/working)\n"
            f".pt files found:\n{lines or '  (none — clone may have failed)'}"
        )
    return long, ft
