"""
data_selector.py  --  interactive selection of acquisition data from the SMB share
=================================================================================
Acquisition and organisation of the frames a run will be analysed from.

What it does:
  1. Checks that the SMB volume is mounted under /Volumes/
  2. Walks the remote tree and lists its sub-folders
  3. Lets you select
       - one or more folders of DARK frames (background, no beam)
       - one or more folders of SIGNAL frames (beam on, with tracks)
  4. Copies the frames locally into Code/Data/ in a fixed layout
  5. Writes a JSON manifest recording exactly what was copied

Usage:
    python data_selector.py
    python data_selector.py --smb-root /Volumes/<volume>/path/to/data
    python data_selector.py --data-dir /path/to/Code/Data --max-per-folder 100
    python data_selector.py --dry-run   (lists what would be copied, copies nothing)

Dependencies: the standard library only; nothing to pip install.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

# On Windows, sys.stdout defaults to the console's legacy code page (cp1252 on a
# European install), and every print() below that carries an arrow, a degree
# sign or an accent then dies with UnicodeEncodeError. That is not cosmetic: the
# CSV is written before the plots, so the script exits 1 having produced the data
# and none of the figures. Force UTF-8 on the two streams, and fall back to
# replacing the offending character rather than crashing.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# --- Defaults ----------------------------------------------------------------

# The share as macOS Finder mounts it: smb://<server>/qbounce/... appears under
# /Volumes/qbounce/... . Override with --smb-root on any other platform.
DEFAULT_SMB_ROOT = Path(
    "/Volumes/qbounce/Experimentierzeiten/2026/Vienna-ATI-2026"
    "/2026-ATI-CMOS/IDScmos/SN-4108881777"
)

# Local cache directory, resolved relative to this file, so Code/Data/.
_SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = _SCRIPT_DIR.parent / "Data"

IMAGE_EXTENSIONS = {".png", ".bmp", ".tiff", ".tif"}

# Largest batch held in memory before flushing, to keep the copy off the heap.
COPY_CHUNK_BYTES = 4 * 1024 * 1024   # 4 MB


# ─── Couleurs ANSI (terminal macOS) ──────────────────────────────────────────

class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    CYAN   = "\033[36m"
    GREEN  = "\033[32m"
    YELLOW = "\033[33m"
    RED    = "\033[31m"
    BLUE   = "\033[34m"
    MAGENTA= "\033[35m"

def bold(s: str)    -> str: return f"{C.BOLD}{s}{C.RESET}"
def cyan(s: str)    -> str: return f"{C.CYAN}{s}{C.RESET}"
def green(s: str)   -> str: return f"{C.GREEN}{s}{C.RESET}"
def yellow(s: str)  -> str: return f"{C.YELLOW}{s}{C.RESET}"
def red(s: str)     -> str: return f"{C.RED}{s}{C.RESET}"
def dim(s: str)     -> str: return f"{C.DIM}{s}{C.RESET}"
def magenta(s: str) -> str: return f"{C.MAGENTA}{s}{C.RESET}"


# ─── Utilitaires ─────────────────────────────────────────────────────────────

def hr(char: str = "─", width: int = 70) -> str:
    return dim(char * width)


def format_size(n_bytes: int) -> str:
    """Format a byte count for display."""
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"


def count_images(folder: Path) -> tuple[int, int]:
    """Return (image count, total bytes) for one folder, without recursing."""
    total_size = 0
    count = 0
    try:
        for p in folder.iterdir():
            if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file():
                count += 1
                total_size += p.stat().st_size
    except PermissionError:
        pass
    return count, total_size


def list_subdirs(folder: Path) -> list[Path]:
    """List the immediate sub-folders, in natural order."""
    try:
        subs = sorted(
            [p for p in folder.iterdir() if p.is_dir()],
            key=lambda p: p.name.lower()
        )
    except (PermissionError, OSError) as e:
        print(red(f"  Impossible de lire {folder}: {e}"))
        subs = []
    return subs


def print_banner() -> None:
    print()
    print(bold(cyan("╔══════════════════════════════════════════════════════════╗")))
    print(bold(cyan("║        Data selector  --  CMOS particle acquisition        ║")))
    print(bold(cyan("╚══════════════════════════════════════════════════════════╝")))
    print()


# ─── Vérification du montage SMB ─────────────────────────────────────────────

def check_smb_mount(smb_root: Path) -> bool:
    """
    Check that the share is reachable.
    On macOS the Finder mounts smb://HOST/SHARE under /Volumes/SHARE/.
    """
    print(hr())
    print(bold("  Étape 1 / 3 — Vérification du volume SMB"))
    print(hr())
    print()
    print(f"  Chemin attendu : {cyan(str(smb_root))}")
    print()

    if not smb_root.exists():
        print(red("  ✗  Volume introuvable !"))
        print()
        print("  Check that the volume is mounted:")
        print(dim("  Finder → Aller → Se connecter au serveur…"))
        print(dim(f"  smb://128.131.63.218/qbounce"))
        print()
        # List what is mounted under /Volumes/, to help the user find it.
        volumes = Path("/Volumes")
        if volumes.exists():
            available = [p.name for p in volumes.iterdir() if p.is_dir()]
            if available:
                print(f"  Volumes actuellement montés : {', '.join(available)}")
        return False

    print(green("  ✓  Volume accessible"))
    print()
    return True


# --- Browsing and folder selection -------------------------------------------

def browse_and_select(
    root: Path,
    role: str,          # "NOIRES (dark)" ou "SIGNAL (expériences)"
    role_short: str,    # "dark" ou "signal"
    already_selected: list[Path] | None = None,
) -> list[Path]:
    """
    Interactive walk through the mounted share.
    Returns the folders selected for this role.
    """
    already_selected = already_selected or []
    selected: list[Path] = []
    current = root

    print(hr())
    print(bold(f"  Étape 2{'' if role_short == 'dark' else 'b'} / 3 — "
               f"Selecting {magenta(role)} folders"))
    print(hr())
    print()
    print(dim("  Commands: number(s) to select  |  'e' to enter a folder"))
    print(dim("            'r' to go back  |  'done' to finish  |  'ls' to refresh"))
    print()

    while True:
        # Show the current folder
        print(bold(f"  📂  {cyan(str(current))}"))
        subdirs = list_subdirs(current)

        if not subdirs:
            print(dim("     (no sub-folders)"))
        else:
            print()
            for i, d in enumerate(subdirs):
                n_img, sz = count_images(d)
                marker = green("  ✓") if d in selected or d in already_selected else "   "
                img_info = (
                    f"{dim(f'  [{n_img} imgs, {format_size(sz)}]')}"
                    if n_img > 0 else dim("  [vide]")
                )
                print(f"  {marker} {bold(str(i+1)):>4}.  {d.name}{img_info}")

        # What has been selected so far
        if selected:
            print()
            print(f"  {green('Selected for this role:')} "
                  + ", ".join(green(d.name) for d in selected))

        print()
        raw = input(f"  {bold('>')} ").strip().lower()
        print()

        if raw in ("done", "d", "ok", ""):
            if not selected:
                confirm = input(
                    yellow("  No folder selected. Continue anyway? [y/N] ")
                ).strip().lower()
                if confirm not in ("o", "oui", "y", "yes"):
                    continue
            break

        if raw == "ls":
            continue

        if raw == "r":
            if current != root:
                current = current.parent
            else:
                print(dim("  (already at the root)"))
            continue

        # Enter a sub-folder: 'e3' or 'e 3'
        if raw.startswith("e"):
            idx_str = raw[1:].strip()
            try:
                idx = int(idx_str) - 1
                if 0 <= idx < len(subdirs):
                    current = subdirs[idx]
                else:
                    print(red(f"  Numéro invalide (1–{len(subdirs)})"))
            except ValueError:
                print(red("  Format: 'e3' to enter folder 3"))
            continue

        # Selection by number, e.g. "1 3 5" or "2"
        parts = raw.replace(",", " ").split()
        valid = True
        to_toggle: list[Path] = []
        for part in parts:
            try:
                idx = int(part) - 1
                if 0 <= idx < len(subdirs):
                    to_toggle.append(subdirs[idx])
                else:
                    print(red(f"  Numéro invalide : {part} (1–{len(subdirs)})"))
                    valid = False
                    break
            except ValueError:
                print(red(f"  Commande non reconnue : '{part}'"))
                valid = False
                break

        if valid:
            for d in to_toggle:
                # Check the folder holds images directly
                n_img, _ = count_images(d)
                if n_img == 0:
                    sub_subs = list_subdirs(d)
                    if sub_subs:
                        print(yellow(
                            f"  !  '{d.name}' holds no images directly "
                            f"({len(sub_subs)} sub-folders). "
                            f"Enter it with 'e{parts[0]}'?"
                        ))
                        continue
                    else:
                        print(yellow(f"  !  '{d.name}' looks empty; selected anyway."))

                if d in selected:
                    selected.remove(d)
                    print(dim(f"  — Désélectionné : {d.name}"))
                else:
                    selected.append(d)
                    print(green(f"  + Sélectionné : {d.name}"))

    return selected


# --- Summary before copying ---------------------------------------------------

def print_summary(
    dark_folders: list[Path],
    signal_folders: list[Path],
    max_per_folder: int | None,
) -> tuple[int, int]:
    """Print the summary and return (total images, total bytes)."""
    print(hr())
    print(bold("  Session summary"))
    print(hr())
    print()

    total_images = 0
    total_bytes = 0

    def _print_group(folders: list[Path], label: str, color_fn) -> None:
        nonlocal total_images, total_bytes
        print(bold(f"  {label}"))
        if not folders:
            print(dim("    (no folder selected)"))
            return
        for f in folders:
            n, sz = count_images(f)
            effective_n = min(n, max_per_folder) if max_per_folder else n
            effective_sz = int(sz * effective_n / max(n, 1))
            total_images += effective_n
            total_bytes  += effective_sz
            limit_note = f" → {max_per_folder} max" if max_per_folder and n > max_per_folder else ""
            print(f"    {color_fn('●')}  {f.name}  "
                  f"{dim(f'{n} imgs, {format_size(sz)}{limit_note}')}")
        print()

    _print_group(dark_folders,   "DARK frames (no beam)", yellow)
    _print_group(signal_folders, "SIGNAL frames (beam on)", green)

    print(f"  {bold('Total à copier :')} "
          f"{bold(str(total_images))} images  "
          f"({bold(format_size(total_bytes))})")
    print()

    return total_images, total_bytes


# --- Copying, with a progress bar ---------------------------------------------

def copy_with_progress(
    src: Path,
    dst: Path,
    current: int,
    total: int,
    dry_run: bool = False,
) -> int:
    """
    Copy src to dst, reporting progress.
    Returns the number of bytes copied, or 0 under --dry-run.
    """
    if dry_run:
        return 0

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        # Already cached: skip when the size matches
        if dst.stat().st_size == src.stat().st_size:
            return 0

    shutil.copy2(src, dst)
    return src.stat().st_size


def progress_bar(current: int, total: int, width: int = 40, suffix: str = "") -> str:
    pct = current / max(total, 1)
    filled = int(pct * width)
    bar = green("█" * filled) + dim("░" * (width - filled))
    return f"  [{bar}] {bold(f'{current}/{total}')}  {dim(suffix)}"


def copy_folders(
    folders: list[Path],
    role: str,          # "dark" ou "signal"
    dest_root: Path,
    max_per_folder: int | None,
    dry_run: bool,
    session_id: str,
) -> list[dict]:
    """
    Copy every image of the listed folders into dest_root/role/<folder name>/.
    Returns the manifest entries.
    """
    manifest_entries: list[dict] = []

    for folder in folders:
        images = sorted(
            [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS],
            key=lambda p: p.name,
        )
        if max_per_folder:
            images = images[:max_per_folder]

        dest_folder = dest_root / role / folder.name
        if not dry_run:
            dest_folder.mkdir(parents=True, exist_ok=True)

        print()
        print(f"  {bold(folder.name)}  →  {dim(str(dest_folder))}")

        copied_bytes = 0
        copied_count = 0
        skipped_count = 0
        errors: list[str] = []

        t0 = time.time()
        for i, img in enumerate(images):
            dst = dest_folder / img.name
            try:
                b = copy_with_progress(img, dst, i + 1, len(images), dry_run)
                if b == 0 and not dry_run:
                    skipped_count += 1
                else:
                    copied_bytes += b
                    copied_count += 1
            except Exception as e:
                errors.append(f"{img.name}: {e}")

            # Refresh the progress bar every fifth image, and at the end
            if (i + 1) % 5 == 0 or (i + 1) == len(images):
                elapsed = time.time() - t0
                speed = copied_bytes / max(elapsed, 0.01)
                suffix = f"{format_size(speed)}/s"
                print(
                    f"\r{progress_bar(i + 1, len(images), suffix=suffix)}",
                    end="",
                    flush=True,
                )

        elapsed_total = time.time() - t0
        print()  # newline after the progress bar
        status_parts = []
        if copied_count:
            status_parts.append(green(f"{copied_count} copiées ({format_size(copied_bytes)})"))
        if skipped_count:
            status_parts.append(dim(f"{skipped_count} already cached"))
        if errors:
            status_parts.append(red(f"{len(errors)} erreurs"))
        print(f"  → {' | '.join(status_parts)}  "
              f"{dim(f'en {elapsed_total:.1f}s')}")

        if errors:
            for err in errors[:5]:
                print(red(f"    ✗ {err}"))
            if len(errors) > 5:
                print(red(f"    … et {len(errors)-5} autres erreurs"))

        manifest_entries.append({
            "role": role,
            "source_folder": str(folder),
            "dest_folder": str(dest_folder),
            "images_copied": copied_count,
            "images_skipped": skipped_count,
            "bytes_copied": copied_bytes,
            "errors": errors,
            "elapsed_s": round(elapsed_total, 2),
        })

    return manifest_entries


# ─── Manifest JSON ────────────────────────────────────────────────────────────

def write_manifest(
    session_id: str,
    dark_entries: list[dict],
    signal_entries: list[dict],
    dest_root: Path,
    dry_run: bool,
) -> Path:
    """Write the session manifest as JSON into dest_root/manifests/."""
    manifest = {
        "session_id": session_id,
        "created_at": datetime.now().isoformat(),
        "dry_run": dry_run,
        "dest_root": str(dest_root),
        "smb_root": str(DEFAULT_SMB_ROOT),
        "dark_folders": dark_entries,
        "signal_folders": signal_entries,
        "summary": {
            "total_dark_images": sum(e["images_copied"] for e in dark_entries),
            "total_signal_images": sum(e["images_copied"] for e in signal_entries),
            "total_bytes": sum(
                e["bytes_copied"] for e in dark_entries + signal_entries
            ),
        },
    }

    manifest_dir = dest_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{session_id}.json"

    if not dry_run:
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

    return manifest_path


# ─── Point d'entrée ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive selection of acquisition data from the SMB share",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--smb-root", type=Path, default=DEFAULT_SMB_ROOT,
        help="Root of the mounted share (a /Volumes/... path)",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=DEFAULT_DATA_DIR,
        help="Local cache directory (Code/Data/)",
    )
    parser.add_argument(
        "--max-per-folder", type=int, default=None,
        metavar="N",
        help="Cap the number of images copied per folder (e.g. 100)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report what would be copied, and write nothing",
    )
    parser.add_argument(
        "--skip-dark", action="store_true",
        help="Skip selecting dark frames (already done)",
    )
    parser.add_argument(
        "--skip-signal", action="store_true",
        help="Skip selecting signal frames (already done)",
    )
    args = parser.parse_args()

    print_banner()

    if args.dry_run:
        print(yellow("  !  DRY RUN: nothing will be copied\n"))

    # ── 1. Vérification du montage ────────────────────────────────────────
    if not check_smb_mount(args.smb_root):
        sys.exit(1)

    # -- 2. Folder selection ------------------------------------------------
    dark_folders: list[Path] = []
    signal_folders: list[Path] = []

    if not args.skip_dark:
        dark_folders = browse_and_select(
            args.smb_root,
            role="NOIRES (dark frames — bruit thermique)",
            role_short="dark",
        )
    else:
        print(dim("  (dark selection skipped)\n"))

    if not args.skip_signal:
        signal_folders = browse_and_select(
            args.smb_root,
            role="SIGNAL (beam on, with tracks)",
            role_short="signal",
            already_selected=dark_folders,
        )
    else:
        print(dim("  (signal selection skipped)\n"))

    # ── 3. Résumé et confirmation ─────────────────────────────────────────
    total_imgs, total_bytes = print_summary(
        dark_folders, signal_folders, args.max_per_folder
    )

    if total_imgs == 0:
        print(yellow("  Aucune image à copier. Abandon."))
        sys.exit(0)

    # Vérification espace disque local
    local_free = shutil.disk_usage(args.data_dir.parent
                                    if not args.data_dir.exists()
                                    else args.data_dir).free
    if total_bytes > local_free * 0.9:
        print(red(
            f"  ⚠  Espace disque insuffisant !\n"
            f"     Nécessaire : {format_size(total_bytes)}\n"
            f"     Disponible : {format_size(local_free)}"
        ))
        confirm = input(yellow("  Continue anyway? [y/N] ")).strip().lower()
        if confirm not in ("o", "oui", "y", "yes"):
            sys.exit(1)
    else:
        print(f"  Espace libre local : {green(format_size(local_free))}")
        print()

    if not args.dry_run:
        confirm = input(
            bold("  Start copying? [Y/n] ")
        ).strip().lower()
        if confirm in ("n", "non", "no"):
            print(dim("  Abandon."))
            sys.exit(0)

    # -- 4. Copy ------------------------------------------------------------
    print()
    print(hr())
    print(bold("  Step 3 of 3 -- copying the frames"))
    print(hr())

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.data_dir.mkdir(parents=True, exist_ok=True)

    dark_entries: list[dict] = []
    signal_entries: list[dict] = []

    if dark_folders:
        print()
        print(bold(yellow("  > DARK frames")))
        dark_entries = copy_folders(
            dark_folders, "dark", args.data_dir,
            args.max_per_folder, args.dry_run, session_id,
        )

    if signal_folders:
        print()
        print(bold(green("  > SIGNAL frames")))
        signal_entries = copy_folders(
            signal_folders, "signal", args.data_dir,
            args.max_per_folder, args.dry_run, session_id,
        )

    # ── 5. Manifest ───────────────────────────────────────────────────────
    manifest_path = write_manifest(
        session_id, dark_entries, signal_entries,
        args.data_dir, args.dry_run,
    )

    # ── 6. Résumé final ───────────────────────────────────────────────────
    print()
    print(hr("═"))
    print()
    total_copied = sum(e["images_copied"] for e in dark_entries + signal_entries)
    total_skipped = sum(e["images_skipped"] for e in dark_entries + signal_entries)
    total_errors = sum(len(e["errors"]) for e in dark_entries + signal_entries)
    total_size = sum(e["bytes_copied"] for e in dark_entries + signal_entries)

    print(f"  {bold('Session :')} {cyan(session_id)}")
    print(f"  {bold('Images copied:')} {green(str(total_copied))}")
    if total_skipped:
        print(f"  {bold('Already cached:')} {dim(str(total_skipped))}")
    if total_errors:
        print(f"  {bold('Erreurs :')} {red(str(total_errors))}")
    print(f"  {bold('Data transferred:')} {format_size(total_size)}")
    if not args.dry_run:
        print(f"  {bold('Manifest :')} {cyan(str(manifest_path))}")
    print()

    # Show the layout that was created
    print(bold("  Structure locale créée :"))
    print(f"  {dim(str(args.data_dir) + '/')}")
    for role, folders in [("dark", dark_folders), ("signal", signal_folders)]:
        if folders:
            print(f"  ├── {bold(role)}/")
            for i, f in enumerate(folders):
                tree_char = "└──" if i == len(folders) - 1 else "├──"
                n_img, _ = count_images(args.data_dir / role / f.name)
                print(f"  │   {tree_char} {f.name}/  "
                      f"{dim(f'({n_img} imgs)')}")
    print(f"  └── manifests/")
    print(f"      └── {session_id}.json")
    print()

    print(green(bold("  ✓  Session terminée.")))
    print()
    print(dim("  Next step: run the C++ detector over these frames"))
    print(dim(f"  ./build/detector --folder {args.data_dir}/signal/<nom_dossier>"))
    print()


if __name__ == "__main__":
    main()