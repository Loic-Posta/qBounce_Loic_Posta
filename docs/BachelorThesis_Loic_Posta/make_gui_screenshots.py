#!/usr/bin/env python3
"""Screenshot every tab of PipelineController's GUI, for the user guide.

Builds the real application (so what is captured is what the operator sees,
not a mock-up), selects each notebook tab in turn, and grabs the window.

macOS: screencapture -l <windowid> would need the window id, which Tk does not
expose portably, so we use the window's on-screen geometry instead and crop.
Run it on a machine with a display; there is no headless path -- a screenshot
of a window that was never drawn would be misleading.
"""
import subprocess
import sys
import time
import tkinter as tk
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures_gui"
OUT.mkdir(exist_ok=True)

sys.path.insert(0, str(HERE.parent.parent))
import importlib.util
spec = importlib.util.spec_from_file_location(
    "pc", HERE.parent.parent / "PipelineController.py")
pc = importlib.util.module_from_spec(spec)
sys.modules["pc"] = pc
spec.loader.exec_module(pc)


def grab(root: tk.Misc, path: Path) -> bool:
    """Capture the region of the screen the window occupies."""
    root.update_idletasks()
    root.update()
    time.sleep(0.6)
    x, y = root.winfo_rootx(), root.winfo_rooty()
    w, h = root.winfo_width(), root.winfo_height()
    r = subprocess.run(["screencapture", "-x", "-R", f"{x},{y},{w},{h}", str(path)],
                       capture_output=True)
    return r.returncode == 0 and path.is_file()


def main() -> int:
    root = tk.Tk()
    root.geometry("1400x950+40+40")
    app = pc.PipelineControllerApp(root)
    root.update()
    time.sleep(1.2)

    nb = app.nb
    tabs = [(i, nb.tab(i, "text")) for i in nb.tabs()]
    print(f"{len(tabs)} tabs: {[t for _, t in tabs]}")

    for idx, name in tabs:
        nb.select(idx)
        slug = "".join(c.lower() if c.isalnum() else "_" for c in name).strip("_")
        dest = OUT / f"gui_{slug}.png"
        ok = grab(root, dest)
        print(f"  {'OK ' if ok else 'FAIL'} {name:<24} -> {dest.name}")

    root.destroy()
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
