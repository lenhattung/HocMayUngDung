"""run_chapter.py -- Chạy lần lượt các ô lệnh của một chương và lưu kết quả để đưa vào sách.

Cách dùng (từ thư mục gốc của dự án):
    python code/run_chapter.py ch01

Việc script làm:
  1. Chạy các file code/<chương>/s*.py theo thứ tự tên, trong CÙNG một không gian biến
     (giống như chạy lần lượt các ô trong Jupyter Notebook).
  2. Ghi phần in ra màn hình của mỗi ô vào outputs/<chương>/<tên ô>.txt.
  3. Tạo notebook code/<chương>/<chương>_notebook.ipynb gồm đúng các ô lệnh đó.
"""
import contextlib
import io
import json
import os
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
chapter = sys.argv[1] if len(sys.argv) > 1 else "ch01"
src_dir = ROOT / "code" / chapter
out_dir = ROOT / "outputs" / chapter
out_dir.mkdir(parents=True, exist_ok=True)
(ROOT / "figures").mkdir(exist_ok=True)

os.chdir(ROOT)                                   # đường dẫn "data/...", "figures/..." tính từ gốc dự án
sys.path.insert(0, str(ROOT / "code"))
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import setup_style                   # chỉ để hình trong sách có cùng phong cách
setup_style()

ns = {"__name__": "__main__"}
cells = sorted(src_dir.glob("s*.py"))
for f in cells:
    code = f.read_text(encoding="utf-8")
    buf = io.StringIO()
    t0 = time.time()
    with contextlib.redirect_stdout(buf):
        exec(compile(code, str(f), "exec"), ns)
    plt.close("all")
    text = buf.getvalue().rstrip("\n")
    (out_dir / (f.stem + ".txt")).write_text(text if text else "(Ô lệnh này không in ra gì.)",
                                             encoding="utf-8")
    # Tiêu đề ô lệnh (dòng chú thích đầu tiên) -> file .tex để sách in ở đầu hộp mã nguồn
    first = code.splitlines()[0].lstrip("# ").strip()
    num, _, title = first.partition(" -- ")
    esc = lambda t: (t.replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("%", "\\%")
                     .replace("&", "\\&").replace("#", "\\#"))
    (out_dir / (f.stem + "_title.tex")).write_text(f"{esc(num)}\\ \\textperiodcentered\\ {esc(title)}%",
                                                   encoding="utf-8")
    width = max((len(l) for l in text.splitlines()), default=0)
    print(f"{f.name:28s} {time.time() - t0:6.1f}s  rộng nhất {width} ký tự", file=sys.__stdout__)

# ---- Xuất notebook để sinh viên chạy theo
nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                   "language_info": {"name": "python"}},
      "cells": [{"cell_type": "markdown", "metadata": {}, "id": "title",
                 "source": [f"# {chapter.upper()} -- Notebook thực hành\n",
                            "Chạy lần lượt từng ô từ trên xuống. Đặt notebook ở thư mục gốc dự án "
                            "(cùng cấp với thư mục `data/` và `figures/`)."]}]}
for i, f in enumerate(cells):
    lines = f.read_text(encoding="utf-8").splitlines(keepends=True)
    nb["cells"].append({"cell_type": "markdown", "metadata": {}, "id": f"h{i}",
                        "source": ["### " + lines[0].lstrip("# ").strip()]})
    nb["cells"].append({"cell_type": "code", "metadata": {}, "id": f"c{i}", "execution_count": None,
                        "outputs": [], "source": lines[1:]})
(src_dir / f"{chapter}_notebook.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1),
                                                    encoding="utf-8")
print("Đã tạo notebook:", src_dir / f"{chapter}_notebook.ipynb", file=sys.__stdout__)
