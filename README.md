# LayoutParser PDF Layout Processor

This project provides a script to sample random pages from PDF files, detect layout zones using LayoutParser (with an optional Detectron2 backend on CPU via WSL), visualize the results, and save annotated images.

---

## Prerequisites

* **Windows 10/11** with **WSL 2** (Ubuntu 20.04+)
* **Python 3** (≥ 3.8)
* **pip**, **virtualenv**, and system packages (poppler-utils, build-essential, cmake, etc.)

---

## Setup

### 1. Clone this repository

```bash
cd /mnt/c/Users/nikdu/my_python_projects
# Replace <repo-url> with your repo or local path
git clone <repo-url> layout_parser_linux
cd layout_parser_linux
```

### 2. Create & activate a virtual environment

```bash
# In your Ubuntu shell (WSL)
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install system dependencies

```bash
sudo apt update
sudo apt install -y poppler-utils build-essential cmake pkg-config libjpeg-dev libpng-dev
```

### 4. Install Python dependencies

```bash
pip install --upgrade pip setuptools wheel cython
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install 'git+https://github.com/facebookresearch/detectron2.git@v0.6'
pip install layoutparser pdf2image PyPDF2 matplotlib
```

### 5. Verify installation (optional)

```bash
python - <<'EOF'
import torch, detectron2, layoutparser as lp
print("torch:", torch.__version__)
print("detectron2:", detectron2.__version__)
print("layoutparser:", lp.__version__)
EOF
```

---

## Usage

1. **Prepare your PDFs**: place PDF files in a folder, e.g.:
   `/mnt/c/Users/nikdu/my_python_projects/layout_parser_linux/pdfs`

2. **Configure paths** in the script or notebook cells:

   * `folder_path`: input PDF directory
   * `output_folder`: directory to save annotated images

3. **Run the processor**:

```bash
python layout_test.py
```

By default, the script samples up to 10 random pages per PDF, detects layout zones, displays them inline (if enabled), and saves PNGs to your `output` folder.

---

## Project Structure

```
layout_parser_linux/
├── .venv/                 # Python virtual environment
├── layout_test.py         # Main processing script
├── layoutprocessor.ipynb  # (Optional) Jupyter notebook version
├── pdfs/                  # Place your PDF files here
├── output/                # Annotated images will be saved here
└── README.md              # This file
```

---

## Notes

* **CPU-only**: This setup uses a CPU‑only Detectron2 build. For GPU support, use a CUDA‑enabled environment or Docker as previously described.
* **Parameters**: Adjust `sample_size`, `dpi`, and `display_inline` in the script to suit your needs.
* **Reproducibility**: To share your exact environment, run:

  ```bash
  pip freeze > requirements.txt
  ```

Happy layout extracting 🚀
