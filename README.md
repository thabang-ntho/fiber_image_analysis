**FiberMicroscope-Toolkit**
*A Python-based toolkit for quantitative, geometrical analysis of fiber-cross-section images, featuring a modular core library and a responsive desktop GUI.*

---

## Coding-Agent Prompt (v4.0)

### 1. Objectives

* **Harden & Refactor** the existing `fiber_analysis_driver.py` into a rock-solid, importable analysis library.
* **Build** a modern, non-blocking desktop GUI (`image_gui_app.py`) that leverages the core library.
* **Automate** quality checks, packaging, and environment setup.
* **Deliver** comprehensive documentation and a battle-tested test suite.

---

### 2. Design Principles

1. **Modularity**

   * Keep core analysis classes and utilities in `fiber_analysis_driver.py` as a standalone Python package.
   * GUI code must import and use only public interfaces—no logic duplication.
2. **Separation of Concerns**

   * Core: image I/O, thresholding, contour extraction, metrics and plotting.
   * GUI: event handling, user interactions, threading, and result display.
3. **Extensibility**

   * New analyses register via `AnalysisFactory` (entry-point or decorator).
   * GUI automatically picks up new `AnalysisType` and `SpecialtyFiberType` values.
4. **Responsiveness**

   * All heavy tasks (image processing, file I/O) run in background threads (e.g. `ThreadPoolExecutor`).
   * GUI must remain interactive—use a progress bar and disable inputs during processing.
5. **Robustness**

   * Use `AnalysisError` for predictable failures and fall back to generic exception handlers.
   * Validate all user inputs with clear error messages.
6. **Reproducibility**

   * Pin dependencies in `pyproject.toml`.
   * Provide environment and build scripts—no manual steps.

---

### 3. Deliverables

#### 3.1 Project Setup & Quality Automation

* **`pyproject.toml`**

  * Declare project metadata, version, dependencies, and tool config sections for Black, Ruff, and Mypy.
* **Pre-commit Hooks** (`.pre-commit-config.yaml`)

  * Run **black**, **ruff**, and **mypy** on each commit.
* **Logging Configuration**

  * Add a top-level `logging.yaml` or configure programmatically in `main()` to write both to console and `analysis.log`.

#### 3.2 Core Library Enhancements (`fiber_analysis_driver.py`)

* **Bug Fixes & TODOs**

  * Correct factory registration typos.
  * Finish all analysis methods (`_analyze_outer`, `_analyze_core`, `_analyze_pm`, etc.).
* **CLI Enhancements**

  * Add flags:

    * `--test` to run `_run_tests()` and exit
    * `--version` (reads from `metadata.version`)
    * `--analysis`, `--fiber_type`, `--poly_tolerance`, `--min_area`, `--num_cores`, `--out`, `--verbose`
  * Validate argument combinations and surface clear usage text.
* **Logging & Error Handling**

  * Replace all prints (outside CLI) with `logger.debug/info/warning/error`.
  * Catch `AnalysisError` vs. generic exceptions in `main()`.
* **Persistence**

  * Ensure `save_results()` writes both JSON and CSV and handles file-I/O errors gracefully.

#### 3.3 GUI Implementation (`image_gui_app.py`)

* **Toolkit**: `tkinter` + `ttkbootstrap`
* **File Input**

  * Support drag-and-drop and a “File → Open” dialog (`.tif, .png, .jpg`).
* **Controls**

  * ComboBox for `AnalysisType`; dynamically show `SpecialtyFiberType` combo when needed.
  * Numeric entries/sliders for `poly_tolerance`, `min_area`, `num_cores`.
  * “Run Analysis” button, only enabled when all required inputs are valid.
* **Execution**

  * Run analysis in `ThreadPoolExecutor`; show an indeterminate or percent progress bar.
  * On exception, catch in worker thread and dispatch to `tkinter.messagebox.showerror`.
* **Results Display**

  * Use `ttk.Notebook` with two tabs:

    * **Metrics**: `ttk.Treeview` table of key/value pairs.
    * **Image**: Canvas or `Label` showing the annotated NumPy image.
* **Export**

  * Menu: “File → Save Metrics As…” (CSV/JSON), “File → Save Image As…” (PNG).
* **UX Enhancements**

  * Dark/light theme toggle in the menu.
  * Responsive resizing of the image panel.

#### 3.4 Packaging & Environment Scripts

* **`scripts/setup_env.sh`**

  * Use **uv** to create a virtual environment named `fiber_analysis`, activate it, then install the project in editable mode (`pip install -e .`) and install development dependencies.
* **`scripts/build_windows.sh`**

  * Use PyInstaller to generate a one-file Windows executable.
* **`scripts/build_linux.sh`**

  * Use Nuitka (or fallback to PyInstaller) for a standalone Linux AppImage or binary.
* **Documentation**

  * Include comments in build scripts explaining each step.

#### 3.5 Documentation & Testing Documentation & Testing

* **`README.md`**

  * Installation, CLI examples, screenshot of GUI, and Python API snippets.
* **API Docs**

  * Generate HTML docs from docstrings via `pdoc --html`.
* **Unit Tests** (`tests/`)

  * `test_driver.py`: sanity checks on CLI flags, factory, `_load_and_prepare_image`.
  * `test_edge_cases.py`: invalid files, blank images, mismatched analysis types.
  * Achieve ≥90% coverage and integrate with CI (e.g. GitHub Actions).

---

### 3.6 Sample Data

* **`sample_images/`**

  * Include at least one representative image per analysis type:

    * `sample_polygonal.png`
    * `sample_pm_fiber.tif`
    * (and so on for multi-core, hollow-core)
  * Use these images in automated tests and bundle them for new users to try immediately.

---

### 4. Project Layout

```
.
├── fiber_analysis_driver.py
├── image_gui_app.py
├── pyproject.toml
├── logging.yaml
├── .pre-commit-config.yaml
├── sample_images/           # Representative example inputs for each analysis type
│   ├── sample_polygonal.png
│   ├── sample_pm_fiber.tif
│   ├── sample_multi_core.jpg
│   └── sample_hollow_core.png
├── scripts/
│   ├── setup_env.sh
│   ├── build_windows.sh
│   └── build_linux.sh
├── tests/
│   ├── test_driver.py
│   └── test_edge_cases.py
└── README.md
```
