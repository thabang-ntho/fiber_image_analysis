#!/usr/bin/env python3
"""
Fiber Analysis Driver Core Library

Provides a unified, object-oriented framework for performing geometric analysis
on microscope cross-sections of optical fibers. Supports both command-line
and GUI-driven workflows.

Components:
- AbstractAnalysis: Defines the core interface for analyses.
- PolygonalFiberAnalysis: "Type 1" analysis for regular polygonal fibers.
- SpecialtyFiberAnalysis: "Type 2" analysis for specialty fibers.
- AnalysisFactory: Registry-based factory for analysis classes.
- main(): CLI driver with configurable logging, verbose mode, test runner, and version.
"""
from __future__ import annotations

import abc
import argparse
import enum
import json
import logging
import sys
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from skimage import color, filters, io, measure, morphology

# -----------------------------------------------------------------------------
# Custom Exceptions
# -----------------------------------------------------------------------------

class AnalysisError(Exception):
    """Base exception for analysis-related errors."""
    pass

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Abstract Base Class and Utilities
# -----------------------------------------------------------------------------

class AbstractAnalysis(abc.ABC):
    """
    Base interface for all fiber analysis implementations.
    """

    def __init__(self, image_path: Path, output_path: Optional[Path] = None) -> None:
        if not image_path.is_file():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        self.image_path = image_path
        self.output_path = output_path
        self.results: Dict[str, Any] = {}
        self.annotated_image: Optional[np.ndarray] = None

    @abc.abstractmethod
    def execute(self) -> None:
        """
        Execute the analysis pipeline. Populate self.results and self.annotated_image.
        """
        raise NotImplementedError

    def _load_and_prepare_image(self) -> Tuple[np.ndarray, np.ndarray]:
        logger.debug(f"Loading image: {self.image_path}")
        image = io.imread(self.image_path)
        if image.ndim == 2:
            rgb_image = color.gray2rgb(image)
            gray_image = image
        else:
            rgb_image = image[..., :3]
            gray_image = color.rgb2gray(rgb_image)
        thresh = filters.threshold_otsu(gray_image)
        binary = gray_image > thresh
        if np.sum(binary) > binary.size * 0.6:
            logger.debug("Inverting binary image for bright background.")
            binary = ~binary
        return rgb_image, binary.astype(np.uint8)

    def _figure_to_array(self, fig: plt.Figure) -> np.ndarray:
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        plt.close(fig)
        return buf.reshape((h, w, 3))

    def save_results(
        self,
        output_prefix: str,
        save_json: bool = True,
        save_csv: bool = True,
    ) -> None:
        if not self.results:
            logger.warning("No results to save; run execute() first.")
            return
        if save_json:
            path = Path(f"{output_prefix}_metrics.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.results, f, indent=4)
            logger.info(f"Saved JSON metrics: {path}")
        if save_csv:
            path = Path(f"{output_prefix}_metrics.csv")
            pd.DataFrame.from_dict(
                self.results, orient="index", columns=["Value"]
            ).to_csv(path)
            logger.info(f"Saved CSV metrics: {path}")
        if self.annotated_image is not None and self.output_path:
            io.imsave(self.output_path, self.annotated_image)
            logger.info(f"Saved annotated image: {self.output_path}")

# -----------------------------------------------------------------------------
# Factory and Enums
# -----------------------------------------------------------------------------

class AnalysisType(str, enum.Enum):
    POLYGONAL = "polygonal"
    SPECIALTY = "specialty"

class SpecialtyFiberType(str, enum.Enum):
    PM_FIBER = "pm_fiber"
    MULTI_CORE = "multi_core"
    HOLLOW_CORE = "hollow_core"

class AnalysisFactory:
    """Registry-based factory for analysis implementations."""

    _registry: Dict[AnalysisType, Type[AbstractAnalysis]] = {}

    @classmethod
    def register(
        cls, key: AnalysisType
    ) -> callable[[Type[AbstractAnalysis]], Type[AbstractAnalysis]]:
        def decorator(subclass: Type[AbstractAnalysis]) -> Type[AbstractAnalysis]:
            cls._registry[key] = subclass
            return subclass
        return decorator

    @classmethod
    def create(
        cls, key: AnalysisType, **kwargs: Any
    ) -> AbstractAnalysis:
        impl = cls._registry.get(key)
        if not impl:
            raise AnalysisError(f"Unknown analysis key: {key}")
        logger.info(f"Instantiating analysis: {key.value}")
        return impl(**kwargs)

# -----------------------------------------------------------------------------
# Polygonal Fiber Analysis
# -----------------------------------------------------------------------------

@AnalysisFactory.register(AnalysisType.POLYGONAL)
class PolygonalFiberAnalysis(AbstractAnalysis):
    """
    "Type 1" analysis for fibers with regular polygonal cladding.
    """
    def __init__(
        self,
        image_path: Path,
        output_path: Optional[Path] = None,
        poly_tolerance: float = 5.0,
        min_area: int = 100,
    ) -> None:
        super().__init__(image_path, output_path)
        self.poly_tolerance = poly_tolerance
        self.min_area = min_area

    def execute(self) -> None:
        rgb, binary = self._load_and_prepare_image()
        contours = self._find_contours(binary)
        if len(contours) < 2:
            raise AnalysisError("Need ≥2 contours for polygonal analysis.")
        outer, core = contours[:2]
        self._analyze_outer(outer, binary)
        self._analyze_core(core)
        self._calc_concentricity()
        self.annotated_image = self._visualize(rgb, outer, core)
        logger.info("Polygonal analysis complete.")

    def _find_contours(self, binary: np.ndarray) -> List[np.ndarray]:
        raw = measure.find_contours(binary, level=0.8)
        filtered = [c for c in raw if cv2.contourArea(c.astype(int)) > self.min_area]
        return sorted(
            filtered,
            key=lambda c: cv2.contourArea(c.astype(int)),
            reverse=True,
        )

    # ... (implement _analyze_outer, _analyze_core, _calc_concentricity, _visualize)

# -----------------------------------------------------------------------------
# Specialty Fiber Analysis
# -----------------------------------------------------------------------------

@AnalysisFactory.register(AnalysisType.SPECIALTY)
class SpecialtyFiberAnalysis(AbstractAnalysis):
    """
    "Type 2" analysis routing for specialty fiber geometries.
    """
    def __init__(
        self,
        image_path: Path,
        output_path: Optional[Path] = None,
        fiber_type: SpecialtyFiberType = SpecialtyFiberType.PM_FIBER,
        min_area: int = 100,
        **kwargs,
    ) -> None:
        super().__init__(image_path, output_path)
        self.fiber_type = fiber_type
        self.min_area = min_area
        self.params = kwargs

    def execute(self) -> None:
        dispatch: Dict[SpecialtyFiberType, Any] = {
            SpecialtyFiberType.PM_FIBER: self._analyze_pm,
            SpecialtyFiberType.HOLLOW_CORE: self._analyze_hollow,
            SpecialtyFiberType.MULTI_CORE: self._analyze_multi,
        }
        func = dispatch.get(self.fiber_type)
        if not func:
            raise AnalysisError(f"No implementation for {self.fiber_type}")
        rgb, binary = self._load_and_prepare_image()
        cleaned = morphology.remove_small_objects(binary.astype(bool), self.min_area)
        contours = self._find_contours(cleaned.astype(np.uint8))
        self.results, self.annotated_image = func(contours, rgb)
        logger.info(f"Specialty analysis ({self.fiber_type.value}) complete.")

    def _find_contours(self, binary: np.ndarray) -> List[np.ndarray]:
        raw = measure.find_contours(binary, level=0.8)
        filtered = [c for c in raw if cv2.contourArea(c.astype(int)) > self.min_area]
        return sorted(
            filtered,
            key=lambda c: cv2.contourArea(c.astype(int)),
            reverse=True,
        )

    # ... (implement _analyze_pm, _analyze_hollow, _analyze_multi)

# -----------------------------------------------------------------------------
# Internal Self-Tests
# -----------------------------------------------------------------------------

def _run_tests() -> None:
    logger.info("--- Running Internal Self-Tests ---")
    test_file = Path("test_image.png")
    img = np.zeros((300, 300), dtype=np.uint8)
    cv2.circle(img, (150, 150), 100, 255, -1)
    cv2.circle(img, (155, 155), 20, 0, -1)
    io.imsave(test_file, img)

    try:
        poly = AnalysisFactory.create(
            AnalysisType.POLYGONAL, image_path=test_file
        )
        poly.execute()
        assert "circumscribed_circle_diameter" in poly.results
        logger.info("✅ Polygonal test passed")

        spec = AnalysisFactory.create(
            AnalysisType.SPECIALTY,
            image_path=test_file,
            fiber_type=SpecialtyFiberType.HOLLOW_CORE,
        )
        spec.execute()
        assert "hollow_core_diameter" in spec.results
        logger.info("✅ Specialty test passed")
    except Exception as e:
        logger.error(f"Test failure: {e}", exc_info=True)
    finally:
        test_file.unlink(missing_ok=True)
        logger.info("--- Tests Completed ---")

# -----------------------------------------------------------------------------
# CLI Driver
# -----------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fiber Cross-Section Analysis Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("image_path", nargs="?", type=Path,
                        help="Path to the image file to analyze.")
    parser.add_argument(
        "--analysis", type=AnalysisType,
        choices=list(AnalysisType),
        help="Type of analysis to perform.",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Path to save the annotated image.",
    )
    parser.add_argument(
        "--fiber_type", type=SpecialtyFiberType,
        choices=list(SpecialtyFiberType), default=SpecialtyFiberType.PM_FIBER,
        help="Specific geometry for specialty analysis.",
    )
    parser.add_argument(
        "--poly_tolerance", type=float, default=5.0,
        help="Tolerance for polygon approximation (Type 1).",
    )
    parser.add_argument(
        "--min_area", type=int, default=100,
        help="Minimum contour area to consider.",
    )
    parser.add_argument(
        "--num_cores", type=int, default=None,
        help="Expected number of cores for multi-core analysis.",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Run internal self-tests and exit.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable DEBUG-level logging.",
    )
    parser.add_argument(
        "--version", action="version",
        version=f"%(prog)s {metadata.version('fiber-microscope-toolkit')}"
    )

    args = parser.parse_args()
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    if args.test:
        _run_tests()
        return 0

    if not args.image_path or not args.analysis:
        parser.error("--image_path and --analysis are required unless --test is used.")

    try:
        kwargs: Dict[str, Any] = {
            "image_path": args.image_path,
            "output_path": args.out,
            "min_area": args.min_area,
        }
        if args.analysis == AnalysisType.POLYGONAL:
            kwargs["poly_tolerance"] = args.poly_tolerance
        else:
            kwargs["fiber_type"] = args.fiber_type
            kwargs["num_cores"] = args.num_cores

        analyzer = AnalysisFactory.create(args.analysis, **kwargs)
        analyzer.execute()

        print(pd.DataFrame.from_dict(
            analyzer.results, orient="index", columns=["Value"]
        ))
        analyzer.save_results(args.image_path.stem)

    except (FileNotFoundError, AnalysisError) as e:
        logger.error(f"Error: {e}")
        return 1
    except Exception:
        logger.exception("Unexpected error")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
