"""
Generate terrain derivative rasters from a projected DEM:
    - Aspect (degrees clockwise from north, via gdaldem)
    - Plan curvature (m⁻¹, perpendicular to slope direction)
    - Profile curvature (m⁻¹, along slope direction)
    - Topographic Wetness Index (dimensionless, via richdem or pysheds)

Curvatures follow Zevenbergen & Thorne (1987) — the same method used by
ArcGIS, SAGA (default), and RichDEM. A quadratic surface is fit exactly
through the 3×3 neighbourhood of each cell and the second derivatives are
extracted analytically.

TWI = ln(a / tan(β)) where `a` is specific catchment area (m²/m) and `β`
is local slope (radians). High TWI values indicate convergent, low-gradient
areas where water accumulates; low values indicate steep, divergent terrain.

Convention (Zevenbergen & Thorne, matching ArcGIS):
    - Profile curvature: NEGATIVE = upwardly convex (accelerating flow),
      POSITIVE = upwardly concave (decelerating flow).
    - Plan curvature: POSITIVE = laterally convex (diverging flow, ridges),
      NEGATIVE = laterally concave (converging flow, hollows).
    - Zero = planar.

Curvature values are in m⁻¹. They are typically small; some tools multiply
by 100 for display. This module saves raw m⁻¹ values.

References:
    Zevenbergen, L.W. & Thorne, C.R. (1987). Quantitative analysis of land
    surface topography. Earth Surface Processes and Landforms 12, 47–56.
    Beven, K.J. & Kirkby, M.J. (1979). A physically based, variable
    contributing area model of basin hydrology. Hydrological Sciences
    Bulletin 24(1), 43–69.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import rasterio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_projected(dem_path: Path) -> int:
    """Open the DEM and verify it's in a projected CRS. Returns EPSG code."""
    with rasterio.open(dem_path) as src:
        if src.crs is None:
            raise ValueError(f"{dem_path} has no CRS defined.")
        if src.crs.is_geographic:
            raise ValueError(
                f"{dem_path} is in a geographic CRS ({src.crs}). "
                "Reproject to a projected CRS (e.g. EPSG:3448) first."
            )
        return src.crs.to_epsg()


def _read_dem(dem_path: Path) -> Tuple[np.ndarray, dict, float, float]:
    """Load DEM as float64, mask nodata as NaN, return array + profile + pixel sizes."""
    with rasterio.open(dem_path) as src:
        arr = src.read(1).astype("float64")
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        profile = src.profile.copy()
        dx = abs(src.res[0])
        dy = abs(src.res[1])
    return arr, profile, dx, dy


def _write_float_raster(
    data: np.ndarray, profile: dict, output_path: Path, nodata: float = -9999.0
) -> None:
    """Write a float32 raster with a given nodata value."""
    out = data.astype("float32")
    out[~np.isfinite(out)] = nodata
    profile.update(dtype="float32", nodata=nodata, compress="lzw", count=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(out, 1)


def _sanity_stats(data: np.ndarray, label: str, units: str = "") -> None:
    """Print a quick distribution summary of a raster array."""
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        print(f"  {label}: all NaN")
        return
    pcts = np.percentile(finite, [1, 25, 50, 75, 99])
    print(f"  {label}: n={finite.size:,}")
    print(
        f"    min={finite.min():.4g}  max={finite.max():.4g}  "
        f"mean={finite.mean():.4g}  median={np.median(finite):.4g}  {units}"
    )
    print(
        f"    percentiles 1/25/50/75/99: "
        f"{pcts[0]:.3g} / {pcts[1]:.3g} / {pcts[2]:.3g} / "
        f"{pcts[3]:.3g} / {pcts[4]:.3g}"
    )


# ---------------------------------------------------------------------------
# Aspect
# ---------------------------------------------------------------------------

def generate_aspect(
    dem_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    zero_for_flat: bool = False,
    verbose: bool = True,
) -> Path:
    """Generate aspect raster via `gdaldem aspect` (Horn's method, degrees)."""
    dem_path = Path(dem_path)
    crs_code = _check_projected(dem_path)

    if output_path is None:
        output_path = dem_path.parent / f"Jamaica_Aspect_{crs_code}.tif"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = ["gdaldem", "aspect", str(dem_path), str(output_path), "-compute_edges"]
    if zero_for_flat:
        cmd.append("-zero_for_flat")
    if verbose:
        print(f"[aspect] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    if verbose:
        with rasterio.open(output_path) as src:
            arr = src.read(1).astype("float64")
            if src.nodata is not None:
                arr[arr == src.nodata] = np.nan
        print(f"  → {output_path}")
        _sanity_stats(arr, "aspect", "(degrees from N)")

    return output_path


# ---------------------------------------------------------------------------
# Zevenbergen & Thorne curvature (plan and profile)
# ---------------------------------------------------------------------------
# Surface fit through 3×3 neighbourhood labelled:
#       Z1 Z2 Z3
#       Z4 Z5 Z6
#       Z7 Z8 Z9
# Centre cell = Z5. Partial derivatives (from Zevenbergen & Thorne 1987):
#   D = ((Z4 + Z6) / 2 - Z5) / L²     (∂²z/∂x²)
#   E = ((Z2 + Z8) / 2 - Z5) / L²     (∂²z/∂y²)
#   F = (-Z1 + Z3 + Z7 - Z9) / (4L²)  (∂²z/∂x∂y)
#   G = (-Z4 + Z6) / (2L)             (∂z/∂x)
#   H = (Z2 - Z8) / (2L)              (∂z/∂y)
# where L is grid spacing. If dx ≠ dy, use dx in x-terms and dy in y-terms.
#
# Plan curvature    = -2 * (D*H² + E*G² - F*G*H) / (G² + H²)
# Profile curvature = -2 * (D*G² + E*H² + F*G*H) / (G² + H²)
# (Sign conventions per ArcGIS / spatialEco; undefined where slope = 0.)


def _compute_curvatures(
    dem: np.ndarray, dx: float, dy: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute plan and profile curvature from a DEM array.

    Returns (plan_curvature, profile_curvature), both with same shape as dem,
    in units of m⁻¹. Edge cells and flat (slope=0) cells are NaN.
    """
    # Build shifted arrays for the 3×3 neighbourhood. Pad with NaN so edges
    # naturally become NaN after shifting (no wrap-around like np.roll).
    z = np.pad(dem, 1, mode="constant", constant_values=np.nan)
    Z1 = z[:-2, :-2]   # top-left
    Z2 = z[:-2, 1:-1]  # top
    Z3 = z[:-2, 2:]    # top-right
    Z4 = z[1:-1, :-2]  # left
    Z5 = z[1:-1, 1:-1]  # centre
    Z6 = z[1:-1, 2:]   # right
    Z7 = z[2:, :-2]    # bottom-left
    Z8 = z[2:, 1:-1]   # bottom
    Z9 = z[2:, 2:]     # bottom-right

    # Partial derivatives (anisotropic pixel-size form)
    D = ((Z4 + Z6) / 2.0 - Z5) / (dx * dx)
    E = ((Z2 + Z8) / 2.0 - Z5) / (dy * dy)
    F = (-Z1 + Z3 + Z7 - Z9) / (4.0 * dx * dy)
    G = (-Z4 + Z6) / (2.0 * dx)
    H = (Z2 - Z8) / (2.0 * dy)

    p2_q2 = G * G + H * H  # = p² + q² in Z&T notation

    # Suppress divide-by-zero warnings; we'll mask flat cells afterwards
    with np.errstate(divide="ignore", invalid="ignore"):
        plan = -2.0 * (D * H * H + E * G * G - F * G * H) / p2_q2
        profile = -2.0 * (D * G * G + E * H * H + F * G * H) / p2_q2

    # Flat cells (slope ≈ 0) have undefined curvature
    flat = p2_q2 < 1e-12
    plan[flat] = np.nan
    profile[flat] = np.nan

    return plan, profile


def generate_curvatures(
    dem_path: Union[str, Path],
    plan_output: Optional[Union[str, Path]] = None,
    profile_output: Optional[Union[str, Path]] = None,
    verbose: bool = True,
) -> Tuple[Path, Path]:
    """Generate plan and profile curvature rasters (Zevenbergen & Thorne 1987)."""
    dem_path = Path(dem_path)
    crs_code = _check_projected(dem_path)

    if plan_output is None:
        plan_output = dem_path.parent / f"Jamaica_PlanCurv_{crs_code}.tif"
    if profile_output is None:
        profile_output = dem_path.parent / f"Jamaica_ProfileCurv_{crs_code}.tif"
    plan_output = Path(plan_output)
    profile_output = Path(profile_output)

    if verbose:
        print(f"[curvature] Reading DEM: {dem_path.name}")
    dem, profile, dx, dy = _read_dem(dem_path)
    if verbose:
        print(f"  Pixel size: dx={dx:.2f} m, dy={dy:.2f} m  (shape: {dem.shape})")

    if verbose:
        print("[curvature] Computing plan and profile curvature (Zevenbergen & Thorne 1987)...")
    plan, prof = _compute_curvatures(dem, dx, dy)

    _write_float_raster(plan, profile, plan_output)
    _write_float_raster(prof, profile, profile_output)

    if verbose:
        print(f"  → {plan_output}")
        _sanity_stats(plan, "plan curvature", "(m⁻¹, +laterally convex / -concave)")
        print(f"  → {profile_output}")
        _sanity_stats(
            prof, "profile curvature", "(m⁻¹, -upwardly convex / +concave)"
        )

    return plan_output, profile_output


# ---------------------------------------------------------------------------
# Topographic Wetness Index (TWI)
# ---------------------------------------------------------------------------
# TWI = ln(a / tan(β))
# where:
#   a = specific catchment area (m²/m) = flow accumulation × pixel_area / contour_length
#   β = local slope (radians)
#
# Workflow:
#   1. Fill depressions in the DEM (pit-filling) so flow can route to edges.
#   2. Compute flow accumulation (number of upslope cells draining to each cell).
#   3. Compute slope in radians.
#   4. TWI = ln((accum * pixel_area / contour_length) / tan(slope))
#
# Implementation uses richdem if available (fast, handles pit-filling and
# flow routing internally); falls back to pysheds otherwise.


def _compute_twi_richdem(
    dem_path: Path, output_path: Path, verbose: bool = True
) -> None:
    """Compute TWI using richdem."""
    import richdem as rd

    if verbose:
        print("[TWI] Using richdem backend")
        print("  Loading DEM...")
    dem = rd.LoadGDAL(str(dem_path))
    if verbose:
        print("  Filling depressions...")
    dem_filled = rd.FillDepressions(dem, in_place=False)

    if verbose:
        print("  Computing flow accumulation (D∞)...")
    accum = rd.FlowAccumulation(dem_filled, method="Dinf")

    if verbose:
        print("  Computing slope (radians)...")
    slope_deg = rd.TerrainAttribute(dem_filled, attrib="slope_degrees")
    slope_rad = np.deg2rad(np.asarray(slope_deg))

    # Pixel size (assumed isotropic; take min if not)
    with rasterio.open(dem_path) as src:
        dx = abs(src.res[0])
        dy = abs(src.res[1])
        profile = src.profile.copy()
    pixel_area = dx * dy
    contour_length = (dx + dy) / 2.0  # approximation for anisotropic grids

    # Specific catchment area: a = accumulated_cells * pixel_area / contour_length
    # richdem FlowAccumulation returns number of upslope cells (incl. self)
    accum_arr = np.asarray(accum)
    sca = (accum_arr * pixel_area) / contour_length

    # TWI with a small floor on tan(slope) to avoid division by zero on flats.
    # Flats get the minimum tan-slope value (corresponding to ~0.1° slope).
    min_tan = np.tan(np.deg2rad(0.1))
    tan_slope = np.maximum(np.tan(slope_rad), min_tan)

    with np.errstate(invalid="ignore", divide="ignore"):
        twi = np.log(sca / tan_slope)

    _write_float_raster(twi, profile, output_path)


def _compute_twi_pysheds(
    dem_path: Path, output_path: Path, verbose: bool = True
) -> None:
    """Compute TWI using pysheds (pure-Python fallback, D8 flow routing)."""
    from pysheds.grid import Grid

    if verbose:
        print("[TWI] Using pysheds backend (D8 flow routing)")
        print("  Loading DEM...")
    grid = Grid.from_raster(str(dem_path))
    dem = grid.read_raster(str(dem_path))

    if verbose:
        print("  Filling pits...")
    pit_filled = grid.fill_pits(dem)
    if verbose:
        print("  Filling depressions (this is the slow step for large DEMs)...")
    flooded = grid.fill_depressions(pit_filled)
    if verbose:
        print("  Resolving flats...")
    inflated = grid.resolve_flats(flooded)

    if verbose:
        print("  Computing D8 flow direction...")
    fdir = grid.flowdir(inflated)
    if verbose:
        print("  Computing flow accumulation...")
    accum = grid.accumulation(fdir)

    # Read geo metadata from the original DEM for writing output
    with rasterio.open(dem_path) as src:
        dx = abs(src.res[0])
        dy = abs(src.res[1])
        profile = src.profile.copy()

    # Slope from the hydrologically-corrected DEM
    if verbose:
        print("  Computing slope...")
    dem_arr = np.asarray(inflated, dtype="float64")
    # Mask any residual nodata values
    if profile.get("nodata") is not None:
        dem_arr[dem_arr == profile["nodata"]] = np.nan

    dzdy, dzdx = np.gradient(dem_arr, dy, dx)
    slope_rad = np.arctan(np.sqrt(dzdx ** 2 + dzdy ** 2))

    pixel_area = dx * dy
    contour_length = (dx + dy) / 2.0

    # Specific catchment area. pysheds accumulation counts upstream cells
    # (not including self), so we add 1.
    accum_arr = np.asarray(accum, dtype="float64") + 1.0
    sca = (accum_arr * pixel_area) / contour_length

    # Avoid log(0) and divide-by-zero on flats
    min_tan = np.tan(np.deg2rad(0.1))
    tan_slope = np.maximum(np.tan(slope_rad), min_tan)

    with np.errstate(invalid="ignore", divide="ignore"):
        twi = np.log(sca / tan_slope)

    _write_float_raster(twi, profile, output_path)


def generate_twi(
    dem_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    backend: str = "auto",
    verbose: bool = True,
) -> Path:
    """
    Generate a Topographic Wetness Index raster.

    Parameters
    ----------
    dem_path : str or Path
        Input DEM in a projected CRS with metre units.
    output_path : str or Path, optional
        Output path. Defaults to ``<dem_dir>/Jamaica_TWI_<CRS>.tif``.
    backend : {"auto", "richdem", "pysheds"}
        Which package to use. "auto" tries richdem first, then pysheds.
    verbose : bool, default True

    Returns
    -------
    Path to the TWI raster (float32, dimensionless).
    """
    dem_path = Path(dem_path)
    crs_code = _check_projected(dem_path)

    if output_path is None:
        output_path = dem_path.parent / f"Jamaica_TWI_{crs_code}.tif"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resolve backend
    def _try_richdem():
        """Verify richdem imports AND its native engine actually loads.

        The `import richdem` check is not enough — on Python 3.11, richdem
        often imports successfully while its C++ engine fails to load and
        GDAL bindings are missing. We test an actual call that exercises
        the engine to catch these cases.
        """
        try:
            import richdem as rd
            # The import prints "COULD NOT LOAD RichDEM ENGINE!" if broken.
            # Calling a function like rd.rdarray triggers the actual failure.
            import numpy as np
            test = rd.rdarray(np.zeros((3, 3), dtype="float32"), no_data=-9999)
            rd.TerrainAttribute(test, attrib="slope_degrees")
            # Also verify GDAL is available for LoadGDAL
            if not getattr(rd, "GDAL_AVAILABLE", False):
                return False
            return True
        except Exception:
            return False

    def _try_pysheds():
        try:
            import pysheds  # noqa: F401
            return True
        except ImportError:
            return False

    if backend == "richdem":
        if not _try_richdem():
            raise ImportError("richdem is not installed. `pip install richdem`")
        _compute_twi_richdem(dem_path, output_path, verbose)
    elif backend == "pysheds":
        if not _try_pysheds():
            raise ImportError("pysheds is not installed. `pip install pysheds`")
        _compute_twi_pysheds(dem_path, output_path, verbose)
    elif backend == "auto":
        if _try_richdem():
            _compute_twi_richdem(dem_path, output_path, verbose)
        elif _try_pysheds():
            _compute_twi_pysheds(dem_path, output_path, verbose)
        else:
            raise ImportError(
                "Neither richdem nor pysheds is installed. Install one with:\n"
                "  pip install richdem    # (recommended)\n"
                "  pip install pysheds    # (pure-Python alternative)"
            )
    else:
        raise ValueError(f"Unknown backend: {backend!r}")

    # Sanity-check
    if verbose:
        with rasterio.open(output_path) as src:
            arr = src.read(1).astype("float64")
            if src.nodata is not None:
                arr[arr == src.nodata] = np.nan
        print(f"  → {output_path}")
        _sanity_stats(arr, "TWI", "(dimensionless; higher = wetter)")
        print(
            "  Expected: mean ~5–10, high values (>12) in valleys/hollows, "
            "low values (<4) on steep ridges."
        )

    return output_path


# ---------------------------------------------------------------------------
# Convenience: generate all in one call
# ---------------------------------------------------------------------------

def generate_all_terrain_derivatives(
    dem_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    include_twi: bool = True,
    twi_backend: str = "auto",
    verbose: bool = True,
) -> dict:
    """
    Generate aspect, plan curvature, profile curvature, and TWI from a
    projected DEM.

    Parameters
    ----------
    dem_path : str or Path
        Input DEM (projected CRS, metres).
    output_dir : str or Path, optional
        Where to save the output rasters. Defaults to the DEM's parent dir.
    include_twi : bool, default True
        If False, skip TWI (useful if richdem/pysheds aren't installed).
    twi_backend : {"auto", "richdem", "pysheds"}
        Backend for TWI computation.
    verbose : bool, default True

    Returns a dict with keys 'aspect', 'plan_curvature', 'profile_curvature',
    and (if include_twi) 'twi', each mapping to the output raster Path.
    """
    dem_path = Path(dem_path)
    crs_code = _check_projected(dem_path)

    if output_dir is None:
        output_dir = dem_path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    aspect_path = output_dir / f"Jamaica_Aspect_{crs_code}.tif"
    plan_path = output_dir / f"Jamaica_PlanCurv_{crs_code}.tif"
    profile_path = output_dir / f"Jamaica_ProfileCurv_{crs_code}.tif"
    twi_path = output_dir / f"Jamaica_TWI_{crs_code}.tif"

    generate_aspect(dem_path, output_path=aspect_path, verbose=verbose)
    generate_curvatures(
        dem_path,
        plan_output=plan_path,
        profile_output=profile_path,
        verbose=verbose,
    )

    result = {
        "aspect": aspect_path,
        "plan_curvature": plan_path,
        "profile_curvature": profile_path,
    }

    if include_twi:
        generate_twi(
            dem_path, output_path=twi_path, backend=twi_backend, verbose=verbose
        )
        result["twi"] = twi_path

    if verbose:
        print("\n✓ All terrain derivatives generated.")

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate aspect, curvatures, and TWI from a projected DEM."
    )
    parser.add_argument("dem", type=str, help="Input DEM (projected CRS).")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--no-twi", action="store_true", help="Skip TWI generation.")
    parser.add_argument("--twi-backend", type=str, default="auto",
                        choices=["auto", "richdem", "pysheds"])
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    generate_all_terrain_derivatives(
        dem_path=args.dem,
        output_dir=args.output_dir,
        include_twi=not args.no_twi,
        twi_backend=args.twi_backend,
        verbose=not args.quiet,
    )