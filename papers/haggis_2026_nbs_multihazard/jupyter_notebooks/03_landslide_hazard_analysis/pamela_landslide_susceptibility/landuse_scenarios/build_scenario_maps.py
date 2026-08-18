"""Build land-use scenario map products."""

"""
Build reafforestation and deforestation scenario maps for landslide
susceptibility modelling.

Only polygons whose mean slope exceeds a threshold (default 40°) are
converted. Polygons flagged as fully convertible (AFFOREST_P == 1 or
FOREST_PRO == 1) are reclassified entirely; polygons flagged as partially
convertible (value == 0.5) are split in half along their longest axis, with
one half reclassified and the other left as-is.
"""

from __future__ import annotations

import os
import subprocess
import warnings
from pathlib import Path
from typing import Tuple, Union

import geopandas as gpd
import numpy as np
from rasterstats import zonal_stats
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box
from shapely.validation import make_valid

warnings.filterwarnings("ignore", category=UserWarning)


# --- Defaults ---
DEFAULT_SLOPE_THRESHOLD = 40  # degrees
DEFAULT_TARGET_CRS = "EPSG:3448"  # Jamaica Metric Grid

# Shapefile numeric fields have a fixed width; values above ~10 million can
# exceed the default width and trigger pyogrio RuntimeWarnings.
_SHAPEFILE_SAFE_FLOAT_MAX = 1e7


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _extract_polygons(geom):
    """Extract only polygon parts from a geometry."""
    if geom.is_empty:
        return geom
    if isinstance(geom, (Polygon, MultiPolygon)):
        return geom
    if isinstance(geom, GeometryCollection):
        polys = [g for g in geom.geoms if isinstance(g, (Polygon, MultiPolygon))]
        if len(polys) == 0:
            return Polygon()
        if len(polys) == 1:
            return polys[0]
        return MultiPolygon(polys)
    return geom


def split_polygon_half(geom):
    """Split a polygon approximately in half using bounding-box bisection."""
    geom = make_valid(geom)
    minx, miny, maxx, maxy = geom.bounds

    if (maxx - minx) >= (maxy - miny):
        mid = (minx + maxx) / 2
        box1 = box(minx - 1, miny - 1, mid, maxy + 1)
        box2 = box(mid, miny - 1, maxx + 1, maxy + 1)
    else:
        mid = (miny + maxy) / 2
        box1 = box(minx - 1, miny - 1, maxx + 1, mid)
        box2 = box(minx - 1, mid, maxx + 1, maxy + 1)

    half1 = _extract_polygons(geom.intersection(box1))
    half2 = _extract_polygons(geom.intersection(box2))
    return half1, half2


# ---------------------------------------------------------------------------
# Shapefile-safety helper
# ---------------------------------------------------------------------------

def _sanitize_large_floats(gdf: gpd.GeoDataFrame, verbose: bool = True) -> gpd.GeoDataFrame:
    """Round float columns whose max value exceeds the shapefile safe width.

    Writes large floats as nullable Int64 to avoid pyogrio fixed-width
    RuntimeWarnings (e.g. for Shape_Area, Area new).
    """
    for col in gdf.select_dtypes(include=["float64", "float32"]).columns:
        col_max = gdf[col].abs().max()
        if np.isfinite(col_max) and col_max > _SHAPEFILE_SAFE_FLOAT_MAX:
            if verbose:
                print(
                    f"  Rounding '{col}' to int (max value {col_max:.1f} "
                    f"exceeds shapefile safe width)."
                )
            gdf[col] = gdf[col].round().astype("Int64")
    return gdf


# ---------------------------------------------------------------------------
# Slope raster handling
# ---------------------------------------------------------------------------

def _reproject_slope_raster(
    slope_raster: Union[str, Path],
    target_crs: str,
    verbose: bool = True,
) -> str:
    """Reproject the slope raster to the target CRS if not already cached."""
    slope_raster = str(slope_raster)
    # Build a CRS-tagged filename (e.g. Jamaica_Slope_3448.tif)
    crs_code = target_crs.split(":")[-1]
    reproj_path = slope_raster.replace(".tif", f"_{crs_code}.tif")

    if not os.path.exists(reproj_path):
        if verbose:
            print(f"Reprojecting slope raster to {target_crs}...")
        subprocess.run(
            ["gdalwarp", "-t_srs", target_crs, "-r", "bilinear",
             slope_raster, reproj_path],
            check=True,
        )
        if verbose:
            print(f"Saved reprojected raster: {reproj_path}")
    elif verbose:
        print(f"Using existing reprojected raster: {reproj_path}")
    return reproj_path


# ---------------------------------------------------------------------------
# Scenario builders
# ---------------------------------------------------------------------------

def _build_reafforestation(
    gdf: gpd.GeoDataFrame,
    slope_threshold: float,
    forest_classify: str,
    forest_ls: int,
) -> Tuple[gpd.GeoDataFrame, int]:
    """Convert convertible polygons on steep slopes to forest."""
    rows = []
    converted = 0

    for _, row in gdf.iterrows():
        on_steep = row["mean_slope"] > slope_threshold

        if row["AFFOREST_P"] == 1 and on_steep:
            new_row = row.copy()
            new_row["Classify"] = forest_classify
            new_row["FOREST_PRO"] = 1
            new_row["AFFOREST_P"] = 0
            new_row["LS_SUSCEPT"] = forest_ls
            rows.append(new_row)
            converted += 1

        elif row["AFFOREST_P"] == 0.5 and on_steep:
            half1, half2 = split_polygon_half(row.geometry)

            forest_row = row.copy()
            forest_row["geometry"] = half1
            forest_row["Classify"] = forest_classify
            forest_row["FOREST_PRO"] = 1
            forest_row["AFFOREST_P"] = 0
            forest_row["LS_SUSCEPT"] = forest_ls

            remain_row = row.copy()
            remain_row["geometry"] = half2
            remain_row["FOREST_PRO"] = row["FOREST_PRO"]
            remain_row["AFFOREST_P"] = 0

            rows.append(forest_row)
            rows.append(remain_row)
            converted += 1

        else:
            rows.append(row.copy())

    out = gpd.GeoDataFrame(rows, crs=gdf.crs).reset_index(drop=True)
    out = out[~out.geometry.is_empty].copy()
    return out, converted


def _build_deforestation(
    gdf: gpd.GeoDataFrame,
    slope_threshold: float,
    agri_classify: str,
    agri_ls: int,
) -> Tuple[gpd.GeoDataFrame, int]:
    """Convert convertible forest polygons on steep slopes to agriculture."""
    rows = []
    converted = 0

    for _, row in gdf.iterrows():
        on_steep = row["mean_slope"] > slope_threshold

        if row["FOREST_PRO"] == 1 and on_steep:
            new_row = row.copy()
            new_row["Classify"] = agri_classify
            new_row["FOREST_PRO"] = 0
            new_row["AFFOREST_P"] = 1
            new_row["LS_SUSCEPT"] = agri_ls
            rows.append(new_row)
            converted += 1

        elif row["FOREST_PRO"] == 0.5 and on_steep:
            half1, half2 = split_polygon_half(row.geometry)

            agri_row = row.copy()
            agri_row["geometry"] = half1
            agri_row["Classify"] = agri_classify
            agri_row["FOREST_PRO"] = 0
            agri_row["AFFOREST_P"] = 1
            agri_row["LS_SUSCEPT"] = agri_ls

            remain_row = row.copy()
            remain_row["geometry"] = half2
            remain_row["AFFOREST_P"] = row["AFFOREST_P"]

            rows.append(agri_row)
            rows.append(remain_row)
            converted += 1

        else:
            rows.append(row.copy())

    out = gpd.GeoDataFrame(rows, crs=gdf.crs).reset_index(drop=True)
    out = out[~out.geometry.is_empty].copy()
    return out, converted


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_scenario_maps(
    input_shp: Union[str, Path],
    slope_raster: Union[str, Path],
    output_reaffor: Union[str, Path],
    output_defor: Union[str, Path],
    forest_classify: str,
    forest_ls: int,
    agri_classify: str,
    agri_ls: int,
    slope_threshold: float = DEFAULT_SLOPE_THRESHOLD,
    target_crs: str = DEFAULT_TARGET_CRS,
    slope_nodata: float = -9999,
    verbose: bool = True,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Build reafforestation and deforestation scenario shapefiles.

    Parameters
    ----------
    input_shp : str or Path
        Reclassified land-cover shapefile (must contain ``Classify``,
        ``FOREST_PRO``, ``AFFOREST_P``, ``LS_SUSCEPT`` columns).
    slope_raster : str or Path
        Slope raster (degrees). Will be reprojected to ``target_crs`` if a
        cached reprojected copy is not already present alongside the source.
    output_reaffor, output_defor : str or Path
        Output shapefile paths for the two scenarios.
    forest_classify : str
        Class name assigned to reafforested areas (e.g. "Secondary Forest").
    forest_ls : int
        LS_SUSCEPT value assigned to reafforested areas.
    agri_classify : str
        Class name assigned to deforested areas.
    agri_ls : int
        LS_SUSCEPT value assigned to deforested areas.
    slope_threshold : float, default 40
        Minimum mean slope (degrees) for a polygon to be eligible for
        conversion.
    target_crs : str, default "EPSG:3448"
        CRS the slope raster should be reprojected to before zonal stats.
    slope_nodata : float, default -9999
        Nodata value to pass to ``rasterstats.zonal_stats``.
    verbose : bool, default True
        If True, print progress and summary messages.

    Returns
    -------
    (gdf_reaffor, gdf_defor) : tuple of GeoDataFrame
    """
    input_shp = Path(input_shp)
    output_reaffor = Path(output_reaffor)
    output_defor = Path(output_defor)
    output_reaffor.parent.mkdir(parents=True, exist_ok=True)
    output_defor.parent.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    gdf = gpd.read_file(input_shp)
    if verbose:
        print(f"Loaded {len(gdf)} features")

    # --- Slope raster: reproject to CRS of the shapefile (or target_crs) ---
    slope_raster_reproj = _reproject_slope_raster(slope_raster, target_crs, verbose)

    # --- Compute mean slope per polygon ---
    if verbose:
        print("Computing mean slope per polygon...")
    slope_stats = zonal_stats(
        gdf, slope_raster_reproj, stats=["mean"], nodata=slope_nodata
    )
    gdf["mean_slope"] = [
        s["mean"] if s["mean"] is not None else 0 for s in slope_stats
    ]

    steep = (gdf["mean_slope"] > slope_threshold).sum()
    if verbose:
        print(
            f"Polygons with mean slope > {slope_threshold}°: "
            f"{steep} / {len(gdf)}"
        )

    # --- Reafforestation ---
    if verbose:
        print("\n--- Building Reafforestation Scenario ---")
    gdf_reaffor, n_reaffor = _build_reafforestation(
        gdf, slope_threshold, forest_classify, forest_ls
    )
    gdf_reaffor = _sanitize_large_floats(gdf_reaffor, verbose=verbose)
    gdf_reaffor.to_file(output_reaffor)
    if verbose:
        print(f"Reafforestation: {n_reaffor} polygons converted on steep slopes")
        print(f"Total features: {len(gdf_reaffor)} → {output_reaffor}")

    # --- Deforestation ---
    if verbose:
        print("\n--- Building Deforestation Scenario ---")
    gdf_defor, n_defor = _build_deforestation(
        gdf, slope_threshold, agri_classify, agri_ls
    )
    gdf_defor = _sanitize_large_floats(gdf_defor, verbose=verbose)
    gdf_defor.to_file(output_defor)
    if verbose:
        print(f"Deforestation: {n_defor} polygons converted on steep slopes")
        print(f"Total features: {len(gdf_defor)} → {output_defor}")

    # --- Summary ---
    if verbose:
        print("\n=== SUMMARY ===")
        print(f"Original: {len(gdf)} features")
        print(f"Reafforestation: {len(gdf_reaffor)} features")
        print(f"Deforestation: {len(gdf_defor)} features")

        for label, scenario in [
            ("Reafforestation", gdf_reaffor),
            ("Deforestation", gdf_defor),
        ]:
            print(f"\n{label} - LS_SUSCEPT distribution:")
            print(scenario["LS_SUSCEPT"].value_counts().sort_index())
            print(f"\n{label} - FOREST_PRO distribution:")
            print(scenario["FOREST_PRO"].value_counts().sort_index())

    return gdf_reaffor, gdf_defor


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build reafforestation and deforestation scenario shapefiles."
    )
    parser.add_argument("input_shp", type=str, help="Input reclassified land-cover shapefile.")
    parser.add_argument("slope_raster", type=str, help="Slope raster (degrees).")
    parser.add_argument("output_reaffor", type=str, help="Output reafforestation shapefile.")
    parser.add_argument("output_defor", type=str, help="Output deforestation shapefile.")
    parser.add_argument("--forest-classify", type=str, required=True,
                        help="Class name assigned to reafforested areas.")
    parser.add_argument("--forest-ls", type=int, required=True,
                        help="LS_SUSCEPT value assigned to reafforested areas.")
    parser.add_argument("--agri-classify", type=str, required=True,
                        help="Class name assigned to deforested areas.")
    parser.add_argument("--agri-ls", type=int, required=True,
                        help="LS_SUSCEPT value assigned to deforested areas.")
    parser.add_argument(
        "--slope-threshold",
        type=float,
        default=DEFAULT_SLOPE_THRESHOLD,
        help=f"Minimum mean slope in degrees (default: {DEFAULT_SLOPE_THRESHOLD}).",
    )
    parser.add_argument(
        "--target-crs",
        type=str,
        default=DEFAULT_TARGET_CRS,
        help=f"CRS for slope reprojection (default: {DEFAULT_TARGET_CRS}).",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output.")
    args = parser.parse_args()

    build_scenario_maps(
        input_shp=args.input_shp,
        slope_raster=args.slope_raster,
        output_reaffor=args.output_reaffor,
        output_defor=args.output_defor,
        forest_classify=args.forest_classify,
        forest_ls=args.forest_ls,
        agri_classify=args.agri_classify,
        agri_ls=args.agri_ls,
        slope_threshold=args.slope_threshold,
        target_crs=args.target_crs,
        verbose=not args.quiet,
    )