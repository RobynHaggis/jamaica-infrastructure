"""
Extract raster values at training point locations.

Reads a CSV of training points with latitude/longitude (or an existing
GeoDataFrame), samples each supplied raster at those points, and writes out
an enriched CSV with one column per raster.

Uses rasterio.sample for speed (~100-1000× faster than per-point xarray
lookups) and explicitly masks nodata values as NaN so they don't poison
downstream model training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio


def sample_raster_at_points(
    raster_path: Union[str, Path],
    gdf: gpd.GeoDataFrame,
) -> np.ndarray:
    """
    Sample a raster at the geometry locations of a GeoDataFrame.

    Reprojects points to the raster's CRS if needed. Returns a float array
    the same length as ``gdf``, with NaN for points falling outside the
    raster bounds or on nodata pixels.
    """
    raster_path = Path(raster_path)

    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        nodata = src.nodata

        # Reproject points if needed
        if gdf.crs != raster_crs:
            pts = gdf.to_crs(raster_crs)
        else:
            pts = gdf

        # Build coordinate tuple list — rasterio.sample wants (x, y) pairs
        coords = [(geom.x, geom.y) if geom is not None and not geom.is_empty
                  else (np.nan, np.nan)
                  for geom in pts.geometry]

        # sample() returns a generator of 1-element arrays (one per band)
        values = np.array(
            [v[0] if v is not None and len(v) > 0 else np.nan
             for v in src.sample(coords)],
            dtype="float64",
        )

    # Mask nodata values
    if nodata is not None:
        values[values == nodata] = np.nan

    # Also catch common sentinel nodata values that may not be declared
    values[values == -9999] = np.nan
    values[values == 32767] = np.nan

    return values


def extract_training_features(
    input_csv: Union[str, Path],
    raster_paths: Dict[str, Union[str, Path]],
    output_csv: Union[str, Path],
    lon_col: str = "longitude",
    lat_col: str = "latitude",
    input_crs: str = "EPSG:4326",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load training points and sample each supplied raster at those points.

    Parameters
    ----------
    input_csv : str or Path
        CSV of training points, must have ``lon_col`` and ``lat_col``.
    raster_paths : dict
        Mapping from output column name -> raster path.
        E.g. ``{"slope": "/path/to/slope.tif", "elevation": "/path/dem.tif"}``.
    output_csv : str or Path
        Path to write the enriched CSV. Must include the ``.csv`` extension.
    lon_col, lat_col : str
        Column names for longitude and latitude in the input CSV.
    input_crs : str, default "EPSG:4326"
        CRS of the input coordinates.
    verbose : bool, default True
        Print progress per raster and a summary of NaN counts.

    Returns
    -------
    pandas.DataFrame
        The enriched training table (also written to ``output_csv``).
    """
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if output_csv.suffix.lower() != ".csv":
        raise ValueError(
            f"output_csv should end in .csv, got: {output_csv.name}"
        )

    # Load training points
    training = pd.read_csv(input_csv)
    if verbose:
        print(f"Loaded {len(training):,} training points from {input_csv.name}")

    for col in (lon_col, lat_col):
        if col not in training.columns:
            raise KeyError(
                f"Column '{col}' not found in input CSV. "
                f"Available: {list(training.columns)}"
            )

    gdf = gpd.GeoDataFrame(
        training,
        geometry=gpd.points_from_xy(training[lon_col], training[lat_col]),
        crs=input_crs,
    )

    # Sample each raster
    for col_name, raster_path in raster_paths.items():
        if verbose:
            print(f"  Sampling {col_name!r} from {Path(raster_path).name}...")
        values = sample_raster_at_points(raster_path, gdf)
        training[col_name] = values
        if verbose:
            n_nan = int(np.isnan(values).sum())
            print(
                f"    → extracted {len(values) - n_nan:,} values, "
                f"{n_nan:,} NaN "
                f"({n_nan / len(values) * 100:.1f}%)"
            )

    # Write out (drop geometry; write just the tabular data)
    training.to_csv(output_csv, index=False)
    if verbose:
        print(f"\n✓ Saved enriched training data to {output_csv}")

    return training


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Sample rasters at training points and write an enriched CSV."
    )
    parser.add_argument("input_csv", type=str, help="Input training CSV.")
    parser.add_argument("output_csv", type=str, help="Output enriched CSV.")
    parser.add_argument(
        "raster_paths_json",
        type=str,
        help="JSON mapping {column_name: raster_path}.",
    )
    parser.add_argument("--lon-col", type=str, default="longitude")
    parser.add_argument("--lat-col", type=str, default="latitude")
    parser.add_argument("--input-crs", type=str, default="EPSG:4326")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    with open(args.raster_paths_json) as f:
        raster_paths = json.load(f)

    extract_training_features(
        input_csv=args.input_csv,
        raster_paths=raster_paths,
        output_csv=args.output_csv,
        lon_col=args.lon_col,
        lat_col=args.lat_col,
        input_crs=args.input_crs,
        verbose=not args.quiet,
    )