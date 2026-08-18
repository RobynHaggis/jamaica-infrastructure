"""
Reclassify and allocate Landslide Susceptibility.

Maps land-cover classes to three attributes used in landslide susceptibility
modelling:
    - FOREST_PRO : forest protection factor (0, 0.5, 1)
    - AFFOREST_P : afforestation potential (0, 0.5, 1)
    - LS_SUSCEPT : landslide susceptibility class (0-5)

The lookup table mapping class names to these three attributes is supplied
by the caller — see the notebook for the values used in this study.

Reference for the classification scheme:
    Palau, R. M., Nadim, F., Paulsen, E., & Storrøsten, E. (2023).
    A new model for global landslide susceptibility assessment and
    scenario-based hazard assessment. Global Infrastructure Resilience
    Index (GIRI), CDRI.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple, Union

import geopandas as gpd
import numpy as np


def load_lookup_from_json(path: Union[str, Path]) -> Dict[str, Tuple[float, float, int]]:
    """Load a lookup table from a JSON file.

    The JSON file should map class name strings to 3-element arrays, e.g.:

        {"Secondary Forest": [1, 0, 3], ...}
    """
    with open(path) as f:
        raw = json.load(f)
    return {k: tuple(v) for k, v in raw.items()}


def reclassify_landslide_susceptibility(
    input_shp: Union[str, Path],
    output_shp: Union[str, Path],
    lookup: Dict[str, Tuple[float, float, int]],
    classify_col: str = "Classify",
    verbose: bool = True,
) -> gpd.GeoDataFrame:
    """
    Reclassify a land-cover shapefile and allocate landslide susceptibility
    attributes based on a lookup table.

    Parameters
    ----------
    input_shp : str or Path
        Path to the input land-cover shapefile. Must contain the column
        specified by ``classify_col``.
    output_shp : str or Path
        Path to write the reclassified shapefile.
    lookup : dict
        Mapping from land-cover class name to a
        ``(FOREST_PRO, AFFOREST_P, LS_SUSCEPT)`` tuple. Supplied by the
        caller (e.g. defined in the notebook or loaded via
        ``load_lookup_from_json``) so the classification scheme stays
        with the study context.
    classify_col : str, default "Classify"
        Name of the land-cover class column in the input shapefile.
    verbose : bool, default True
        If True, print summary messages (unmatched classes, save path, sample).

    Returns
    -------
    geopandas.GeoDataFrame
        The reclassified GeoDataFrame with added columns ``FOREST_PRO``,
        ``AFFOREST_P``, and ``LS_SUSCEPT``.
    """
    input_shp = Path(input_shp)
    output_shp = Path(output_shp)
    output_shp.parent.mkdir(parents=True, exist_ok=True)

    # --- Load shapefile ---
    gdf = gpd.read_file(input_shp)

    if classify_col not in gdf.columns:
        raise KeyError(
            f"Column '{classify_col}' not found in {input_shp}. "
            f"Available columns: {list(gdf.columns)}"
        )

    # Strip whitespace from classify column to avoid mismatches
    gdf["_classify_clean"] = gdf[classify_col].str.strip()

    # Build a whitespace-stripped version of the lookup for matching.
    stripped_lookup = {k.strip(): v for k, v in lookup.items()}

    gdf["FOREST_PRO"] = gdf["_classify_clean"].map(
        lambda x: stripped_lookup.get(x, (None, None, None))[0]
    )
    gdf["AFFOREST_P"] = gdf["_classify_clean"].map(
        lambda x: stripped_lookup.get(x, (None, None, None))[1]
    )
    gdf["LS_SUSCEPT"] = gdf["_classify_clean"].map(
        lambda x: stripped_lookup.get(x, (None, None, None))[2]
    )

    # --- Check for unmatched classes ---
    unmatched = gdf[gdf["FOREST_PRO"].isna()][classify_col].unique()
    if verbose:
        if len(unmatched) > 0:
            print(f"WARNING: {len(unmatched)} unmatched class(es):")
            for c in unmatched:
                print(f"  '{c}'")
        else:
            print("All classes matched successfully.")

    # Drop temp column
    gdf = gdf.drop(columns=["_classify_clean"])

    # --- Handle float precision/width to avoid shapefile write warnings ---
    # Shapefile numeric fields have a fixed width; large float values
    # (e.g. areas in square metres over ~10 million) can exceed the default
    # field width and trigger pyogrio RuntimeWarnings. Any float column whose
    # max absolute value exceeds the safe threshold is rounded to integer
    # (which is fine for areas/lengths in m² or m — sub-metre precision is
    # rarely meaningful at this scale).
    _SHAPEFILE_SAFE_FLOAT_MAX = 1e7  # ~10 million: comfortably fits default width

    for col in gdf.select_dtypes(include=["float64", "float32"]).columns:
        col_max = gdf[col].abs().max()
        if np.isfinite(col_max) and col_max > _SHAPEFILE_SAFE_FLOAT_MAX:
            if verbose:
                print(
                    f"  Rounding '{col}' to int (max value {col_max:.1f} "
                    f"exceeds shapefile safe width)."
                )
            gdf[col] = gdf[col].round().astype("Int64")

    # --- Save ---
    gdf.to_file(output_shp)

    if verbose:
        print(f"Saved to {output_shp}")
        print("\nSample output:")
        print(gdf[[classify_col, "FOREST_PRO", "AFFOREST_P", "LS_SUSCEPT"]].head(10))

    return gdf


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Reclassify land-cover shapefile with landslide susceptibility attributes."
    )
    parser.add_argument("input_shp", type=str, help="Path to input land-cover shapefile.")
    parser.add_argument("output_shp", type=str, help="Path to output reclassified shapefile.")
    parser.add_argument(
        "lookup_json",
        type=str,
        help="Path to JSON file mapping class -> [FOREST_PRO, AFFOREST_P, LS_SUSCEPT].",
    )
    parser.add_argument(
        "--classify-col",
        type=str,
        default="Classify",
        help="Name of the land-cover class column (default: 'Classify').",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress messages.")
    args = parser.parse_args()

    lookup = load_lookup_from_json(args.lookup_json)

    reclassify_landslide_susceptibility(
        input_shp=args.input_shp,
        output_shp=args.output_shp,
        lookup=lookup,
        classify_col=args.classify_col,
        verbose=not args.quiet,
    )
    