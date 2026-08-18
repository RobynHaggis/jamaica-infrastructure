#!/usr/bin/env python
"""Allocate landslide source-ID avoided EADs to forest/restoration footprints.

This step starts from the source-ID attribution outputs created by
``landslide_source_id_attribution_combined_class.py``. It rasterizes the
land-cover layer used in the river-flood NbS work onto the landslide source-ID
grid:

* existing/protected forest footprint: ``forest_flood_equivalent_values > 0``
* restoration footprint: ``afforestable_values > 0``

Within each source zone, the source-level positive avoided EAD is allocated to
overlapping land-cover patches in proportion to weighted source-grid cell area.
Source IDs with no relevant footprint, and Source_ID 0, remain unallocated and
are recorded in QC outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize


BASE_PATH = Path("/Users/robynhaggis/Documents/Geospatial_analysis/dphil_papers")
PAPER_2_PATH = BASE_PATH / "dphil_paper_2"
PAPER_3_PATH = BASE_PATH / "dphil_paper_3"

JAMAICA_CRS = "EPSG:3448"

LAND_USE_FOREST_AFFORESTABLE_PATH = PAPER_2_PATH / "processed_data/land_use_forest_and_afforestable.gpkg"
LAND_USE_LAYER = "land_use"

SOURCE_ATTRIBUTION_ROOT = (
    PAPER_3_PATH
    / "results/02_damage_estimates/landslide_damages/source_id_attribution_combined_class"
)
SOURCE_IDS_RASTER = PAPER_3_PATH / "inputs/landslides/runout_attribution/source_ids_Baseline.tif"

DAMAGE_CASES = ["minimum", "maximum"]

ATTRIBUTION_TARGETS = {
    "protection": {
        "benefit_column": "Protection_Positive_Avoided_EAD_USD",
        "footprint_weight_column": "forest_flood_equivalent_values",
        "footprint_label": "existing_forest",
        "output_prefix": "landslide_protection_existing_forest",
    },
    "restoration": {
        "benefit_column": "Reafforestation_Positive_Avoided_EAD_USD",
        "footprint_weight_column": "afforestable_values",
        "footprint_label": "afforestable",
        "output_prefix": "landslide_restoration_afforestable",
    },
}


def format_usd_readable(value) -> str:
    if pd.isna(value):
        return "NA"
    value = float(value)
    sign = "-" if value < 0 else ""
    abs_value = abs(value)
    if abs_value >= 1_000_000_000:
        return f"{sign}US${abs_value / 1_000_000_000:,.2f} billion"
    if abs_value >= 1_000_000:
        return f"{sign}US${abs_value / 1_000_000:,.2f} million"
    if abs_value >= 1_000:
        return f"{sign}US${abs_value / 1_000:,.1f} thousand"
    return f"{sign}US${abs_value:,.0f}"


def fix_polygon_geometries(gdf: gpd.GeoDataFrame, explode: bool = False) -> gpd.GeoDataFrame:
    """Return valid non-empty polygon geometries."""

    out = gdf.loc[gdf.geometry.notnull()].copy()
    invalid_mask = ~out.geometry.is_valid
    if invalid_mask.any():
        try:
            out.loc[invalid_mask, "geometry"] = out.loc[invalid_mask].geometry.buffer(0)
        except Exception:
            out.loc[invalid_mask, "geometry"] = out.loc[invalid_mask].geometry.make_valid()
    out = out.loc[~out.geometry.is_empty].copy()
    if explode:
        out = out.explode(index_parts=False, ignore_index=True)
    out = out.loc[out.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    return out


def source_case_dir(source_root: Path, damage_case: str) -> Path:
    case_dir = source_root / damage_case
    if not case_dir.exists():
        raise FileNotFoundError(case_dir)
    return case_dir


def load_source_tables(source_root: Path, damage_case: str) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    case_dir = source_case_dir(source_root, damage_case)
    source_csv = case_dir / "landslide_source_avoided_ead_by_source_combined_class.csv"
    source_gpkg = case_dir / "landslide_source_avoided_ead_by_source_combined_class.gpkg"

    for path in [source_csv, source_gpkg]:
        if not path.exists():
            raise FileNotFoundError(path)

    source_all = pd.read_csv(source_csv)
    source_gdf = gpd.read_file(source_gpkg).to_crs(JAMAICA_CRS)

    source_all["Source_ID"] = pd.to_numeric(source_all["Source_ID"], errors="coerce").fillna(0).astype("int64")
    source_gdf["Source_ID"] = pd.to_numeric(source_gdf["Source_ID"], errors="coerce").fillna(0).astype("int64")
    source_gdf = fix_polygon_geometries(source_gdf, explode=False)
    source_gdf["Source_Area_m2"] = source_gdf.geometry.area

    return source_all, source_gdf


def load_footprint(target_config: dict[str, str]) -> gpd.GeoDataFrame:
    if not LAND_USE_FOREST_AFFORESTABLE_PATH.exists():
        raise FileNotFoundError(LAND_USE_FOREST_AFFORESTABLE_PATH)

    weight_col = target_config["footprint_weight_column"]
    keep_columns = ["OBJECTID", "Classify", "LU_CODE", weight_col, "geometry"]
    footprint = gpd.read_file(
        LAND_USE_FOREST_AFFORESTABLE_PATH,
        layer=LAND_USE_LAYER,
        columns=keep_columns,
    )
    if footprint.crs is None:
        raise ValueError(f"Footprint layer has no CRS: {LAND_USE_FOREST_AFFORESTABLE_PATH}")
    if str(footprint.crs).upper() != JAMAICA_CRS:
        footprint = footprint.to_crs(JAMAICA_CRS)

    if weight_col not in footprint.columns:
        raise KeyError(f"Footprint layer missing '{weight_col}'")

    footprint[weight_col] = pd.to_numeric(footprint[weight_col], errors="coerce").fillna(0.0)
    footprint = footprint.loc[footprint[weight_col] > 0, keep_columns].copy()
    footprint = footprint.reset_index(drop=True)
    footprint["Landuse_OBJECTID"] = pd.to_numeric(footprint["OBJECTID"], errors="coerce").astype("Int64")
    footprint["Patch_ID"] = np.arange(1, len(footprint) + 1, dtype="int64")
    footprint = footprint.rename(
        columns={
            "Classify": "Patch_Classify",
            "LU_CODE": "Patch_LU_CODE",
            weight_col: "Footprint_Weight",
        }
    )
    footprint["Footprint_Type"] = target_config["footprint_label"]
    footprint["Footprint_Area_m2"] = footprint.geometry.area
    return footprint


def rasterize_source_patch_weights(
    footprint: gpd.GeoDataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Cross-tabulate Source_ID cells against a rasterized patch footprint."""

    if not SOURCE_IDS_RASTER.exists():
        raise FileNotFoundError(SOURCE_IDS_RASTER)
    if footprint.empty:
        empty = pd.DataFrame(
            columns=[
                "Source_ID",
                "Patch_ID",
                "Source_Grid_Cell_Count",
                "Source_Grid_Area_m2",
                "Weighted_Source_Grid_Area_m2",
            ]
        )
        return empty, {}

    with rasterio.open(SOURCE_IDS_RASTER) as src:
        source_ids = src.read(1)
        profile = src.profile.copy()
        transform = src.transform
        out_shape = (src.height, src.width)
        nodata = src.nodata
        pixel_area_m2 = abs(src.transform.a * src.transform.e)

    if nodata is not None:
        source_ids = np.where(source_ids == nodata, 0, source_ids)

    valid_footprint = footprint.loc[footprint.geometry.notnull() & ~footprint.geometry.is_empty].copy()
    patch_shapes = (
        (geom, int(patch_id))
        for geom, patch_id in zip(valid_footprint.geometry, valid_footprint["Patch_ID"])
        if pd.notna(patch_id)
    )
    weight_shapes = (
        (geom, float(weight))
        for geom, weight in zip(valid_footprint.geometry, valid_footprint["Footprint_Weight"])
    )

    patch_ids = rasterize(
        patch_shapes,
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype="int32",
        all_touched=False,
    )
    footprint_weights = rasterize(
        weight_shapes,
        out_shape=out_shape,
        transform=transform,
        fill=0.0,
        dtype="float32",
        all_touched=False,
    )

    mask = (source_ids > 0) & (patch_ids > 0) & (footprint_weights > 0)
    if not np.any(mask):
        empty = pd.DataFrame(
            columns=[
                "Source_ID",
                "Patch_ID",
                "Source_Grid_Cell_Count",
                "Source_Grid_Area_m2",
                "Weighted_Source_Grid_Area_m2",
            ]
        )
        context = {
            "source_ids": source_ids,
            "patch_ids": patch_ids,
            "footprint_weights": footprint_weights,
            "profile": profile,
            "pixel_area_m2": pixel_area_m2,
        }
        return empty, context

    source_patch_cells = pd.DataFrame(
        {
            "Source_ID": source_ids[mask].astype("int64", copy=False),
            "Patch_ID": patch_ids[mask].astype("int64", copy=False),
            "Footprint_Weight": footprint_weights[mask].astype("float64", copy=False),
        }
    )
    source_patch_cells["Source_Grid_Cell_Count"] = 1
    source_patch_cells["Source_Grid_Area_m2"] = pixel_area_m2
    source_patch_cells["Weighted_Source_Grid_Area_m2"] = (
        source_patch_cells["Footprint_Weight"] * pixel_area_m2
    )

    weights = (
        source_patch_cells
        .groupby(["Source_ID", "Patch_ID"], as_index=False)
        .agg(
            Source_Grid_Cell_Count=("Source_Grid_Cell_Count", "sum"),
            Source_Grid_Area_m2=("Source_Grid_Area_m2", "sum"),
            Weighted_Source_Grid_Area_m2=("Weighted_Source_Grid_Area_m2", "sum"),
        )
    )

    context = {
        "source_ids": source_ids,
        "patch_ids": patch_ids,
        "footprint_weights": footprint_weights,
        "profile": profile,
        "pixel_area_m2": pixel_area_m2,
    }
    return weights, context


def write_cell_attribution_raster(
    allocation: pd.DataFrame,
    raster_context: dict[str, object],
    output_raster: Path,
) -> float:
    if allocation.empty or not raster_context:
        return 0.0

    source_ids = raster_context["source_ids"]
    patch_ids = raster_context["patch_ids"]
    footprint_weights = raster_context["footprint_weights"]
    profile = raster_context["profile"].copy()
    pixel_area_m2 = float(raster_context["pixel_area_m2"])

    mask = (source_ids > 0) & (patch_ids > 0) & (footprint_weights > 0)
    output = np.full(source_ids.shape, -9999.0, dtype="float32")
    if not np.any(mask):
        return 0.0

    key_base = int(max(int(patch_ids.max()), int(allocation["Patch_ID"].max())) + 1)
    allocation_keys = allocation["Source_ID"].astype("int64") * key_base + allocation["Patch_ID"].astype("int64")
    density_lookup = pd.Series(
        allocation["Benefit_Density_USD_Per_Weighted_m2"].to_numpy(dtype="float64"),
        index=allocation_keys.to_numpy(dtype="int64"),
    )
    cell_keys = source_ids[mask].astype("int64", copy=False) * key_base + patch_ids[mask].astype(
        "int64",
        copy=False,
    )
    density = density_lookup.reindex(cell_keys).fillna(0.0).to_numpy(dtype="float64")
    cell_values = density * footprint_weights[mask].astype("float64", copy=False) * pixel_area_m2
    positive_values = cell_values > 0

    masked_output = output[mask]
    masked_output[positive_values] = cell_values[positive_values].astype("float32")
    output[mask] = masked_output

    profile.update(dtype="float32", count=1, nodata=-9999.0, compress="deflate")
    output_raster.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_raster, "w", **profile) as dst:
        dst.write(output, 1)

    return float(output[output > 0].sum(dtype="float64"))


def attach_patch_geometries(
    patch_summary: pd.DataFrame,
    footprint: gpd.GeoDataFrame,
    output_gpkg: Path,
) -> gpd.GeoDataFrame:
    if patch_summary.empty:
        empty = gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=JAMAICA_CRS)
        return empty

    geometry_cols = [
        "Patch_ID",
        "Landuse_OBJECTID",
        "Patch_Classify",
        "Patch_LU_CODE",
        "Footprint_Type",
        "Footprint_Weight",
        "Footprint_Area_m2",
        "geometry",
    ]
    patches = footprint[geometry_cols].merge(
        patch_summary.drop(
            columns=["Landuse_OBJECTID", "Patch_Classify", "Patch_LU_CODE", "Footprint_Type"],
            errors="ignore",
        ),
        how="inner",
        on="Patch_ID",
    )
    patches = gpd.GeoDataFrame(patches, geometry="geometry", crs=JAMAICA_CRS)
    patches = fix_polygon_geometries(patches, explode=False)
    patches = patches.sort_values("Allocated_Benefit_USD", ascending=False).reset_index(drop=True)
    patches.to_file(output_gpkg, driver="GPKG")
    return patches


def allocate_source_to_footprint(
    source_all: pd.DataFrame,
    source_gdf: gpd.GeoDataFrame,
    footprint: gpd.GeoDataFrame,
    target_name: str,
    target_config: dict[str, str],
    output_dir: Path,
) -> dict[str, Path | pd.DataFrame | gpd.GeoDataFrame]:
    benefit_col = target_config["benefit_column"]
    output_prefix = target_config["output_prefix"]

    if benefit_col not in source_all.columns or benefit_col not in source_gdf.columns:
        raise KeyError(f"Source attribution outputs missing '{benefit_col}'")

    source_all[benefit_col] = pd.to_numeric(source_all[benefit_col], errors="coerce").fillna(0.0)
    source_gdf[benefit_col] = pd.to_numeric(source_gdf[benefit_col], errors="coerce").fillna(0.0)
    total_positive_usd = float(source_all[benefit_col].sum())
    source_zero_positive_usd = float(
        source_all.loc[source_all["Source_ID"] == 0, benefit_col].sum()
    )

    sources = source_gdf.loc[
        source_gdf[benefit_col] > 0,
        ["Source_ID", "Source_Area_m2", benefit_col, "geometry"],
    ].copy()
    sources = sources.rename(columns={benefit_col: "Source_Benefit_USD"})
    mappable_positive_usd = float(sources["Source_Benefit_USD"].sum())

    output_dir.mkdir(parents=True, exist_ok=True)
    patch_gpkg = output_dir / f"{output_prefix}_patch_attribution_combined_class.gpkg"
    patch_csv = output_dir / f"{output_prefix}_patch_attribution_combined_class.csv"
    source_patch_csv = output_dir / f"{output_prefix}_source_patch_allocation_combined_class.csv"
    source_qc_csv = output_dir / f"{output_prefix}_source_allocation_qc_combined_class.csv"
    class_summary_csv = output_dir / f"{output_prefix}_landuse_class_summary_combined_class.csv"
    cell_raster = output_dir / f"{output_prefix}_cell_attribution_usd_combined_class.tif"

    if sources.empty:
        empty_patch = gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=JAMAICA_CRS)
        empty_source_qc = pd.DataFrame(
            [
                {
                    "Target": target_name,
                    "Source_ID": pd.NA,
                    "Source_Benefit_USD": 0.0,
                    "Allocated_Benefit_USD": 0.0,
                    "Unallocated_Mappable_Benefit_USD": 0.0,
                    "Footprint_Overlap_Area_m2": 0.0,
                    "Footprint_Weighted_Area_m2": 0.0,
                }
            ]
        )
        empty_patch.to_file(patch_gpkg, driver="GPKG")
        empty_patch.drop(columns="geometry").to_csv(patch_csv, index=False)
        empty_source_qc.to_csv(source_qc_csv, index=False)
        pd.DataFrame().to_csv(class_summary_csv, index=False)
        return {
            "patch_gpkg": patch_gpkg,
            "patch_csv": patch_csv,
            "source_patch_csv": source_patch_csv,
            "source_qc_csv": source_qc_csv,
            "class_summary_csv": class_summary_csv,
            "cell_raster": cell_raster,
            "patches": empty_patch,
            "source_qc": empty_source_qc,
            "class_summary": pd.DataFrame(),
        }

    source_patch_weights, raster_context = rasterize_source_patch_weights(footprint)

    if source_patch_weights.empty:
        source_qc = sources.drop(columns="geometry").copy()
        source_qc["Target"] = target_name
        source_qc["Allocated_Benefit_USD"] = 0.0
        source_qc["Unallocated_Mappable_Benefit_USD"] = source_qc["Source_Benefit_USD"]
        source_qc["Footprint_Source_Grid_Area_m2"] = 0.0
        source_qc["Footprint_Weighted_Source_Grid_Area_m2"] = 0.0
        source_qc["Allocation_Status"] = "no_relevant_footprint_overlap"
        source_qc.to_csv(source_qc_csv, index=False)
        pd.DataFrame().to_csv(class_summary_csv, index=False)
        pd.DataFrame().to_csv(source_patch_csv, index=False)
        empty_patch = gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=JAMAICA_CRS)
        empty_patch.drop(columns="geometry").to_csv(patch_csv, index=False)
        return {
            "patch_gpkg": patch_gpkg,
            "patch_csv": patch_csv,
            "source_patch_csv": source_patch_csv,
            "source_qc_csv": source_qc_csv,
            "class_summary_csv": class_summary_csv,
            "cell_raster": cell_raster,
            "patches": empty_patch,
            "source_qc": source_qc,
            "class_summary": pd.DataFrame(),
        }

    source_values = sources.drop(columns="geometry").copy()
    allocation = source_patch_weights.merge(source_values, how="inner", on="Source_ID")
    allocation = allocation.loc[allocation["Source_Benefit_USD"] > 0].copy()
    patch_metadata = footprint[
        [
            "Patch_ID",
            "Landuse_OBJECTID",
            "Patch_Classify",
            "Patch_LU_CODE",
            "Footprint_Type",
            "Footprint_Weight",
            "Footprint_Area_m2",
        ]
    ].drop_duplicates("Patch_ID")
    allocation = allocation.merge(patch_metadata, how="left", on="Patch_ID")

    source_weight_totals = allocation.groupby("Source_ID")["Weighted_Source_Grid_Area_m2"].transform("sum")
    allocation["Allocation_Share"] = np.where(
        source_weight_totals > 0,
        allocation["Weighted_Source_Grid_Area_m2"] / source_weight_totals,
        0.0,
    )
    allocation["Allocated_Benefit_USD"] = allocation["Source_Benefit_USD"] * allocation["Allocation_Share"]
    allocation["Allocated_Benefit_PV_50Y_10pct_USD"] = (
        allocation["Allocated_Benefit_USD"] * sum(1.0 / (1.0 + 0.10) ** year for year in range(51))
    )
    allocation["Benefit_Density_USD_Per_Weighted_m2"] = np.where(
        allocation["Weighted_Source_Grid_Area_m2"] > 0,
        allocation["Allocated_Benefit_USD"] / allocation["Weighted_Source_Grid_Area_m2"],
        0.0,
    )
    allocation["Target"] = target_name
    allocation.to_csv(source_patch_csv, index=False)

    patch_numeric = (
        allocation
        .groupby(["Patch_ID", "Landuse_OBJECTID", "Patch_Classify", "Patch_LU_CODE", "Footprint_Type"], dropna=False)
        .agg(
            Allocated_Benefit_USD=("Allocated_Benefit_USD", "sum"),
            Allocated_Benefit_PV_50Y_10pct_USD=("Allocated_Benefit_PV_50Y_10pct_USD", "sum"),
            Source_Count=("Source_ID", "nunique"),
            Source_Patch_Count=("Source_ID", "size"),
            Source_Grid_Cell_Count=("Source_Grid_Cell_Count", "sum"),
            Source_Grid_Area_m2=("Source_Grid_Area_m2", "sum"),
            Weighted_Source_Grid_Area_m2=("Weighted_Source_Grid_Area_m2", "sum"),
        )
        .reset_index()
    )
    patches = attach_patch_geometries(patch_numeric, footprint, patch_gpkg)
    patches["Target"] = target_name
    patches["Allocated_Benefit_Readable"] = patches["Allocated_Benefit_USD"].apply(format_usd_readable)
    patches["Allocated_Benefit_PV_50Y_10pct_Readable"] = patches[
        "Allocated_Benefit_PV_50Y_10pct_USD"
    ].apply(format_usd_readable)
    patches = patches.sort_values("Allocated_Benefit_USD", ascending=False).reset_index(drop=True)
    patches.drop(columns="geometry").to_csv(patch_csv, index=False)
    raster_total_usd = write_cell_attribution_raster(allocation, raster_context, cell_raster)

    allocated_by_source = (
        allocation
        .groupby("Source_ID", as_index=False)
        .agg(
            Allocated_Benefit_USD=("Allocated_Benefit_USD", "sum"),
            Footprint_Source_Grid_Area_m2=("Source_Grid_Area_m2", "sum"),
            Footprint_Weighted_Source_Grid_Area_m2=("Weighted_Source_Grid_Area_m2", "sum"),
            Patch_Count=("Patch_ID", "nunique"),
        )
    )
    source_qc = sources.drop(columns="geometry").merge(allocated_by_source, how="left", on="Source_ID")
    fill_zero_cols = [
        "Allocated_Benefit_USD",
        "Footprint_Source_Grid_Area_m2",
        "Footprint_Weighted_Source_Grid_Area_m2",
        "Patch_Count",
    ]
    source_qc[fill_zero_cols] = source_qc[fill_zero_cols].fillna(0.0)
    source_qc["Target"] = target_name
    source_qc["Unallocated_Mappable_Benefit_USD"] = (
        source_qc["Source_Benefit_USD"] - source_qc["Allocated_Benefit_USD"]
    ).clip(lower=0.0)
    source_qc["Allocation_Status"] = np.where(
        source_qc["Allocated_Benefit_USD"] > 0,
        "allocated_to_relevant_footprint",
        "no_relevant_footprint_overlap",
    )
    source_qc["Source_Benefit_Readable"] = source_qc["Source_Benefit_USD"].apply(format_usd_readable)
    source_qc["Allocated_Benefit_Readable"] = source_qc["Allocated_Benefit_USD"].apply(format_usd_readable)
    source_qc["Unallocated_Mappable_Benefit_Readable"] = source_qc[
        "Unallocated_Mappable_Benefit_USD"
    ].apply(format_usd_readable)
    source_qc.sort_values("Source_Benefit_USD", ascending=False).to_csv(source_qc_csv, index=False)

    class_summary = (
        allocation
        .groupby(["Patch_Classify", "Patch_LU_CODE", "Footprint_Type"], dropna=False)
        .agg(
            Allocated_Benefit_USD=("Allocated_Benefit_USD", "sum"),
            Allocated_Benefit_PV_50Y_10pct_USD=("Allocated_Benefit_PV_50Y_10pct_USD", "sum"),
            Patch_Count=("Patch_ID", "nunique"),
            Source_Count=("Source_ID", "nunique"),
            Source_Grid_Cell_Count=("Source_Grid_Cell_Count", "sum"),
            Source_Grid_Area_m2=("Source_Grid_Area_m2", "sum"),
            Weighted_Source_Grid_Area_m2=("Weighted_Source_Grid_Area_m2", "sum"),
        )
        .reset_index()
        .sort_values("Allocated_Benefit_USD", ascending=False)
    )
    class_summary["Allocated_Benefit_Readable"] = class_summary["Allocated_Benefit_USD"].apply(format_usd_readable)
    class_summary.to_csv(class_summary_csv, index=False)

    allocated_to_footprint_usd = float(allocation["Allocated_Benefit_USD"].sum())
    mappable_but_no_footprint_usd = float(source_qc["Unallocated_Mappable_Benefit_USD"].sum())
    run_summary = pd.DataFrame(
        [
            {
                "Target": target_name,
                "Total_Positive_Source_EAD_USD_Including_Source_ID_0": total_positive_usd,
                "Source_ID_0_Positive_EAD_USD_Not_Mappable": source_zero_positive_usd,
                "Mappable_Positive_Source_EAD_USD": mappable_positive_usd,
                "Allocated_To_Footprint_USD": allocated_to_footprint_usd,
                "Cell_Raster_Positive_Total_USD": raster_total_usd,
                "Mappable_But_No_Relevant_Footprint_USD": mappable_but_no_footprint_usd,
                "Allocated_Pct_Of_Total_Positive_Source_EAD": (
                    100.0 * allocated_to_footprint_usd / total_positive_usd
                    if total_positive_usd > 0
                    else np.nan
                ),
                "Allocated_Pct_Of_Mappable_Positive_Source_EAD": (
                    100.0 * allocated_to_footprint_usd / mappable_positive_usd
                    if mappable_positive_usd > 0
                    else np.nan
                ),
                "Positive_Source_Count_Including_Source_ID_0": int((source_all[benefit_col] > 0).sum()),
                "Positive_Mappable_Source_Count": int(len(sources)),
                "Allocated_Source_Count": int((source_qc["Allocated_Benefit_USD"] > 0).sum()),
                "Patch_Count": int(len(patches)),
            }
        ]
    )
    run_summary.to_csv(output_dir / f"{output_prefix}_run_summary_combined_class.csv", index=False)

    return {
        "patch_gpkg": patch_gpkg,
        "patch_csv": patch_csv,
        "source_patch_csv": source_patch_csv,
        "source_qc_csv": source_qc_csv,
        "class_summary_csv": class_summary_csv,
        "cell_raster": cell_raster,
        "patches": patches,
        "source_qc": source_qc,
        "class_summary": class_summary,
        "run_summary": run_summary,
    }


def write_method_metadata(output_root: Path, source_root: Path, damage_cases: Iterable[str]) -> None:
    metadata = {
        "method": (
            "Rasterize the land_use_forest_and_afforestable land-cover layer to the landslide "
            "Source_ID grid and cross-tabulate Source_ID cells by Patch_ID. Protection benefits "
            "are allocated to features with "
            "forest_flood_equivalent_values > 0; restoration benefits are allocated to features with "
            "afforestable_values > 0. Within each source zone, positive avoided EAD is distributed "
            "across relevant footprint patches in proportion to source-grid cell area multiplied "
            "by the footprint weight."
        ),
        "important_caveat": (
            "Source_ID 0 has no source-zone geometry and cannot be mapped to forest/restoration patches. "
            "It is retained in source-ID CSV outputs and reported as not mappable in run-summary QC."
        ),
        "source_attribution_root": str(source_root),
        "land_use_forest_afforestable_path": str(LAND_USE_FOREST_AFFORESTABLE_PATH),
        "land_use_layer": LAND_USE_LAYER,
        "crs": JAMAICA_CRS,
        "damage_cases": list(damage_cases),
        "targets": ATTRIBUTION_TARGETS,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    with open(output_root / "landslide_forest_patch_attribution_method_metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)


def run_forest_patch_attribution(
    source_root: Path = SOURCE_ATTRIBUTION_ROOT,
    damage_cases: Iterable[str] = ("minimum", "maximum"),
) -> dict[str, dict[str, dict[str, Path | pd.DataFrame | gpd.GeoDataFrame]]]:
    damage_cases = list(damage_cases)
    unknown_cases = sorted(set(damage_cases) - set(DAMAGE_CASES))
    if unknown_cases:
        raise ValueError(f"Unknown damage cases: {unknown_cases}")

    write_method_metadata(source_root, source_root, damage_cases)

    footprints = {
        target_name: load_footprint(target_config)
        for target_name, target_config in ATTRIBUTION_TARGETS.items()
    }

    outputs: dict[str, dict[str, dict[str, Path | pd.DataFrame | gpd.GeoDataFrame]]] = {}
    for damage_case in damage_cases:
        source_all, source_gdf = load_source_tables(source_root, damage_case)
        case_output_dir = source_case_dir(source_root, damage_case) / "forest_patch_attribution"
        outputs[damage_case] = {}
        print(f"[{damage_case}] Source rows: {len(source_all):,}; mappable geometries: {len(source_gdf):,}")

        for target_name, target_config in ATTRIBUTION_TARGETS.items():
            print(
                f"[{damage_case}] Allocating {target_name} to "
                f"{target_config['footprint_label']} footprint..."
            )
            target_outputs = allocate_source_to_footprint(
                source_all=source_all,
                source_gdf=source_gdf,
                footprint=footprints[target_name],
                target_name=target_name,
                target_config=target_config,
                output_dir=case_output_dir,
            )
            outputs[damage_case][target_name] = target_outputs
            run_summary = target_outputs.get("run_summary")
            if isinstance(run_summary, pd.DataFrame) and not run_summary.empty:
                row = run_summary.iloc[0]
                print(
                    f"[{damage_case}] {target_name}: allocated "
                    f"{format_usd_readable(row['Allocated_To_Footprint_USD'])} to "
                    f"{int(row['Patch_Count']):,} patches "
                    f"({row['Allocated_Pct_Of_Mappable_Positive_Source_EAD']:.1f}% of mappable positive source EAD)."
                )

    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ATTRIBUTION_ROOT)
    parser.add_argument(
        "--damage-cases",
        nargs="+",
        default=["minimum", "maximum"],
        choices=DAMAGE_CASES,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_forest_patch_attribution(
        source_root=args.source_root,
        damage_cases=args.damage_cases,
    )
