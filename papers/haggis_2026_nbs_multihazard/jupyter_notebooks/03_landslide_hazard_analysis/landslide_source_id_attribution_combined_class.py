#!/usr/bin/env python
"""Attribute landslide combined-class avoided EADs to source-zone IDs.

This step uses the existing combined-class vector/raster intersections and the
new runout-attribution rasters named ``cell_source_*_dominant.tif``.  It samples
the source ID raster at each split asset cell before aggregating direct damages
to source zones and computing EADs.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio


BASE_PATH = Path("/Users/robynhaggis/Documents/Geospatial_analysis/dphil_papers")
PAPER_3_PATH = BASE_PATH / "dphil_paper_3"
DATA_ROOT = BASE_PATH / "dphil_common_cross_cutting/common_incoming_data"
NETWORK_METADATA_FILE = DATA_ROOT / "networks/network_layers_hazard_intersections_details.csv"

LANDSLIDE_RESULTS_ROOT = PAPER_3_PATH / "results/02_damage_estimates/landslide_damages"
INTERSECTIONS_PATH = (
    PAPER_3_PATH
    / "results/01_hazard_infrastructure_network_intersections/landslide_network_intersections_combined_class"
)
HAZARD_LAYERS_FILE = INTERSECTIONS_PATH / "landslide_rasters_for_intersections_combined_class.csv"
RUNOUT_ATTRIBUTION_DIR = PAPER_3_PATH / "inputs/landslides/runout_attribution"
SOURCE_ZONES_FILE = RUNOUT_ATTRIBUTION_DIR / "source_zones_Baseline.gpkg"
DEFAULT_OUTPUT_DIR = LANDSLIDE_RESULTS_ROOT / "source_id_attribution_combined_class"

JAMAICA_CRS_CODE = 3448
JMD_PER_USD = 150.0
USD_PER_JMD = 1.0 / JMD_PER_USD

SCENARIOS = ["baseline", "deforestation", "reafforestation"]
SCENARIO_FILE_LABELS = {
    "baseline": "Baseline",
    "deforestation": "Deforestation",
    "reafforestation": "Reafforestation",
}
RETURN_PERIODS = [5, 10, 25, 50, 100]

DAMAGE_CASE_PARAMETERS = {
    "minimum": {
        "cost_uncertainty_parameter": 0.0,
        "damage_uncertainty_parameter": 0.0,
    },
    "maximum": {
        "cost_uncertainty_parameter": 1.0,
        "damage_uncertainty_parameter": 1.0,
    },
}

LANDSLIDE_CLASS_DAMAGE_RATIO_LOWER = {
    0: 0.0,
    1: 0.05,
    2: 0.30,
    3: 0.50,
    4: 0.80,
}
LANDSLIDE_CLASS_DAMAGE_RATIO_UPPER = {
    0: 0.0,
    1: 0.30,
    2: 0.60,
    3: 0.80,
    4: 1.00,
}

BENEFIT_COLUMNS = [
    "Protection_Net_Avoided_EAD_USD",
    "Protection_Positive_Avoided_EAD_USD",
    "Protection_Increased_Damage_USD",
    "Reafforestation_Net_Avoided_EAD_USD",
    "Reafforestation_Positive_Avoided_EAD_USD",
    "Reafforestation_Increased_Damage_USD",
    "Combined_Benefit_Reafforestation_vs_Deforestation_USD",
]


def source_raster_path(scenario: str, return_period: int) -> Path:
    label = SCENARIO_FILE_LABELS[scenario]
    tif_path = RUNOUT_ATTRIBUTION_DIR / f"cell_source_{label}_rp{return_period}_dominant.tif"
    if tif_path.exists():
        return tif_path
    tiff_path = tif_path.with_suffix(".tiff")
    if tiff_path.exists():
        return tiff_path
    raise FileNotFoundError(
        f"Missing source attribution raster for {scenario} RP{return_period}: "
        f"{tif_path} or {tiff_path}"
    )


def interpolate_landslide_class_damage_ratios(uncertainty_parameter: float) -> dict[int, float]:
    return {
        class_id: lower_ratio
        + uncertainty_parameter * (LANDSLIDE_CLASS_DAMAGE_RATIO_UPPER[class_id] - lower_ratio)
        for class_id, lower_ratio in LANDSLIDE_CLASS_DAMAGE_RATIO_LOWER.items()
    }


def compute_discount_factor(years: int = 50, annual_discount_rate: float = 0.10) -> float:
    return sum(1.0 / (1.0 + annual_discount_rate) ** year for year in range(years + 1))


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


def resolve_network_asset_file(asset_relative_path: str) -> Path:
    relative_asset_path = Path(asset_relative_path)
    asset_file_in_common_incoming_data = DATA_ROOT / relative_asset_path
    asset_file_in_nested_networks_folder = DATA_ROOT / "networks" / relative_asset_path

    if asset_file_in_common_incoming_data.exists():
        return asset_file_in_common_incoming_data
    if asset_file_in_nested_networks_folder.exists():
        return asset_file_in_nested_networks_folder

    raise FileNotFoundError(
        f"Could not find asset file '{relative_asset_path}'. Checked: "
        f"{asset_file_in_common_incoming_data} ; {asset_file_in_nested_networks_folder}"
    )


def convert_usd_costs_to_jmd(row, cost_value_col: str, cost_unit_col: str) -> float:
    unit_value = str(row[cost_unit_col])
    if ("US$" in unit_value) or ("USD" in unit_value):
        return JMD_PER_USD * row[cost_value_col]
    return row[cost_value_col]


def modify_cost_units(row, cost_unit_col: str, damage_cost_col: str = "damage_cost") -> float:
    unit_value = str(row[cost_unit_col])
    if "/km" in unit_value:
        return 0.001 * row[damage_cost_col]
    return row[damage_cost_col]


def cleaned_damage_cost_unit(unit_series: pd.Series, layer_type: str) -> pd.Series:
    units = unit_series.fillna("J$").astype(str)
    if layer_type == "nodes":
        return units
    return units.str.split("/").str[0]


def load_asset_cost_lookup(asset_info, cost_uncertainty_parameter: float) -> tuple[pd.DataFrame, str]:
    asset_gpkg_file = resolve_network_asset_file(asset_info.path)
    asset_df = gpd.read_file(asset_gpkg_file, layer=asset_info.asset_layer)
    asset_id_col = asset_info.asset_id_column

    if asset_id_col not in asset_df.columns:
        raise KeyError(
            f"Asset ID column '{asset_id_col}' not found in {asset_gpkg_file} ({asset_info.asset_layer})"
        )

    min_cost_col = asset_info.asset_min_cost_column if isinstance(asset_info.asset_min_cost_column, str) else None
    max_cost_col = asset_info.asset_max_cost_column if isinstance(asset_info.asset_max_cost_column, str) else None
    mean_cost_col = asset_info.asset_mean_cost_column if isinstance(asset_info.asset_mean_cost_column, str) else None
    cost_unit_col = asset_info.asset_cost_unit_column if isinstance(asset_info.asset_cost_unit_column, str) else None

    if not cost_unit_col or cost_unit_col not in asset_df.columns:
        asset_df["_tmp_cost_unit"] = "J$"
        cost_unit_col = "_tmp_cost_unit"

    for maybe_cost_col in [min_cost_col, max_cost_col, mean_cost_col]:
        if maybe_cost_col and maybe_cost_col in asset_df.columns:
            asset_df[maybe_cost_col] = pd.to_numeric(asset_df[maybe_cost_col], errors="coerce").fillna(0.0)

    if min_cost_col and min_cost_col in asset_df.columns:
        asset_df[min_cost_col] = asset_df.apply(
            lambda row: convert_usd_costs_to_jmd(row, min_cost_col, cost_unit_col),
            axis=1,
        )
    if max_cost_col and max_cost_col in asset_df.columns:
        asset_df[max_cost_col] = asset_df.apply(
            lambda row: convert_usd_costs_to_jmd(row, max_cost_col, cost_unit_col),
            axis=1,
        )

    asset_df[cost_unit_col] = asset_df[cost_unit_col].replace(["USD", "US$"], "J$", regex=True)

    if asset_info.sector == "energy" and asset_info.asset_layer == "edges" and "length" in asset_df.columns:
        if min_cost_col and min_cost_col in asset_df.columns:
            asset_df[min_cost_col] = np.where(asset_df["length"] > 0, asset_df[min_cost_col] / asset_df["length"], 0.0)
        if max_cost_col and max_cost_col in asset_df.columns:
            asset_df[max_cost_col] = np.where(asset_df["length"] > 0, asset_df[max_cost_col] / asset_df["length"], 0.0)
        asset_df[cost_unit_col] = "J$/m"

    if min_cost_col and min_cost_col in asset_df.columns and max_cost_col and max_cost_col in asset_df.columns:
        asset_df["damage_cost"] = asset_df[min_cost_col] + cost_uncertainty_parameter * (
            asset_df[max_cost_col] - asset_df[min_cost_col]
        )
    elif mean_cost_col and mean_cost_col in asset_df.columns:
        asset_df["damage_cost"] = asset_df[mean_cost_col]
    elif min_cost_col and min_cost_col in asset_df.columns:
        asset_df["damage_cost"] = asset_df[min_cost_col]
    elif max_cost_col and max_cost_col in asset_df.columns:
        asset_df["damage_cost"] = asset_df[max_cost_col]
    else:
        asset_df["damage_cost"] = 0.0

    asset_df["damage_cost"] = asset_df.apply(lambda row: modify_cost_units(row, cost_unit_col), axis=1)
    asset_lookup = asset_df[[asset_id_col, "damage_cost", cost_unit_col]].copy()
    return asset_lookup, cost_unit_col


def add_exposure_dimensions(gdf: gpd.GeoDataFrame, layer_type: str) -> gpd.GeoDataFrame:
    out = gdf.copy()
    if layer_type == "edges":
        out["Exposure"] = out.geometry.length
        out["Exposure_Unit"] = "m"
    elif layer_type == "areas":
        out["Exposure"] = out.geometry.area
        out["Exposure_Unit"] = "m2"
    else:
        out["Exposure"] = 1.0
        out["Exposure_Unit"] = "unit"
    return out


def hazard_column(scenario: str, return_period: int) -> str:
    return f"landslide_combined_class_{scenario}_rp_{return_period}"


def expected_hazard_columns() -> list[str]:
    return [hazard_column(scenario, rp) for scenario in SCENARIOS for rp in RETURN_PERIODS]


def read_geoparquet_columns(path: Path, columns: list[str]) -> gpd.GeoDataFrame:
    try:
        return gpd.read_parquet(path, columns=columns)
    except TypeError:
        return gpd.read_parquet(path)[columns]


def sample_source_ids(source_raster: Path, x_indices: np.ndarray, y_indices: np.ndarray) -> np.ndarray:
    with rasterio.open(source_raster) as src:
        band = src.read(1)
        if x_indices.min(initial=0) < 0 or y_indices.min(initial=0) < 0:
            raise ValueError(f"Negative cell indices found for {source_raster}")
        if x_indices.max(initial=0) >= src.width or y_indices.max(initial=0) >= src.height:
            raise ValueError(f"Cell indices exceed raster bounds for {source_raster}")
        return band[y_indices, x_indices].astype("int64", copy=False)


def check_source_raster_inventory(output_dir: Path) -> pd.DataFrame:
    if not HAZARD_LAYERS_FILE.exists():
        raise FileNotFoundError(HAZARD_LAYERS_FILE)

    hazard_layers = pd.read_csv(HAZARD_LAYERS_FILE)
    reference_hazard = Path(hazard_layers.loc[0, "fname"])
    if not reference_hazard.exists():
        raise FileNotFoundError(reference_hazard)

    rows = []
    with rasterio.open(reference_hazard) as ref:
        reference_crs = ref.crs
        reference_shape = (ref.height, ref.width)
        reference_transform = tuple(ref.transform)

    for scenario in SCENARIOS:
        for return_period in RETURN_PERIODS:
            path = source_raster_path(scenario, return_period)
            with rasterio.open(path) as src:
                transform_delta = np.max(np.abs(np.array(tuple(src.transform)) - np.array(reference_transform)))
                grid_matches = (
                    src.crs == reference_crs
                    and (src.height, src.width) == reference_shape
                    and transform_delta <= 1e-8
                )
                rows.append(
                    {
                        "Scenario": scenario,
                        "ReturnPeriod": return_period,
                        "Source_Raster": str(path),
                        "CRS": str(src.crs),
                        "Height": src.height,
                        "Width": src.width,
                        "NoData": src.nodata,
                        "Data_Type": src.dtypes[0],
                        "Transform_Max_Abs_Delta_vs_Hazard": transform_delta,
                        "Grid_Matches_Hazard": grid_matches,
                    }
                )

    inventory = pd.DataFrame(rows)
    inventory_file = output_dir / "source_raster_inventory_combined_class.csv"
    inventory.to_csv(inventory_file, index=False)

    bad = inventory.loc[~inventory["Grid_Matches_Hazard"]]
    if not bad.empty:
        raise ValueError(
            "One or more cell_source rasters do not align with the combined-class hazard grid. "
            f"See {inventory_file}"
        )
    return inventory


def process_asset_damage_by_source(asset_info, damage_case: str, output_dir: Path) -> pd.DataFrame:
    params = DAMAGE_CASE_PARAMETERS[damage_case]
    damage_ratios = interpolate_landslide_class_damage_ratios(params["damage_uncertainty_parameter"])
    asset_key = f"{asset_info.asset_gpkg}_{asset_info.asset_layer}"
    hazard_layers_name = HAZARD_LAYERS_FILE.stem
    intersection_file = (
        INTERSECTIONS_PATH
        / f"{asset_info.asset_gpkg}_splits__{hazard_layers_name}__{asset_info.asset_layer}.geoparquet"
    )

    if not intersection_file.exists():
        print(f"[{damage_case}] Missing intersections for {asset_key}: {intersection_file}")
        return pd.DataFrame()

    required_columns = [
        asset_info.asset_id_column,
        "geometry",
        "cell_index_0_x",
        "cell_index_0_y",
        *expected_hazard_columns(),
    ]
    intersection_gdf = read_geoparquet_columns(intersection_file, required_columns)
    if intersection_gdf.empty:
        return pd.DataFrame()

    if intersection_gdf.crs is None:
        raise ValueError(f"Intersection file has no CRS: {intersection_file}")
    if str(intersection_gdf.crs).upper() != f"EPSG:{JAMAICA_CRS_CODE}":
        intersection_gdf = intersection_gdf.to_crs(epsg=JAMAICA_CRS_CODE)

    intersection_gdf = add_exposure_dimensions(intersection_gdf, asset_info.asset_layer)
    asset_lookup, cost_unit_col = load_asset_cost_lookup(asset_info, params["cost_uncertainty_parameter"])

    work = intersection_gdf[
        [
            asset_info.asset_id_column,
            "geometry",
            "Exposure",
            "Exposure_Unit",
            "cell_index_0_x",
            "cell_index_0_y",
            *expected_hazard_columns(),
        ]
    ].copy()
    work = pd.merge(work, asset_lookup, how="left", on=asset_info.asset_id_column)
    work["damage_cost"] = pd.to_numeric(work["damage_cost"], errors="coerce").fillna(0.0)
    work["Damage_Cost_Unit"] = cleaned_damage_cost_unit(work[cost_unit_col], asset_info.asset_layer)
    work["Base_Damage_JD"] = work["damage_cost"] * work["Exposure"]

    x_indices = pd.to_numeric(work["cell_index_0_x"], errors="raise").to_numpy(dtype=np.int64)
    y_indices = pd.to_numeric(work["cell_index_0_y"], errors="raise").to_numpy(dtype=np.int64)
    base_damage = work["Base_Damage_JD"].to_numpy(dtype="float64", copy=False)
    exposure = work["Exposure"].to_numpy(dtype="float64", copy=False)

    grouped_rows = []
    for scenario in SCENARIOS:
        for return_period in RETURN_PERIODS:
            source_ids = sample_source_ids(source_raster_path(scenario, return_period), x_indices, y_indices)
            hazard_values = (
                pd.to_numeric(work[hazard_column(scenario, return_period)], errors="coerce")
                .fillna(0)
                .round()
                .astype("int16")
                .to_numpy()
            )
            damage_ratio = np.array([damage_ratios.get(int(value), 0.0) for value in hazard_values], dtype="float64")
            direct_damage_jd = base_damage * damage_ratio

            grouped = (
                pd.DataFrame(
                    {
                        "Source_ID": source_ids,
                        "Direct_Damages_JD": direct_damage_jd,
                        "Exposure": exposure,
                        "Hazard_Class": hazard_values,
                    }
                )
                .groupby("Source_ID", as_index=False)
                .agg(
                    Direct_Damages_JD=("Direct_Damages_JD", "sum"),
                    Exposure=("Exposure", "sum"),
                    Split_Row_Count=("Source_ID", "size"),
                    Hazard_Class_Max=("Hazard_Class", "max"),
                )
            )
            grouped["Damage_Case"] = damage_case
            grouped["Sector"] = asset_info.sector
            grouped["Subsector"] = asset_info.asset_description
            grouped["Asset"] = asset_info.asset_gpkg
            grouped["Layer"] = asset_info.asset_layer
            grouped["Scenario"] = scenario
            grouped["ReturnPeriod"] = return_period
            grouped["Direct_Damages_USD"] = grouped["Direct_Damages_JD"] * USD_PER_JMD
            grouped["Is_Attributed_Source"] = grouped["Source_ID"] > 0
            grouped["Attribution_Status"] = np.where(
                grouped["Is_Attributed_Source"],
                "cell_source_dominant",
                "unattributed_source_id_0",
            )
            grouped_rows.append(grouped)

    out = pd.concat(grouped_rows, ignore_index=True)
    ordered_columns = [
        "Damage_Case",
        "Sector",
        "Subsector",
        "Asset",
        "Layer",
        "Scenario",
        "ReturnPeriod",
        "Source_ID",
        "Is_Attributed_Source",
        "Attribution_Status",
        "Direct_Damages_JD",
        "Direct_Damages_USD",
        "Exposure",
        "Split_Row_Count",
        "Hazard_Class_Max",
    ]
    out = out[ordered_columns]

    per_asset_dir = output_dir / damage_case / "direct_damage_by_source_assets"
    per_asset_dir.mkdir(parents=True, exist_ok=True)
    per_asset_file = per_asset_dir / f"{asset_key}_source_return_period_direct_damages_combined_class.parquet"
    out.to_parquet(per_asset_file, index=False)
    print(f"[{damage_case}] {asset_key}: {len(out):,} grouped source/RP rows")
    return out


def calculate_ead_from_direct_damages(
    direct_damage: pd.DataFrame,
    group_columns: list[str],
    value_column: str = "Direct_Damages_JD",
) -> pd.DataFrame:
    if direct_damage.empty:
        return pd.DataFrame(columns=[*group_columns, "Scenario", "EAD_JD", "EAD_USD"])

    grouped = (
        direct_damage
        .groupby([*group_columns, "Scenario", "ReturnPeriod"], dropna=False, as_index=False)[value_column]
        .sum()
    )
    wide = grouped.pivot_table(
        index=[*group_columns, "Scenario"],
        columns="ReturnPeriod",
        values=value_column,
        aggfunc="sum",
        fill_value=0.0,
    )

    for return_period in RETURN_PERIODS:
        if return_period not in wide.columns:
            wide[return_period] = 0.0
    wide = wide[RETURN_PERIODS]

    rp_values = np.array(sorted(RETURN_PERIODS, reverse=True), dtype="float64")
    probabilities = 1.0 / rp_values
    damage_matrix = wide[list(rp_values.astype(int))].to_numpy(dtype="float64", copy=False)
    ead_jd = np.trapz(damage_matrix, x=probabilities, axis=1)

    out = wide.reset_index()[[*group_columns, "Scenario"]].copy()
    out["EAD_JD"] = ead_jd
    out["EAD_USD"] = out["EAD_JD"] * USD_PER_JMD
    return out


def build_avoided_metrics(ead: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    if ead.empty:
        return pd.DataFrame()

    scenario_ead = ead.pivot_table(
        index=group_columns,
        columns="Scenario",
        values="EAD_USD",
        aggfunc="sum",
        fill_value=0.0,
    )
    for scenario in SCENARIOS:
        if scenario not in scenario_ead.columns:
            scenario_ead[scenario] = 0.0

    out = scenario_ead.reset_index().rename(
        columns={
            "baseline": "EAD_Baseline_USD",
            "deforestation": "EAD_Deforestation_USD",
            "reafforestation": "EAD_Reafforestation_USD",
        }
    )
    out["Protection_Net_Avoided_EAD_USD"] = out["EAD_Deforestation_USD"] - out["EAD_Baseline_USD"]
    out["Reafforestation_Net_Avoided_EAD_USD"] = out["EAD_Baseline_USD"] - out["EAD_Reafforestation_USD"]
    out["Protection_Positive_Avoided_EAD_USD"] = out["Protection_Net_Avoided_EAD_USD"].clip(lower=0.0)
    out["Reafforestation_Positive_Avoided_EAD_USD"] = out["Reafforestation_Net_Avoided_EAD_USD"].clip(lower=0.0)
    out["Protection_Increased_Damage_USD"] = (-out["Protection_Net_Avoided_EAD_USD"]).clip(lower=0.0)
    out["Reafforestation_Increased_Damage_USD"] = (-out["Reafforestation_Net_Avoided_EAD_USD"]).clip(lower=0.0)
    out["Combined_Benefit_Reafforestation_vs_Deforestation_USD"] = (
        out["EAD_Deforestation_USD"] - out["EAD_Reafforestation_USD"]
    )
    return out


def add_discounted_columns(table: pd.DataFrame, discount_factor: float) -> pd.DataFrame:
    out = table.copy()
    for column in BENEFIT_COLUMNS:
        if column in out.columns:
            out[f"{column}_PV_50Y_10pct"] = out[column] * discount_factor
            out[f"{column}_Readable"] = out[column].apply(format_usd_readable)
            out[f"{column}_PV_50Y_10pct_Readable"] = out[f"{column}_PV_50Y_10pct"].apply(format_usd_readable)
    return out


def write_source_zone_geopackage(
    source_table: pd.DataFrame,
    output_file: Path,
    source_id_column: str = "Source_ID",
) -> None:
    if source_table.empty:
        return
    source_zones = gpd.read_file(SOURCE_ZONES_FILE).rename(columns={"source_id": source_id_column})
    source_zones = source_zones[[source_id_column, "area_m2", "geometry"]]
    source_rows = source_table.loc[pd.to_numeric(source_table[source_id_column], errors="coerce").fillna(0) > 0].copy()
    source_gdf = source_zones.merge(source_rows, how="inner", on=source_id_column)
    source_gdf.to_file(output_file, driver="GPKG")


def read_existing_ead_totals(damage_case: str) -> pd.DataFrame:
    existing_file = (
        LANDSLIDE_RESULTS_ROOT
        / f"results_landslide_{damage_case}_scenario_combined_class/damage_estimates/landslide_ead_overall_totals_usd_combined_class.csv"
    )
    if not existing_file.exists():
        return pd.DataFrame()
    existing = pd.read_csv(existing_file)
    existing["Scenario"] = existing["Scenario"].astype(str).str.lower()
    return existing[["Scenario", "Total_EAD_USD"]].rename(columns={"Total_EAD_USD": "Existing_Total_EAD_USD"})


def save_case_outputs(damage_case: str, direct_damage: pd.DataFrame, output_dir: Path) -> dict[str, pd.DataFrame]:
    case_dir = output_dir / damage_case
    case_dir.mkdir(parents=True, exist_ok=True)
    discount_factor = compute_discount_factor()

    direct_damage_file = case_dir / "landslide_source_return_period_direct_damages_combined_class.parquet"
    direct_damage.to_parquet(direct_damage_file, index=False)

    direct_qc = (
        direct_damage
        .groupby(["Damage_Case", "Scenario", "ReturnPeriod", "Is_Attributed_Source"], as_index=False)
        .agg(
            Direct_Damages_JD=("Direct_Damages_JD", "sum"),
            Direct_Damages_USD=("Direct_Damages_USD", "sum"),
            Source_Row_Count=("Source_ID", "size"),
        )
    )
    direct_qc_file = case_dir / "landslide_source_return_period_direct_damage_qc_combined_class.csv"
    direct_qc.to_csv(direct_qc_file, index=False)

    direct_status_qc = (
        direct_damage
        .groupby(["Damage_Case", "Scenario", "ReturnPeriod", "Attribution_Status"], as_index=False)
        .agg(
            Direct_Damages_JD=("Direct_Damages_JD", "sum"),
            Direct_Damages_USD=("Direct_Damages_USD", "sum"),
            Source_Row_Count=("Source_ID", "size"),
        )
    )
    direct_status_qc.to_csv(
        case_dir / "landslide_source_return_period_direct_damage_status_qc_combined_class.csv",
        index=False,
    )

    source_ead = calculate_ead_from_direct_damages(
        direct_damage,
        ["Damage_Case", "Source_ID", "Is_Attributed_Source", "Attribution_Status"],
    )
    source_sector_ead = calculate_ead_from_direct_damages(
        direct_damage,
        ["Damage_Case", "Source_ID", "Is_Attributed_Source", "Attribution_Status", "Sector"],
    )
    source_subsector_ead = calculate_ead_from_direct_damages(
        direct_damage,
        ["Damage_Case", "Source_ID", "Is_Attributed_Source", "Attribution_Status", "Sector", "Subsector"],
    )

    source_avoided = add_discounted_columns(
        build_avoided_metrics(
            source_ead,
            ["Damage_Case", "Source_ID", "Is_Attributed_Source", "Attribution_Status"],
        ),
        discount_factor,
    )
    source_sector_avoided = add_discounted_columns(
        build_avoided_metrics(
            source_sector_ead,
            ["Damage_Case", "Source_ID", "Is_Attributed_Source", "Attribution_Status", "Sector"],
        ),
        discount_factor,
    )
    source_subsector_avoided = add_discounted_columns(
        build_avoided_metrics(
            source_subsector_ead,
            ["Damage_Case", "Source_ID", "Is_Attributed_Source", "Attribution_Status", "Sector", "Subsector"],
        ),
        discount_factor,
    )

    source_ead.to_csv(case_dir / "landslide_source_ead_by_source_combined_class.csv", index=False)
    source_sector_ead.to_csv(case_dir / "landslide_source_ead_by_source_sector_combined_class.csv", index=False)
    source_subsector_ead.to_csv(case_dir / "landslide_source_ead_by_source_subsector_combined_class.csv", index=False)
    source_avoided.to_csv(case_dir / "landslide_source_avoided_ead_by_source_combined_class.csv", index=False)
    source_sector_avoided.to_csv(case_dir / "landslide_source_avoided_ead_by_source_sector_combined_class.csv", index=False)
    source_subsector_avoided.to_csv(case_dir / "landslide_source_avoided_ead_by_source_subsector_combined_class.csv", index=False)

    write_source_zone_geopackage(
        source_avoided,
        case_dir / "landslide_source_avoided_ead_by_source_combined_class.gpkg",
    )

    ead_qc = source_ead.groupby(["Damage_Case", "Scenario"], as_index=False)["EAD_USD"].sum()
    existing_ead = read_existing_ead_totals(damage_case)
    if not existing_ead.empty:
        ead_qc = ead_qc.merge(existing_ead, how="left", on="Scenario")
        ead_qc["Difference_vs_Existing_USD"] = ead_qc["EAD_USD"] - ead_qc["Existing_Total_EAD_USD"]
        ead_qc["Abs_Difference_vs_Existing_USD"] = ead_qc["Difference_vs_Existing_USD"].abs()
    ead_qc.to_csv(case_dir / "landslide_source_ead_total_qc_combined_class.csv", index=False)

    print(f"[{damage_case}] Saved source direct-damage parquet: {direct_damage_file}")
    print(f"[{damage_case}] Source EAD rows: {len(source_ead):,}; source avoided rows: {len(source_avoided):,}")
    return {
        "source_ead": source_ead,
        "source_sector_ead": source_sector_ead,
        "source_subsector_ead": source_subsector_ead,
        "source_avoided": source_avoided,
        "source_sector_avoided": source_sector_avoided,
        "source_subsector_avoided": source_subsector_avoided,
    }


def min_max_range(table: pd.DataFrame, group_columns: list[str], value_columns: list[str]) -> pd.DataFrame:
    if table.empty:
        return pd.DataFrame()

    case_values = {}
    for damage_case in DAMAGE_CASE_PARAMETERS:
        case_values[damage_case] = (
            table.loc[table["Damage_Case"] == damage_case, [*group_columns, *value_columns]]
            .copy()
            .set_index(group_columns)
        )

    all_index = case_values["minimum"].index.union(case_values["maximum"].index)
    pieces = []
    for damage_case, prefix in [("minimum", "Minimum"), ("maximum", "Maximum")]:
        case_table = case_values[damage_case].reindex(all_index).fillna(0.0)
        pieces.append(case_table.rename(columns={col: f"{prefix}_{col}" for col in value_columns}))

    combined = pd.concat(pieces, axis=1).reset_index()
    for value_column in value_columns:
        min_col = f"{value_column}_Range_Min"
        max_col = f"{value_column}_Range_Max"
        combined[min_col] = combined[[f"Minimum_{value_column}", f"Maximum_{value_column}"]].min(axis=1)
        combined[max_col] = combined[[f"Minimum_{value_column}", f"Maximum_{value_column}"]].max(axis=1)
        combined[f"{min_col}_Readable"] = combined[min_col].apply(format_usd_readable)
        combined[f"{max_col}_Readable"] = combined[max_col].apply(format_usd_readable)
    return combined


def save_min_max_outputs(case_outputs: dict[str, dict[str, pd.DataFrame]], output_dir: Path) -> None:
    if set(case_outputs) != set(DAMAGE_CASE_PARAMETERS):
        return

    combined_dir = output_dir / "min_max"
    combined_dir.mkdir(parents=True, exist_ok=True)

    source_avoided_all = pd.concat(
        [case_outputs[damage_case]["source_avoided"] for damage_case in DAMAGE_CASE_PARAMETERS],
        ignore_index=True,
    )
    source_sector_avoided_all = pd.concat(
        [case_outputs[damage_case]["source_sector_avoided"] for damage_case in DAMAGE_CASE_PARAMETERS],
        ignore_index=True,
    )
    source_subsector_avoided_all = pd.concat(
        [case_outputs[damage_case]["source_subsector_avoided"] for damage_case in DAMAGE_CASE_PARAMETERS],
        ignore_index=True,
    )

    value_columns = [
        "Protection_Net_Avoided_EAD_USD",
        "Protection_Positive_Avoided_EAD_USD",
        "Protection_Increased_Damage_USD",
        "Reafforestation_Net_Avoided_EAD_USD",
        "Reafforestation_Positive_Avoided_EAD_USD",
        "Reafforestation_Increased_Damage_USD",
        "Combined_Benefit_Reafforestation_vs_Deforestation_USD",
        "Protection_Positive_Avoided_EAD_USD_PV_50Y_10pct",
        "Protection_Increased_Damage_USD_PV_50Y_10pct",
        "Reafforestation_Positive_Avoided_EAD_USD_PV_50Y_10pct",
        "Reafforestation_Increased_Damage_USD_PV_50Y_10pct",
    ]

    source_range = min_max_range(
        source_avoided_all,
        ["Source_ID", "Is_Attributed_Source", "Attribution_Status"],
        value_columns,
    )
    source_sector_range = min_max_range(
        source_sector_avoided_all,
        ["Source_ID", "Is_Attributed_Source", "Attribution_Status", "Sector"],
        value_columns,
    )
    source_subsector_range = min_max_range(
        source_subsector_avoided_all,
        ["Source_ID", "Is_Attributed_Source", "Attribution_Status", "Sector", "Subsector"],
        value_columns,
    )

    source_range.to_csv(combined_dir / "landslide_source_avoided_ead_by_source_min_max_combined_class.csv", index=False)
    source_sector_range.to_csv(
        combined_dir / "landslide_source_avoided_ead_by_source_sector_min_max_combined_class.csv",
        index=False,
    )
    source_subsector_range.to_csv(
        combined_dir / "landslide_source_avoided_ead_by_source_subsector_min_max_combined_class.csv",
        index=False,
    )
    write_source_zone_geopackage(
        source_range,
        combined_dir / "landslide_source_avoided_ead_by_source_min_max_combined_class.gpkg",
    )

    drafting = (
        source_range.loc[source_range["Is_Attributed_Source"]]
        .sort_values("Protection_Positive_Avoided_EAD_USD_Range_Max", ascending=False)
        .head(50)
    )
    drafting.to_csv(
        combined_dir / "landslide_top50_source_avoided_ead_drafting_values_min_max_combined_class.csv",
        index=False,
    )
    print(f"[min_max] Saved combined source range outputs in {combined_dir}")


def write_method_metadata(output_dir: Path, damage_cases: Iterable[str], asset_keys: Iterable[str] | None) -> None:
    metadata = {
        "method": (
            "Sample cell_source_{Scenario}_rp{ReturnPeriod}_dominant.tif at each existing "
            "combined-class split-intersection cell, aggregate direct damages by Source_ID and "
            "return period, then compute EAD using the same trapezoid-over-exceedance-probability "
            "method as the landslide EAD notebooks."
        ),
        "source_id_zero": (
            "Source_ID 0 is nodata/unattributed in the cell_source rasters. It can carry non-zero "
            "combined-class damages because the combined-class rasters also include cells outside "
            "the dominant cell-source attribution footprint."
        ),
        "protection_metric": "Protection_Net_Avoided_EAD_USD = EAD_Deforestation_USD - EAD_Baseline_USD by Source_ID.",
        "reafforestation_metric": "Reafforestation_Net_Avoided_EAD_USD = EAD_Baseline_USD - EAD_Reafforestation_USD by Source_ID.",
        "positive_metric": "Positive_Avoided clips net avoided EAD at zero; Increased_Damage is the magnitude of negative avoided EAD.",
        "return_periods": RETURN_PERIODS,
        "damage_cases": list(damage_cases),
        "asset_keys": list(asset_keys) if asset_keys else "all",
        "base_path": str(BASE_PATH),
        "intersections_path": str(INTERSECTIONS_PATH),
        "runout_attribution_dir": str(RUNOUT_ATTRIBUTION_DIR),
        "source_zones_file": str(SOURCE_ZONES_FILE),
        "jmd_per_usd": JMD_PER_USD,
        "discount_years": 50,
        "discount_rate": 0.10,
        "discount_factor": compute_discount_factor(),
    }
    with open(output_dir / "landslide_source_id_attribution_method_metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)


def run_source_id_attribution(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    damage_cases: Iterable[str] = ("minimum", "maximum"),
    asset_keys: Iterable[str] | None = None,
) -> dict[str, dict[str, pd.DataFrame]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    damage_cases = list(damage_cases)
    asset_keys = set(asset_keys) if asset_keys else None

    unknown_cases = sorted(set(damage_cases) - set(DAMAGE_CASE_PARAMETERS))
    if unknown_cases:
        raise ValueError(f"Unknown damage cases: {unknown_cases}")

    for required_path in [NETWORK_METADATA_FILE, HAZARD_LAYERS_FILE, SOURCE_ZONES_FILE]:
        if not required_path.exists():
            raise FileNotFoundError(required_path)

    inventory = check_source_raster_inventory(output_dir)
    print(f"Validated {len(inventory)} source rasters against the combined-class grid.")

    asset_data_details = pd.read_csv(NETWORK_METADATA_FILE)
    if asset_keys:
        asset_data_details["asset_key"] = asset_data_details["asset_gpkg"] + "_" + asset_data_details["asset_layer"]
        asset_data_details = asset_data_details.loc[asset_data_details["asset_key"].isin(asset_keys)].copy()
        missing_asset_keys = asset_keys - set(asset_data_details["asset_key"])
        if missing_asset_keys:
            raise ValueError(f"Asset keys not found in network metadata: {sorted(missing_asset_keys)}")
    if asset_data_details.empty:
        raise ValueError("No assets selected for source attribution.")

    write_method_metadata(output_dir, damage_cases, asset_keys)

    case_outputs: dict[str, dict[str, pd.DataFrame]] = {}
    for damage_case in damage_cases:
        asset_outputs = []
        for asset_info in asset_data_details.itertuples(index=False):
            asset_output = process_asset_damage_by_source(asset_info, damage_case, output_dir)
            if not asset_output.empty:
                asset_outputs.append(asset_output)

        if not asset_outputs:
            raise ValueError(f"No direct-damage source attribution outputs were produced for {damage_case}.")

        direct_damage = pd.concat(asset_outputs, ignore_index=True)
        case_outputs[damage_case] = save_case_outputs(damage_case, direct_damage, output_dir)

    save_min_max_outputs(case_outputs, output_dir)
    return case_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--damage-cases",
        nargs="+",
        default=["minimum", "maximum"],
        choices=sorted(DAMAGE_CASE_PARAMETERS),
    )
    parser.add_argument(
        "--asset-keys",
        nargs="*",
        default=None,
        help="Optional asset keys like roads_edges or rail_nodes for a partial run.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_source_id_attribution(
        output_dir=args.output_dir,
        damage_cases=args.damage_cases,
        asset_keys=args.asset_keys,
    )
