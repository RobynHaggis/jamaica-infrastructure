import numpy as np
import pandas as pd
import scipy
import re
from scipy.integrate import simpson, trapezoid
import rasterio
import rasterio.features as features
from shapely.geometry import box
from rasterio.mask import mask



def mm_to_in(x_mm: float) -> float:
    return x_mm / 25.4

FX_JMD_PER_USD = 150.0  # 1 USD = 150 JMD
FX = 150.0

SCALE_MN = 1e6

# --- FX: 1 USD = 150 J$ -------------------------------------------------------
JMD_PER_USD = 150.0              # 1 USD = 150 J$
USD_PER_JMD = 1.0 / JMD_PER_USD  # = 0.006666...

# Keys in these dictionaries must match the "Classify" values in landuse shapefile
# N.B. deliberate matching of typos and double spaces
LANDUSE_FOREST_PROPORTION = {
    "Bamboo": 1,
    "Bamboo and Fields": 0.5,  # 50% Ag, 50% Bamboo (if you consider bamboo here)
    "Bamboo and Secondary Forest": 1,  # 50% Forest, 50% Ag (if you want some ag fraction)
    "Bare Rock": 0,
    "Bauxite Extraction": 0,
    "Buildings and other infrastructures": 0,
    "Closed broadleaved forest (Primary Forest)": 1,
    "Disturbed broadleaved forest (Secondary Forest)": 1,
    "Fields  and Bamboo": 0.5,  # 50% Ag, 50% Bamboo
    "Fields and Secondary Forest": 0.5,  # 50% Ag, 50% Forest
    "Fields or Secondary Forest/Pine Plantation": 0.5,  # 50% Ag, 50% Secondary Forest/Pine
    "Fields: Bare Land": 0,
    "Fields: Herbaceous crops, fallow, cultivated vegetables": 0,
    "Fields: Pasture,Human disturbed, grassland": 0,
    "Hardwood Plantation: Euculytus": 1,
    "Hardwood Plantation: Mahoe": 1,
    "Hardwood Plantation: Mahogany": 1,
    "Hardwood Plantation: Mixed": 1,
    "Herbaceous Wetland": 0,
    "Mangrove Forest": 0,
    "Open dry forest - Short": 1,
    "Open dry forest - Tall (Woodland/Savanna)": 1,
    "Plantation: Tree crops, shrub crops, sugar cane, banana": 1,
    "Quarry": 0,
    "Secondary Forest": 1,
    "Swamp Forest": 0,
    "Water Body": 0,
}


LANDUSE_AFFORESTABLE_PROPORTION = {
    "Bamboo": 0,
    "Bamboo and Fields": 0.5,  # 50% Ag, 50% Bamboo (if you consider bamboo here)
    "Bamboo and Secondary Forest": 0,  # 50% Forest, 50% Ag (if you want some ag fraction)
    "Bare Rock": 0,
    "Bauxite Extraction": 1,
    "Buildings and other infrastructures": 0,
    "Closed broadleaved forest (Primary Forest)": 0,
    "Disturbed broadleaved forest (Secondary Forest)": 0,
    "Fields  and Bamboo": 0.5,  # 50% Ag, 50% Bamboo
    "Fields and Secondary Forest": 0.5,  # 50% Ag, 50% Forest
    "Fields or Secondary Forest/Pine Plantation": 0.5,  # 50% Ag, 50% Secondary Forest/Pine
    "Fields: Bare Land": 1,
    "Fields: Herbaceous crops, fallow, cultivated vegetables": 1,
    "Fields: Pasture,Human disturbed, grassland": 1,
    "Hardwood Plantation: Euculytus": 0,
    "Hardwood Plantation: Mahoe": 0,
    "Hardwood Plantation: Mahogany": 0,
    "Hardwood Plantation: Mixed": 0,
    "Herbaceous Wetland": 0,
    "Mangrove Forest": 0,
    "Open dry forest - Short": 0,
    "Open dry forest - Tall (Woodland/Savanna)": 0,
    "Plantation: Tree crops, shrub crops, sugar cane, banana": 0,
    "Quarry": 1,
    "Secondary Forest": 0,
    "Swamp Forest": 0,
    "Water Body": 0,
}

# Map asset class to high-level sector
sector_map = {
    "buildings_assigned_economic_activity_areas": "buildings",
    "rail_edges": "transport",
    "rail_nodes": "transport",
    "roads_edges": "transport",
    "roads_nodes": "transport",
    "airport_polygon_areas": "transport",
    "port_polygon_areas": "transport",
    "irrigation_assets_NIC_edges": "water",
    "irrigation_assets_NIC_nodes": "water",
    "pipelines_NWC_edges": "water",
    "potable_facilities_NWC_nodes": "water",
    "waste_water_facilities_NWC_nodes": "water",
    "electricity_network_v3.1_nodes": "energy",
}


# --- Mappings ----------------------------------------------------------------
transport_subsector_map = {
    "rail_edges": "rail",
    "rail_nodes": "rail",
    "roads_edges": "roads",
    "roads_nodes": "roads",
    "airport_polygon_areas": "airports",
    "port_polygon_areas": "ports",
}
water_subsector_map = {
    "irrigation_assets_NIC_edges": "irrigation",
    "irrigation_assets_NIC_nodes": "irrigation",
    "pipelines_NWC_edges": "pipelines",
    "potable_facilities_NWC_nodes": "potable",
    "waste_water_facilities_NWC_nodes": "wastewater",
}


transport_cols = [
    "transport_subsector",
    "avoided_ead_JMD", "avoided_ead_USD",
    "avoided_ead_JMD_mn", "avoided_ead_USD_mn",
    "share_of_transport_pct",
]
water_cols = [
    "water_subsector",
    "avoided_ead_JMD", "avoided_ead_USD",
    "avoided_ead_JMD_mn", "avoided_ead_USD_mn",
    "share_of_water_pct",
]

def interpolate_damages(rp: float, rp_l: float, value_l: np.ndarray, rp_u: float, value_u: np.ndarray) -> np.ndarray:
    """Interpolate between two return period ndarrays
    """
    rp_factor = (np.log(rp) - np.log(rp_l)) / (np.log(rp_u) - np.log(rp_l))
    value = value_l + ((value_u - value_l) * rp_factor)

    return value


def pick_upper_lower_rps(rp: float, rps: list[float]) -> tuple[float]:
    bin_index = np.searchsorted(rps, rp, side="left")
    rp_l = rps[bin_index - 1]
    rp_u = rps[bin_index]
    return rp_l, rp_u

def get_rp_cols(df):
    rp_cols = [col for col in df.columns if "rp" in col]
    rps = [float(col.replace("rp", "")) for col in rp_cols]
    return rp_cols, rps




def interpolate_rp_damages(rps_to_calculate, damages):
    rp_cols, rps = get_rp_cols(damages)
    interpolated_damages = pandas.DataFrame()
    rps = sorted(rps)

    # print(f"{rps:} {rps_to_calculate:}")

    # Calculate and save interpolated damages for new return periods
    for rp in rps_to_calculate:
        rp_l, rp_u = pick_upper_lower_rps(rp, rps)
        rp_damages = interpolate_damages(rp, rp_l, damages[f"rp{rp_l}"], rp_u, damages[f"rp{rp_u}"])
        interpolated_damages[f"rp{rp}"] = rp_damages

    return interpolated_damages


def peak_flow_reduction(forest_percentage_change):
    # set up data frame with values from literature
    lookup = pandas.DataFrame({
        'catchment_forest_percentage': [0, 6, 14, 21, 62, 100],
        'rp5.0': [0, 3, 13, 18, 48, 48],
        'rp100.0': [0, 1, 8, 11, 32, 32],
    })
    data = lookup.catchment_forest_percentage
    rp_cols, rps = get_rp_cols(lookup)

    # set up interpolator
    interpolator = scipy.interpolate.RegularGridInterpolator(
        (rps, data),
        lookup[rp_cols].values.T,
        method='linear',
    )


    # set up return periods to get peak flow reduction values (includes 5 and 100)
    interp_rps = [5.0, 10.0, 20.0, 50.0, 100.0]

    # do the interpolation
    vals = interpolator(([interp_rps], np.array(forest_percentage_change).reshape(len(forest_percentage_change), 1))).T
    data = pandas.DataFrame({
        'forest_percentage_change': forest_percentage_change,
    })
    for rp, val in zip(interp_rps, vals):
        data[f"rp{rp}"] = val

    return data


def rp_change_given_flow_reduction(reduction_percent, interp_rp):
    ccra_flow_reductions = pandas.DataFrame({
        'reduction_percent': [0.0, 5.0, 10.0, 20.0, 40.0, 100.0],
        'rp2.0': [2.0, 2.4, 3.2, 6.8, 169,169],
        'rp2.3': [2.3, 2.8, 3.7, 7.9, 196,196],
        'rp5.0': [5.0, 6.5, 8.9, 19, 235, 235],
        'rp10.0': [10.0, 13,19,43,473, 473],
        'rp25.0': [25.0, 35, 51, 123, 1330, 1330],
        'rp50.0': [50.0, 72,107,268,2967, 2967],
        'rp100.0': [100.0, 147,224,582, 6648, 6648],
        'rp500.0': [500.0, 772,1229,3441,43094, 43094],
        'rp1000.0': [1000.0, 1571,2544,7345,95943, 95943],
    })

    # proportion of baseline flow
    ccra_flow_reductions['flow'] = (1 - ccra_flow_reductions.reduction_percent / 100)

    rp_cols, rps = get_rp_cols(ccra_flow_reductions)

    data = ccra_flow_reductions.flow
    flow_to_interpolate = 1 - reduction_percent / 100

    interpolator = scipy.interpolate.RegularGridInterpolator(
        (rps, data),
        ccra_flow_reductions[rp_cols].values.T,
        method='linear',
    )

    # set up return periods to get peak flow reduction values (includes 5 and 100)
    interp_rps = [interp_rp]

    # do the interpolation
    vals = interpolator(([interp_rps], np.array(flow_to_interpolate).reshape(len(flow_to_interpolate), 1))).T
    data = pandas.DataFrame({
        'reduction_percent': reduction_percent,
    })
    for rp, val in zip(interp_rps, vals):
        data[f"rp{rp}"] = val
    return data

def select_damages(sector_damages, variant='mean'):
    rps = [20.0, 50.0, 100.0, 200.0, 500.0, 1500.0]
    colnames = [f"fluvial__rp_{int(rp)}__rcp_baseline__epoch_2010__conf_None_{variant}" for rp in rps]
    to_rename = {colname: f"rp{rp}" for colname, rp in zip(colnames, rps)}
    selected_damages = sector_damages.set_index('HYBAS_ID')[colnames].rename(columns=to_rename).copy()
    selected_damages["rp1000000000.0"] = selected_damages["rp1500.0"].copy()
    selected_damages["rp0.0001"] = 0
    return selected_damages

def process_sector_damage_for_rp(rp, all_sector_damage, hydrobasins):
    """
    Process the damages for a given return period.
    
    Parameters:
      rp (int or float): the return period (e.g. 20, 50, 100)
      all_sector_damage (pd.DataFrame): DataFrame containing damage data.
      hydrobasins (pd.DataFrame): DataFrame with geometry and HYBAS_ID.
    
    Returns:
      pd.DataFrame: A DataFrame with standardized columns:
                  ["HYBAS_ID", "geometry", "damages", "damages_with_nbs", 
                   "avoided_damages", "rp"]
    """
    # Define column names based on the specified return period
    fluvial_col = f'baseline__fluvial__rp_{rp}__baseline__fluvial__rp_mean'
    future_col = f'future__fluvial__rp_{rp}__future__fluvial__rp_mean'
    
    # Group the damage data by HYBAS_ID and sum (in case you have duplicate HYBAS_IDs)
    damage_df = all_sector_damage[['HYBAS_ID', fluvial_col, future_col]].groupby('HYBAS_ID').sum()
    
    # Join with the hydrobasins geometry
    damage_df = hydrobasins[['HYBAS_ID', 'geometry']].set_index('HYBAS_ID').join(damage_df)
    
    # Calculate avoided damages (baseline minus future)
    damage_df[f'avoided__fluvial__rp_{rp}'] = damage_df[fluvial_col] - damage_df[future_col]
    
    # Add the return period column
    damage_df['rp'] = rp
    
    # Reset the index and rename the columns consistently:
    #   "damages" will be taken as the baseline fluvial damage column and
    #   "damages_with_nbs" as the future fluvial damage column,
    #   "avoided_damages" from the calculated column.
    damage_df = damage_df.reset_index()
    damage_df.columns = ["HYBAS_ID", "geometry", "damages", "damages_with_nbs", "avoided_damages", "rp"]
    
    return damage_df


def calculate_ead(df):
    rp_cols = [col for col in df.columns if re.match(r"^rp\d+(\.\d+)?$", col)]
    rp_vals = sorted([float(col.replace("rp", "")) for col in rp_cols])
    rp_vals = np.array(rp_vals[::-1])
    probabilities = 1 / rp_vals
    rp_damages = df[[f"rp{r}" for r in rp_vals]]
    # return simpson(rp_damages, x=probabilities, axis=1)
    return trapezoid(rp_damages, x=probabilities, axis=1)


# def calculate_ead(df):
#     # Filter only columns that match "rp" followed by digits.
#     rp_cols = [col for col in df.columns if re.match(r'^rp\d+$', col)]
#     # Sort the columns using the reciprocal of the integer value after "rp"
#     rp_cols = sorted(rp_cols, key=lambda col: 1 / float(col.replace("rp", "")))
#     rps = np.array([int(col.replace("rp", "")) for col in rp_cols])
#     probabilities = 1 / rps
#     rp_damages = df[rp_cols]
#     return simpson(rp_damages, x=probabilities, axis=1)


# 1) Zonal sum helper (sums raster values within each polygon; no area calc)
def zonal_sum_only(raster_path, gdf, id_col="catchment_uid", all_touched=False):
    rows = []
    with rasterio.open(raster_path) as src:
        gdf_proj = gdf.to_crs(src.crs).copy()
        gdf_proj["geometry"] = gdf_proj.geometry.buffer(0)  # repair invalid geoms
        rb = box(*src.bounds)
        gdf_proj = gdf_proj[gdf_proj.intersects(rb)].copy()

        nd = src.nodata
        scale = (src.scales[0] if getattr(src, "scales", None) else 1.0) or 1.0
        offset = (src.offsets[0] if getattr(src, "offsets", None) else 0.0) or 0.0

        for _, r in gdf_proj.iterrows():
            try:
                data, _ = mask(src, [r.geometry.__geo_interface__],
                               crop=True, filled=False, all_touched=all_touched)
            except ValueError:
                rows.append({id_col: r[id_col], "sum": 0.0})
                continue

            band = data[0].astype("float64")
            band = band * scale + offset  # apply scale/offset if present

            ma = np.ma.array(band, mask=np.ma.getmaskarray(band))  # keep outside masked
            if nd is not None:
                ma = np.ma.masked_where(band == nd, ma)
            ma = np.ma.masked_invalid(ma)

            rows.append({id_col: r[id_col], "sum": float(ma.sum()) if ma.count() else 0.0})

    out = pd.DataFrame(rows)
    # ensure every ID appears (zeros for non-overlapping polygons)
    all_ids = gdf[[id_col]].copy()
    out = all_ids.merge(out, on=id_col, how="left").fillna({"sum": 0.0})
    return out



def stats(df, col):
    n = len(df)
    v = df[col]
    pos = (v > 0).sum()
    zero = (v == 0).sum()
    neg = (v < 0).sum()   # just in case
    nan = v.isna().sum()
    return pd.Series({
        "catchments_total": n,
        "positive_count": int(pos),
        "zero_count": int(zero),
        "negative_count": int(neg),
        "nan_count": int(nan),
        "positive_share_pct": round(100 * pos / n, 1),
        "zero_share_pct": round(100 * zero / n, 1),
    })

def sum_by_catchment_usd(r_path, catchments_gdf, label):
    with rasterio.open(r_path) as src:
        crs, transform, shape = src.crs, src.transform, src.shape
        cats = catchments_gdf.to_crs(crs)[["catchment_uid", "geometry"]].copy()
        cats["catchment_uid"] = pd.to_numeric(cats["catchment_uid"], errors="coerce").astype("Int64")
        cats = cats.dropna(subset=["catchment_uid"]).copy()
        # optional geometry clean-up if needed:
        # cats["geometry"] = cats.geometry.buffer(0)

        # rasterize catchment IDs onto the SAME grid as the raster
        shapes_iter = ((geom, int(uid)) for geom, uid in zip(cats.geometry, cats["catchment_uid"]))
        lab = features.rasterize(
            shapes=shapes_iter,
            out_shape=shape,
            transform=transform,
            fill=0,
            all_touched=True,
            dtype="int32",
        )

        data = src.read(1, masked=True)  # masked nodata

    valid = ~data.mask
    labels = lab[valid]
    values = data.data[valid].astype("float64")

    keep = labels > 0
    labels, values = labels[keep], values[keep]

    # sums in J$
    sums = np.bincount(labels, weights=values)
    uids = np.nonzero(sums)[0]
    df = pd.DataFrame({"catchment_uid": uids, f"{label}_jmd": sums[uids]})

    # convert to USD and USD millions
    df[f"{label}_usd"]    = df[f"{label}_jmd"] / FX_JMD_PER_USD
    df[f"{label}_usd_mn"] = df[f"{label}_usd"] / 1e6

    national_jmd = float(values.sum())
    national_usd = national_jmd / FX_JMD_PER_USD
    print(f"{label}: national total = US${national_usd:,.2f}  (J${national_jmd:,.2f})")
    return df

# old function
# def zonal_sum_only(raster_path, gdf, id_col="catchment_uid", all_touched=True):
#     rows = []
#     with rasterio.open(raster_path) as src:
#         gdf_proj = gdf.to_crs(src.crs).copy()
#         gdf_proj["geometry"] = gdf_proj.geometry.buffer(0)
#         rb = box(*src.bounds)
#         gdf_proj = gdf_proj[gdf_proj.intersects(rb)].copy()

#         scale = (src.scales[0] if getattr(src, "scales", None) else 1.0) or 1.0
#         offset = (src.offsets[0] if getattr(src, "offsets", None) else 0.0) or 0.0

#         for _, r in gdf_proj.iterrows():
#             try:
#                 data, _ = mask(src, [r.geometry.__geo_interface__],
#                                crop=True, filled=False, all_touched=all_touched)
#             except ValueError:
#                 rows.append({id_col: r[id_col], "sum": 0.0}); continue

#             ma = np.ma.array(data[0], copy=False).astype("float64")
#             ma = ma * scale + offset
#             rows.append({id_col: r[id_col], "sum": float(ma.sum()) if ma.count() else 0.0})

#     out = pd.DataFrame(rows)
#     all_ids = gdf[[id_col]].copy()
#     return all_ids.merge(out, on=id_col, how="left").fillna({"sum": 0.0})



# --- 8) Pretty prints (1-decimal % as requested) ------------------------------
def _fmt_pct(v): return "" if pd.isna(v) else f"{v:.1%}"
def _fmt_money(v): return f"{v:,.2f}"



# --- Pretty print helpers -----------------------------------------------------
def _pretty_print(tbl, money_suffix):
    fmt = {
        "baseline_ead": "{:,.2f}".format,
        "future_ead":   "{:,.2f}".format,
        "avoided_ead":  "{:,.2f}".format,
        "avoided_share": "{:.1%}".format,
        "share_of_total_baseline": "{:.1%}".format,
        "share_of_total_avoided": "{:.1%}".format,
    }
    pretty = tbl.rename(columns={
        "baseline_ead": f"baseline_ead [{money_suffix}]",
        "future_ead":   f"future_ead [{money_suffix}]",
        "avoided_ead":  f"avoided_ead [{money_suffix}]",
    })
    print(pretty.to_string(index=False, formatters=fmt))


def _pretty(df, unit):
    d = df.copy()
    for c in money_cols: d[c] = d[c].map(_fmt_money)
    d["avoided_share"] = d["avoided_share"].map(_fmt_pct)
    for c in ("share_of_total_baseline", "share_of_total_avoided"):
        if c in d: d[c] = d[c].map(_fmt_pct)
    d = d.rename(columns={
        "baseline": f"baseline [{unit}]",
        "future":   f"future [{unit}]",
        "avoided":  f"avoided [{unit}]"
    })
    cols = ["sector","rp",
            f"baseline [{unit}]", f"future [{unit}]", f"avoided [{unit}]",
            "avoided_share","share_of_total_baseline","share_of_total_avoided"]
    return d[cols].sort_values(["sector","rp"])



def _pretty(df, unit):
    d = df.copy()
    for c in money_cols: d[c] = d[c].map(_fmt_money)
    d["avoided_share"] = d["avoided_share"].map(_fmt_pct)
    for c in ("share_of_total_baseline", "share_of_total_avoided"):
        if c in d: d[c] = d[c].map(_fmt_pct)
    d = d.rename(columns={
        "baseline": f"baseline [{unit}]",
        "future":   f"future [{unit}]",
        "avoided":  f"avoided [{unit}]"
    })
    cols = ["sector","rp",
            f"baseline [{unit}]", f"future [{unit}]", f"avoided [{unit}]",
            "avoided_share","share_of_total_baseline","share_of_total_avoided"]
    return d[cols].sort_values(["sector","rp"])





# --- Helper: summarise any subsector mapping (no unmapped reporting) ---------
def build_subsector_summary(df, mapping, new_col, total_label, unit="mn"):
    """
    df: DataFrame with ['asset_class','avoided_ead'] in JMD
    mapping: dict asset_class -> subsector label
    new_col: subsector column name to create
    unit: 'mn' (1e6) or 'bil' (1e9) for scaled columns
    """
    scale = 1e6 if unit == "mn" else 1e9
    jmd_scaled = f"avoided_ead_JMD_{'mn' if unit=='mn' else 'bil'}"
    usd_scaled = f"avoided_ead_USD_{'mn' if unit=='mn' else 'bil'}"

    tmp = df.copy()
    tmp[new_col] = tmp["asset_class"].map(mapping)
    mapped = tmp.loc[tmp[new_col].notna()].copy()

    # Aggregate
    sums = (mapped.groupby(new_col, as_index=False)
                  .agg(avoided_ead_JMD=("avoided_ead", "sum")))
    total = float(sums["avoided_ead_JMD"].sum())

    # Currency + scaled columns
    sums["avoided_ead_USD"] = sums["avoided_ead_JMD"] * USD_PER_JMD
    sums[jmd_scaled] = sums["avoided_ead_JMD"] / scale
    sums[usd_scaled] = sums["avoided_ead_USD"] / scale

    # Share (% of sector total)
    if total > 0:
        sums["share_pct"] = 100.0 * sums["avoided_ead_JMD"] / total
    else:
        sums["share_pct"] = np.nan

    # Sort + round for neatness
    display_df = (sums.sort_values("avoided_ead_JMD", ascending=False)
                       .assign(**{
                           jmd_scaled: lambda d: d[jmd_scaled].round(1),
                           usd_scaled: lambda d: d[usd_scaled].round(1),
                           "share_pct": lambda d: d["share_pct"].round(1),
                       }))

    # TOTAL row
    total_row = pd.DataFrame([{
        new_col: total_label,
        "avoided_ead_JMD": total,
        "avoided_ead_USD": total * USD_PER_JMD,
        jmd_scaled: round(total / scale, 1),
        usd_scaled: round(total * USD_PER_JMD / scale, 1),
        "share_pct": 100.0 if total > 0 else np.nan,
    }])

    return pd.concat([display_df, total_row], ignore_index=True)

# --- Optional helper: sector summary (keeps both currencies) -----------------
def build_sector_summary(df):
    tmp = df.copy()
    if "sector" not in tmp.columns:
        if "sector_map" in globals():            # provide sector_map elsewhere if needed
            tmp["sector"] = tmp["asset_class"].map(sector_map)
        else:
            return None

    sums = (tmp.dropna(subset=["sector"])
               .groupby("sector", as_index=False)
               .agg(avoided_ead_JMD=("avoided_ead", "sum")))
    sums["avoided_ead_USD"] = sums["avoided_ead_JMD"] * USD_PER_JMD
    sums["avoided_ead_JMD_bil"] = sums["avoided_ead_JMD"] / 1e9
    sums["avoided_ead_USD_bil"] = sums["avoided_ead_USD"] / 1e9

    total = float(sums["avoided_ead_JMD"].sum())
    total_row = pd.DataFrame([{
        "sector": "TOTAL",
        "avoided_ead_JMD": total,
        "avoided_ead_USD": total * USD_PER_JMD,
        "avoided_ead_JMD_bil": total / 1e9,
        "avoided_ead_USD_bil": total * USD_PER_JMD / 1e9,
    }])

    return pd.concat(
        [sums.sort_values("avoided_ead_USD", ascending=False), total_row], ignore_index=True
    )



