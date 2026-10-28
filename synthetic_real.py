"""
synthetic_real.py — A synthetic extract that looks like the real one (Kamelot, run of 19-oct-2026).

The synthetic of the tests (synthetic_v3.py) is a feature-coverage matrix: a dozen series, each one
built to exercise one feature. This one is the opposite: a portfolio with the shape of the real
extract, so a run on it says the same things the real run says, and the decisions of the framework
can be tried on it before they are tried on the real data. It is NOT used by the tests' checks; the
tests keep synthetic_v3.

WHAT IT REPRODUCES (the real figure it was calibrated against, in brackets)
  · the columns of the production extract: three regional levels (30 countries), product level 1 and 2,
    renewal type, term, band, partner, the three marks, net_new, prev_OperationGroup, the exact
    discount (null when unknown), the sku, and the 17 measures
  · a COHORT simulation: what falls due in month t is what was bought (renewed or acquired) in t − term.
    So the pipeline of every month comes from the renewals of a year before, and the acquisitions are
    what the simulation of step 4.1 has to reproduce
  · the renewal rate: renewal type, term, band, region, product and partner effects; a random level per
    series that walks month to month (the techniques that follow a level win: holt_damped, ses, ma3);
    extra-binomial noise (φ ≈ 2); a portfolio level that rose in 2026 (2025: 0.63 → 2026: 0.67-0.69);
    no seasonality in the rate
  · the marks: the extract keeps their LAST state. In the closed months they are final; in the future
    ones only part of them is visible today (dormant 9 % at one month, 3 % at twelve; final 9-11 %)
  · the discount: unknown in ~59 % of the USD due; the renewers of a known bucket pay more when the
    discount ends (the contract rule 1/(1−d) realised from 98 % down to 66 %)
  · the discount is measured against the MSRP of the contract at expiry, so it already carries the increases
    between the purchase and the expiry (DISCOUNT_AGAINST_MSRP_AT_EXPIRY)
  · the price increases: per price group (3 regional levels × product level 1), steps of 3-8 % in a few
    common months (the 2025-03 wave); a renewal inside the 12 months after a step pays the new price on
    a due value bought at the old one
  · the isolated renewals (same product, no retention event): their ratio, its dispersion and the six bands
  · the acquisitions: Pure-New flat (level ≈ 0.97), Not-New halved from mid-2024 to mid-2025 (≈ 0.54)
  · the partner migration: DigitalRiver licences bought from 2023-11 move to NEXWAY, so the old series
    keep a long history with almost nothing due at the end, and the new ones start with a short one
  · the extract was taken in the middle of October 2026: the renewals of September are booked (≈ 99 %),
    October's half, later months almost nothing; the pipeline of 2027-09..12 is only partly built
    (the 1-year licences bought in 2026-09 and half of 2026-10)
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import math

import numpy as np
import pandas as pd

# ─── the calendar ────────────────────────────────────────────────────────────────
SIMULATION_START = "2016-01"        # the burn-in: three cycles of the 3-year licences before the extract starts
EXTRACT_START = "2022-12"           # the first month of the extract (the real one starts in December 2022)
EXTRACT_END = "2027-12"
LAST_COMPLETE_MONTH = "2026-08"     # the last month whose renewals are all booked
SNAPSHOT_MONTH = "2026-10"          # the extract was taken in the middle of this month
LAST_SIMULATED_RENEWAL = "2026-12"  # renewals later than this do not feed any visible pipeline (the extract stops here)
LAST_TRUTH_MONTH = "2027-08"        # the last month whose whole pipeline is in the extract: its future truth can be told
MIGRATION_MONTH = "2023-11"         # DigitalRiver licences bought from this month on are sold through NEXWAY

# what share of the renewals of a due month is booked at the snapshot (the rest, still to come)
BOOKED_RESULTS = {"2026-09": 0.995, "2026-10": 0.45, "2026-11": 0.004, "2026-12": 0.002}
# what share of what was bought in a month is already in the extract as pipeline (a licence exists
# once it is bought: September complete, October half, later almost nothing)
BOOKED_PURCHASES = {"2026-10": 0.45, "2026-11": 0.01, "2026-12": 0.004}

# ─── the dimensions: value, weight (share of the series drawn), effect on the logit of the rate ───
PRICE_SCALE = 0.80                     # calibrated: the average price of a due unit ≈ $35
# (level 1, level 2, level 3, weight, list price per unit in USD at a fixed exchange rate, before PRICE_SCALE)
REGIONS = [
    ("Europe", "North Europe", "DACH", 14.0, 46.0), ("Europe", "North Europe", "UK & Ireland", 5.0, 44.0),
    ("Europe", "North Europe", "Benelux", 3.0, 44.0), ("Europe", "North Europe", "Nordics", 2.0, 48.0),
    ("Europe", "North Europe", "Eastern Europe", 2.0, 26.0),
    ("Europe", "South Europe", "France", 6.0, 42.0), ("Europe", "South Europe", "Italy", 5.0, 38.0),
    ("Europe", "South Europe", "Iberia", 4.0, 36.0), ("Europe", "South Europe", "Greece", 1.0, 32.0),
    ("Europe", "Central Europe", "Poland", 2.0, 28.0), ("Europe", "Central Europe", "Czechia", 1.0, 28.0),
    ("Russia++", "Russia", "Russia", 18.0, 22.0),
    ("Russia++", "CIS", "Kazakhstan", 2.0, 18.0), ("Russia++", "CIS", "Belarus", 1.0, 16.0),
    ("Russia++", "CIS", "Uzbekistan", 0.5, 14.0),
    ("Americas", "North America", "US", 4.0, 48.0), ("Americas", "North America", "Canada", 2.0, 44.0),
    ("Americas", "LATAM", "Brazil", 4.0, 24.0), ("Americas", "LATAM", "Mexico", 3.0, 24.0),
    ("Americas", "LATAM", "SOLA", 1.5, 22.0),
    ("APAC", "Core APAC", "Greater China", 1.5, 30.0), ("APAC", "Core APAC", "Korea", 0.8, 32.0),
    ("APAC", "Core APAC", "AEC", 1.5, 26.0),
    ("APAC", "Japan & ANZ", "Japan", 2.0, 44.0), ("APAC", "Japan & ANZ", "Australia & Oceania", 1.5, 46.0),
    ("APAC", "South Asia", "India", 1.5, 14.0),
    ("META", "Middle East", "Middle East", 1.5, 30.0), ("META", "Turkey", "Turkey", 1.2, 20.0),
    ("META", "Africa", "Africa", 1.0, 22.0), ("META", "Gulf", "Saudi Arabia", 0.8, 36.0)]
REGION_LEVEL_1_EFFECT = {"Europe": 0.0, "Russia++": 0.35, "Americas": -0.10, "APAC": -0.15, "META": -0.25}
REGION_LEVEL_3_EFFECT_SD = 0.30

# (product level 1, product level 2, weight, effect, price multiplier)
PRODUCTS = [
    ("Front Line", "KIS MD", 30.0, 0.00, 1.0), ("Front Line", "KTS MD", 15.0, 0.05, 1.3),
    ("Front Line", "KAV", 8.0, -0.10, 0.7),
    ("New Portfolio", "Kaspersky Standard", 12.0, -0.15, 0.9), ("New Portfolio", "Kaspersky Plus", 14.0, -0.05, 1.2),
    ("New Portfolio", "Kaspersky Premium", 12.0, 0.05, 1.6),
    ("Standalone", "Standalone product", 5.0, -0.30, 0.6), ("Standalone", "Kaspersky VPN", 4.0, -0.40, 0.5)]

# (renewal type, weight, effect)
RENEWAL_TYPES = [("Auto Renewal", 0.78, 0.55), ("Manual Retention", 0.22, -1.00)]

# (term, weight, effect, months between two due dates, price multiplier)
TERMS = [("1 year", 0.73, 0.00, 12, 1.0), ("2 year", 0.24, -0.60, 24, 1.8), ("3 year", 0.03, -0.65, 36, 2.5)]

# (band, weight, effect, price multiplier): the rate barely changes with the band (bands 1-2 ≈ 1.5 pp lower)
BANDS = [("1", 43.0, -0.08, 1.0), ("2", 14.0, -0.06, 1.2), ("3", 27.0, 0.06, 1.5), ("4", 0.8, 0.06, 1.7),
         ("5", 14.0, 0.05, 2.0), ("6", 0.01, 0.40, 2.2), ("7", 0.015, 0.25, 2.4), ("10", 1.6, 0.00, 3.0),
         ("20", 0.08, -0.10, 5.0)]

# partners by region level 1: (partner, weight); the effect of every partner on the rate
PARTNERS_BY_REGION = {
    "Russia++": [("AllSoft", 0.5), ("NEXWAY", 0.5)],
    "Europe": [("NEXWAY", 0.55), ("DigitalRiver", 0.25), ("Avangate", 0.15), ("Digital Media", 0.05)],
    "default": [("NEXWAY", 0.6), ("Avangate", 0.3), ("Digital Media", 0.1)]}
PARTNER_EFFECT = {"NEXWAY": 0.0, "AllSoft": 0.10, "Avangate": -0.10, "DigitalRiver": 0.0, "Digital Media": -0.20}
# the share of the units whose discount is UNKNOWN in the extract, by partner
UNKNOWN_DISCOUNT_BY_PARTNER = {"NEXWAY": 0.48, "AllSoft": 0.95, "Avangate": 0.70, "DigitalRiver": 0.55,
                               "Digital Media": 0.80}

# ─── the origin of a due licence: net_new × prev_OperationGroup ───────────────────
# (net_new, prev_OperationGroup, share inside its net_new, effect on the logit)
ORIGINS = [("Retention_Not-New", "Renewal", 0.86, 0.00), ("Retention_Not-New", "Upgrade", 0.08, 0.10),
           ("Retention_Not-New", "Migration", 0.06, -0.15),
           ("Acquisition_Pure-New", "No Prev", 0.88, -0.55), ("Acquisition_Pure-New", "Trial", 0.12, -0.90),
           ("Acquisition_Not-New", "No Prev", 0.70, -0.35), ("Acquisition_Not-New", "Migration", 0.30, -0.50)]
RETENTION_ORIGINS, PURE_NEW_ORIGINS, NOT_NEW_ORIGINS = [0, 1, 2], [3, 4], [5, 6]
MANUAL_ACQUISITION_EFFECT = -0.40          # a manual first renewal is much weaker than an automatic one
PURE_NEW_SHARE_OF_ACQUISITIONS = 0.78

# ─── the marks: the combination of the three (final state) and its effect on the logit ───
MARK_COMBINATIONS = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (0, 1, 1), (1, 0, 1)]   # dormant, softcancel, not_installed
MARK_EFFECT = np.array([0.00, -1.15, -2.15, -0.70, -4.00, -4.10, -2.20])
# the final share of every combination, by renewal type and by whether the licence comes from an acquisition
MARK_SHARES = {("Auto Renewal", False): [0.815, 0.078, 0.073, 0.016, 0.014, 0.002, 0.002],
               ("Manual Retention", False): [0.880, 0.095, 0.004, 0.018, 0.000, 0.000, 0.003],
               ("Auto Renewal", True): [0.760, 0.085, 0.080, 0.050, 0.018, 0.004, 0.003],
               ("Manual Retention", True): [0.810, 0.115, 0.004, 0.060, 0.000, 0.000, 0.011]}
# the share of a final mark that is ALREADY visible in the extract, by months ahead of the last closed
# month (1 = the current month); beyond the table, the value of the key "later"
MARK_VISIBLE_BY_MONTHS_AHEAD = {
    "dormant": [1.00, 0.93, 1.00, 0.95, 0.86, 0.79, 0.75, 0.68, 0.62, 0.51, 0.46, 0.27],
    "softcancel": [0.94, 0.77, 0.72, 0.64, 0.58, 0.54, 0.54, 0.48, 0.42, 0.42, 0.42, 0.41],
    "not_installed": [0.83, 0.72, 0.82, 0.84, 0.97, 0.93, 0.93, 0.97, 0.97, 1.00, 1.00, 1.00]}
MARK_VISIBLE_LATER = {"dormant": 0.09, "softcancel": 0.06, "not_installed": 0.03}
# WHEN a softcancel is set: the share of the final softcancels already set j months before the due month,
# S(j) = P(k ≥ j), j = 0 … 12 (the last one: 12 months or more). It is the visibility above seen from the month of
# the snapshot (a future month m months after the last complete one is seen with k ≥ m − 2): so the extract and the
# maturation columns tell the same story
MATURATION_MONTHS = 12
SOFTCANCEL_SEEN_MONTHS_BEFORE = np.array([1.00] + MARK_VISIBLE_BY_MONTHS_AHEAD["softcancel"][2:12] + [MARK_VISIBLE_LATER["softcancel"]] * 2)
SOFTCANCEL_TIMING = np.append(-np.diff(SOFTCANCEL_SEEN_MONTHS_BEFORE), SOFTCANCEL_SEEN_MONTHS_BEFORE[-1])   # P(k = j)

# ─── the discount of the previous purchase: 9 known buckets + unknown ───────────────
DISCOUNT_BUCKETS = 9                                   # 0-10 %, 10-20 % … 80-90 %; index 9 = unknown
KNOWN_BUCKET_SHARES = {"retention_auto": [.50, .13, .12, .08, .06, .05, .04, .015, .005],
                       "retention_manual": [.20, .10, .12, .14, .14, .13, .10, .05, .02],
                       "acquisition": [.15, .05, .08, .12, .40, .10, .06, .03, .01]}
DISCOUNT_RATE_EFFECT = np.array([0, 0, 0, 0, 0, -0.25, -0.25, -0.25, -0.25, 0.0])
IMPLIED_DISCOUNT_OF_UNKNOWN = 0.15                      # the price of a due licence whose discount is unknown
# the discount the extract carries is measured against the MSRP of the contract AT EXPIRY (not at purchase): an
# increase between the purchase and the expiry is already in it. For a month still to come, the MSRP is the one of
# the snapshot (a later increase is not known yet). False = the discount of the purchase (the old synthetic)
DISCOUNT_AGAINST_MSRP_AT_EXPIRY = True
# what a RENEWER pays over what fell due, by the bucket of the previous discount (the discount ends or
# shrinks: realised from 98 % to 66 % of the contract rule), before the price increases
RENEWAL_PRICE_RELIEF = np.array([0.96, 1.00, 1.06, 1.16, 1.33, 1.71, 1.85, 2.43, 3.50, 1.05])
# the same for the ISOLATED renewals (same product, no retention event)
ISOLATED_PRICE_RELIEF = np.array([0.97, 1.02, 1.13, 1.17, 1.40, 1.67, 1.44, 3.10, 5.80, 1.00])
RENEWAL_PRICE_NOISE_SD = 0.05

# ─── the renewal rate: the logit of every cell ─────────────────────────────────────
RATE_INTERCEPT = 0.80                 # calibrated: the closed months renew ≈ 64 % of the units
SERIES_EFFECT_SD = 0.25               # a fixed level per series
SERIES_WALK_SD = 0.040                # its level walks month to month (a local level: ses / ma3 / holt win)
SERIES_MONTH_NOISE_SD = 0.20 
SERIES_TREND_SD = 0.020              # a slope per series (logit per month): what the medium horizon misses         # the month of a series moves beyond the binomial (φ ≈ 2)
CELL_MONTH_NOISE_SD = 0.06
PORTFOLIO_SHOCK_SD = 0.025            # what moves every series in the same month (no seasonality)
# the level of the whole portfolio (logit, added to every series), by month; linear between the knots
PORTFOLIO_LEVEL_KNOTS = [("2016-01", 0.00), ("2022-12", -0.15), ("2023-06", 0.25), ("2023-12", 0.15),
                         ("2024-06", -0.12), ("2025-01", -0.08), ("2025-08", -0.02), ("2026-08", 0.13),
                         ("2027-12", 0.13)]

# ─── the volumes ─────────────────────────────────────────────────────────────────
ACQUISITION_SEASON = [1.05, 0.95, 0.95, 0.90, 0.90, 0.90, 0.95, 1.00, 1.00, 1.05, 1.35, 1.30]   # January … December
STEADY_RENEWAL_RATE = 0.64            # to start the cohorts in their steady state
WINBACK_SHARE = 0.05                  # of what does not renew, what comes back the same month as a retention
                                      # (so acquisitions are ~27 % of what falls due, as in the real extract)
SERIES_SIZE_SIGMA = 2.4               # the sizes of the series: lognormal, heavy tail (few series carry the money)
# Not-New acquisitions halve from 2024-09 to 2025-06 (their level in the run: 0.54); Pure-New stay flat
NOT_NEW_DROP = ("2024-09", "2025-06", 0.48)
ACQUISITION_TREND_PER_YEAR = -0.10   # the acquisitions fall slowly: the pipeline of next year is ~4 % below this one

# ─── the price increases ─────────────────────────────────────────────────────────
# the months a price group may raise its tariff, with the probability that it does
INCREASE_MONTHS = {"2023-09": 0.30, "2023-11": 0.15, "2024-03": 0.15, "2024-04": 0.15, "2024-06": 0.15,
                   "2025-03": 0.60, "2025-08": 0.20, "2025-09": 0.20, "2026-01": 0.03, "2026-03": 0.04,
                   "2026-07": 0.04}
INCREASE_SIZE = (0.04, 0.09)
SPECIAL_INCREASES = [("Brazil", "New Portfolio", "2025-05", 0.30), ("Mexico", "New Portfolio", "2025-08", 0.35)]

# ─── the isolated renewals ───────────────────────────────────────────────────────
ISOLATED_SHARE_OF_RENEWED = (0.25, 0.60)     # of the renewed units of an eligible cell, the isolated ones
ISOLATED_SPREAD = (0.15, 0.32)               # the standard deviation of the ratio between licences of a cell
ISOLATED_BAND_EDGES = [0.95, 1.00, 1.05, 1.10, 1.20]
ISOLATED_BAND_NAMES = ["lt095", "095_100", "100_105", "105_110", "110_120", "ge120"]
REACQUIRED_SHARE = 0.03                      # of what did not renew, what is bought again (total_reacquired_*)


# ═══════════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════════

def month_index(months: pd.PeriodIndex, month_text: str) -> int:
    """The position of a 'YYYY-MM' month inside the simulated calendar."""
    return int((pd.Period(month_text, freq="M") - months[0]).n)


def logistic(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def portfolio_level(months: pd.PeriodIndex) -> np.ndarray:
    """The portfolio level of every month (logit), linear between the knots."""
    knot_positions = [month_index(months, month) for month, _ in PORTFOLIO_LEVEL_KNOTS]
    knot_values = [value for _, value in PORTFOLIO_LEVEL_KNOTS]
    return np.interp(np.arange(len(months)), knot_positions, knot_values)


def not_new_multiplier(months: pd.PeriodIndex) -> np.ndarray:
    """1 before the drop, linear down to its floor, the floor after."""
    start, end, floor = NOT_NEW_DROP
    return np.interp(np.arange(len(months)), [month_index(months, start), month_index(months, end)], [1.0, floor])


def mark_visibility_matrix(months_ahead: int) -> np.ndarray:
    """7 × 7: the probability that a licence with a FINAL mark combination shows each combination today
    (every mark of the combination is visible, or not, on its own)."""
    if months_ahead <= 0:
        return np.eye(len(MARK_COMBINATIONS))
    mark_names = ["dormant", "softcancel", "not_installed"]
    visible = []
    for name in mark_names:
        table = MARK_VISIBLE_BY_MONTHS_AHEAD[name]
        visible.append(table[months_ahead - 1] if months_ahead <= len(table) else MARK_VISIBLE_LATER[name])
    matrix = np.zeros((len(MARK_COMBINATIONS), len(MARK_COMBINATIONS)))
    for final_index, final in enumerate(MARK_COMBINATIONS):
        for shown_index, shown in enumerate(MARK_COMBINATIONS):
            probability = 1.0
            for mark_position in range(3):
                if final[mark_position] == 0 and shown[mark_position] == 1:
                    probability = 0.0                       # a mark that will never exist is never visible
                elif final[mark_position] == 1:
                    seen = visible[mark_position]
                    probability *= seen if shown[mark_position] == 1 else (1.0 - seen)
            matrix[final_index, shown_index] = probability
    return matrix / matrix.sum(axis=1, keepdims=True)


# ═══════════════════════════════════════════════════════════════════════════════════
# THE SERIES (one per combination of the business dimensions, without marks or origin)
# ═══════════════════════════════════════════════════════════════════════════════════

def draw_base_series(rng: np.random.Generator, number_of_series: int, units_due_per_month: float) -> pd.DataFrame:
    """The business series: every dimension drawn by its weight, duplicates merged, a size each.
    The sizes are scaled so that the portfolio has about units_due_per_month due in a steady month."""
    region_weights = np.array([region[3] for region in REGIONS])
    product_weights = np.array([product[2] for product in PRODUCTS])
    rows = []
    for _ in range(number_of_series):
        region = REGIONS[rng.choice(len(REGIONS), p=region_weights / region_weights.sum())]
        product = PRODUCTS[rng.choice(len(PRODUCTS), p=product_weights / product_weights.sum())]
        renewal = RENEWAL_TYPES[rng.choice(len(RENEWAL_TYPES), p=[r[1] for r in RENEWAL_TYPES])]
        term = TERMS[rng.choice(len(TERMS), p=[t[1] for t in TERMS])]
        band_weights = np.array([band[1] for band in BANDS])
        band = BANDS[rng.choice(len(BANDS), p=band_weights / band_weights.sum())]
        partners = PARTNERS_BY_REGION.get(region[0], PARTNERS_BY_REGION["default"])
        partner = partners[rng.choice(len(partners), p=[p[1] for p in partners])][0]
        rows.append({"tr_regional_level_1": region[0], "tr_regional_level_2": region[1], "tr_regional_level_3": region[2],
                     "tr_product_level_1": product[0], "tr_product_level_2": product[1], "tr_renewal_type": renewal[0],
                     "tr_term": term[0], "tr_band": band[0], "tr_master_partner_code": partner,
                     "_size": rng.lognormal(0.0, SERIES_SIZE_SIGMA),
                     "_list_price": PRICE_SCALE * region[4] * product[4] * band[3] * term[4],
                     "_term_months": term[3],
                     "_fixed_effect": (REGION_LEVEL_1_EFFECT[region[0]] + product[3] + renewal[2] + term[2] + band[2]
                                       + PARTNER_EFFECT[partner])})
    series = pd.DataFrame(rows)
    keys = [column for column in series.columns if not column.startswith("_")]
    series = (series.groupby(keys, as_index=False)
              .agg(_size=("_size", "sum"), _list_price=("_list_price", "first"), _term_months=("_term_months", "first"),
                   _fixed_effect=("_fixed_effect", "first")))
    # the twin of every DigitalRiver series: from the migration month on, its licences are sold through NEXWAY
    river = series[series["tr_master_partner_code"] == "DigitalRiver"]
    twins = river.assign(tr_master_partner_code="NEXWAY", _size=0.0)
    series = (pd.concat([series, twins]).groupby(keys, as_index=False)
              .agg(_size=("_size", "sum"), _list_price=("_list_price", "first"), _term_months=("_term_months", "first"),
                   _fixed_effect=("_fixed_effect", "first")))
    # the effect of the country, one draw per country
    countries = sorted(series["tr_regional_level_3"].unique())
    country_effect = dict(zip(countries, rng.normal(0.0, REGION_LEVEL_3_EFFECT_SD, len(countries))))
    series["_fixed_effect"] += series["tr_regional_level_3"].map(country_effect)
    series["_fixed_effect"] += rng.normal(0.0, SERIES_EFFECT_SD, len(series))
    # the acquisitions per month of every series: its share of the steady portfolio
    steady_acquisitions = units_due_per_month * (1.0 - STEADY_RENEWAL_RATE) * (1.0 - WINBACK_SHARE)
    series["_acquisitions"] = steady_acquisitions * series["_size"] / series["_size"].sum()
    return series.reset_index(drop=True)


def draw_price_index(rng: np.random.Generator, series: pd.DataFrame, months: pd.PeriodIndex) -> np.ndarray:
    """The tariff of every series, month by month (1 before any increase): the increases are drawn per
    price group (three regional levels × product level 1) and multiply the tariff from their month on."""
    group_columns = ["tr_regional_level_1", "tr_regional_level_2", "tr_regional_level_3", "tr_product_level_1"]
    groups = series[group_columns].drop_duplicates().itertuples(index=False)
    price_index = np.ones((len(series), len(months)))
    for group in groups:
        members = np.all([series[column].to_numpy() == value for column, value in zip(group_columns, group)], axis=0)
        tariff = np.ones(len(months))
        for month, probability in INCREASE_MONTHS.items():
            if rng.random() < probability:
                tariff[month_index(months, month):] *= 1.0 + rng.uniform(*INCREASE_SIZE)
        for country, product, month, size in SPECIAL_INCREASES:
            if group[2] == country and group[3] == product:
                tariff[month_index(months, month):] *= 1.0 + size
        price_index[members] = tariff
    return price_index


# ═══════════════════════════════════════════════════════════════════════════════════
# THE COHORTS: due, marks, discount, renewal — month by month
# ═══════════════════════════════════════════════════════════════════════════════════

def build_raw_like_real(seed: int = 11, number_of_series: int = 600, units_due_per_month: float = 150000.0,
                        return_truth: bool = False):
    """The synthetic extract. number_of_series business series (before marks and origins) and about
    units_due_per_month due in a steady month (the real extract: ~9,600 series and ~370,000 units).

    With return_truth=True, also what the extract cannot know yet: (raw, truth), truth with one row per
    future month up to LAST_TRUTH_MONTH and what really falls due and renews in it, with the FINAL marks
    (units and USD). It is what a forecast made from the extract should have predicted: the only honest
    exam of the whole forecast (marks distribution, uplift and extended horizon included)."""
    rng = np.random.default_rng(seed)
    truth_rng = np.random.default_rng(seed + 9000)     # its own generator: the extract is the same with or without it
    timing_rng = np.random.default_rng(seed + 7000)    # the dates of the softcancels: their own generator too
    months = pd.period_range(SIMULATION_START, EXTRACT_END, freq="M")
    series = draw_base_series(rng, number_of_series, units_due_per_month)
    price_index = draw_price_index(rng, series, months)
    number_of_series, number_of_months = len(series), len(months)
    term_months = series["_term_months"].to_numpy().astype(int)
    is_manual = (series["tr_renewal_type"] == "Manual Retention").to_numpy()
    partner = series["tr_master_partner_code"].to_numpy()

    # [1] the successor of every series: itself, or its NEXWAY twin for the DigitalRiver ones after the migration
    keys = [column for column in series.columns if not column.startswith("_")]
    position_by_key = {tuple(row): position for position, row in enumerate(series[keys].itertuples(index=False))}
    successor = np.arange(number_of_series)
    for position in np.where(partner == "DigitalRiver")[0]:
        twin_key = tuple("NEXWAY" if column == "tr_master_partner_code" else series.at[position, column] for column in keys)
        successor[position] = position_by_key[twin_key]
    migration_position = month_index(months, MIGRATION_MONTH)

    # [2] the purchases of every series and month: renewals and acquisitions (the cohorts start steady)
    purchases_renewed = np.zeros((number_of_series, number_of_months))
    purchases_pure_new = np.zeros((number_of_series, number_of_months))
    purchases_not_new = np.zeros((number_of_series, number_of_months))
    acquisitions = series["_acquisitions"].to_numpy()
    season = np.array([ACQUISITION_SEASON[month.month - 1] for month in months])
    not_new_level = not_new_multiplier(months)
    for position in range(number_of_months):
        expected = acquisitions * season[position] * np.exp(ACQUISITION_TREND_PER_YEAR * position / 12.0)
        pure_new = rng.poisson(expected * PURE_NEW_SHARE_OF_ACQUISITIONS)
        not_new = rng.poisson(expected * (1 - PURE_NEW_SHARE_OF_ACQUISITIONS) * not_new_level[position])
        target = successor if position >= migration_position else np.arange(number_of_series)
        np.add.at(purchases_pure_new[:, position], target, pure_new)
        np.add.at(purchases_not_new[:, position], target, not_new)
        # the first term of every series has nothing due yet: its renewals come from a steady past
        steady_share = 1.0 - (1.0 - STEADY_RENEWAL_RATE) * (1.0 - WINBACK_SHARE)
        steady = rng.poisson(acquisitions * steady_share / (1.0 - steady_share))
        purchases_renewed[:, position] += np.where(position < term_months, steady, 0)

    # [3] the renewal rate: a level per series that walks, the portfolio level and its monthly shock
    walk = np.cumsum(rng.normal(0.0, SERIES_WALK_SD, (number_of_series, number_of_months)), axis=1)
    walk += rng.normal(0.0, SERIES_TREND_SD, number_of_series)[:, None] * np.arange(number_of_months)[None, :]
    walk -= walk[:, [month_index(months, EXTRACT_START)]]   # the fixed effect is the level at the start of the extract
    series_logit = (RATE_INTERCEPT + series["_fixed_effect"].to_numpy()[:, None] + walk
                    + portfolio_level(months)[None, :] + rng.normal(0.0, PORTFOLIO_SHOCK_SD, number_of_months)[None, :]
                    + rng.normal(0.0, SERIES_MONTH_NOISE_SD, (number_of_series, number_of_months)))

    # the fixed shares of every series: origins, marks, discount buckets
    origin_net_new = np.array([0, 0, 0, 1, 1, 2, 2])                       # retention, pure new, not new
    origin_share_inside = np.array([origin[2] for origin in ORIGINS])
    origin_effect = np.array([origin[3] for origin in ORIGINS])
    is_acquisition_origin = origin_net_new > 0
    mark_shares = np.zeros((number_of_series, len(ORIGINS), len(MARK_COMBINATIONS)))
    for origin_position in range(len(ORIGINS)):
        acquisition = bool(is_acquisition_origin[origin_position])
        for series_position in range(number_of_series):
            renewal_type = "Manual Retention" if is_manual[series_position] else "Auto Renewal"
            shares = np.array(MARK_SHARES[(renewal_type, acquisition)])
            mark_shares[series_position, origin_position] = shares / shares.sum()
    unknown_share = np.array([UNKNOWN_DISCOUNT_BY_PARTNER[name] for name in partner])
    discount_shares = np.zeros((number_of_series, len(ORIGINS), DISCOUNT_BUCKETS + 1))
    # a series sells with one or two usual discounts (its promotion and its renewal offer), not with all of them:
    # its known units go to two buckets drawn from the shares of its kind, 75 % / 25 %
    for series_position in range(number_of_series):
        for kind, origin_positions in (("acquisition", PURE_NEW_ORIGINS + NOT_NEW_ORIGINS),
                                       ("retention_manual" if is_manual[series_position] else "retention_auto", RETENTION_ORIGINS)):
            kind_shares = np.array(KNOWN_BUCKET_SHARES[kind]) / np.sum(KNOWN_BUCKET_SHARES[kind])
            main_bucket, second_bucket = rng.choice(DISCOUNT_BUCKETS, size=2, replace=False, p=kind_shares)
            known = np.zeros(DISCOUNT_BUCKETS)
            known[main_bucket], known[second_bucket] = 0.75, 0.25
            known *= 1.0 - unknown_share[series_position]
            discount_shares[series_position, origin_positions] = np.append(known, unknown_share[series_position])
    # the exact discount of every bucket: 0 in the first one (no discount), the middle of the bucket in the others
    exact_discount_choices = np.array([[0.0, 0.0]] + [[0.10 * bucket + 0.05] * 2 for bucket in range(1, DISCOUNT_BUCKETS)])

    # [4] month by month: what falls due, split into origin × mark × discount, and what renews
    extract_start = month_index(months, EXTRACT_START)
    last_complete = month_index(months, LAST_COMPLETE_MONTH)
    last_renewal = month_index(months, LAST_SIMULATED_RENEWAL)
    last_truth = month_index(months, LAST_TRUTH_MONTH) if return_truth else last_renewal
    truth_rows = []
    booked_purchases = np.ones(number_of_months)
    for month_text, share in BOOKED_PURCHASES.items():
        booked_purchases[month_index(months, month_text)] = share
    booked_purchases[month_index(months, "2027-01"):] = 0.0
    collected = []
    series_positions = np.arange(number_of_series)
    for position in range(number_of_months):
        bought = position - term_months
        valid = bought >= 0
        # what falls due: what was bought one term before (only what is already in the extract, for the future)
        due_by_net_new = np.zeros((number_of_series, 3))
        for net_new_position, purchases in enumerate((purchases_renewed, purchases_pure_new, purchases_not_new)):
            true_due = np.where(valid, purchases[series_positions, np.maximum(bought, 0)], 0.0).astype(int)
            in_extract = booked_purchases[np.maximum(bought, 0)]
            due_by_net_new[:, net_new_position] = rng.binomial(true_due, np.where(valid, in_extract, 0.0))
        due_by_net_new = due_by_net_new.astype(int)
        # into origins (net_new × prev_OperationGroup)
        due_by_origin = np.zeros((number_of_series, len(ORIGINS)), dtype=int)
        for net_new_position, origin_positions in enumerate((RETENTION_ORIGINS, PURE_NEW_ORIGINS, NOT_NEW_ORIGINS)):
            shares = origin_share_inside[origin_positions] / origin_share_inside[origin_positions].sum()
            due_by_origin[:, origin_positions] = rng.multinomial(due_by_net_new[:, net_new_position], shares)
        # into the FINAL marks, then into the discount buckets
        due_by_mark = rng.multinomial(due_by_origin, mark_shares)
        due_cells = rng.multinomial(due_by_mark, np.broadcast_to(discount_shares[:, :, None, :], due_by_mark.shape + (DISCOUNT_BUCKETS + 1,)))
        # the rate of every cell and what renews (the true renewals, up to the last simulated month)
        renewed_cells = np.zeros_like(due_cells)
        if position <= last_truth:
            draw = rng if position <= last_renewal else truth_rng
            cell_logit = (series_logit[:, position][:, None, None, None] + origin_effect[None, :, None, None]
                          + (is_manual[:, None] & is_acquisition_origin[None, :])[:, :, None, None] * MANUAL_ACQUISITION_EFFECT
                          + MARK_EFFECT[None, None, :, None] + DISCOUNT_RATE_EFFECT[None, None, None, :]
                          + draw.normal(0.0, CELL_MONTH_NOISE_SD, due_cells.shape))
            renewed_cells = draw.binomial(due_cells, logistic(cell_logit))
            # what renews is bought again, and part of what does not comes back: both fall due one term
            # later as a retention
            won_back = draw.binomial((due_cells - renewed_cells).sum(axis=(1, 2, 3)), WINBACK_SHARE)
            target = successor if position >= migration_position else series_positions
            np.add.at(purchases_renewed[:, position], target, renewed_cells.sum(axis=(1, 2, 3)) + won_back)
        if position < extract_start:
            continue
        month_text = str(months[position])
        # the future truth: what really falls due and renews, with the final marks (before the extract hides it)
        if return_truth and last_complete < position <= last_truth:
            true_cells = cells_of_the_month(truth_rng, position, month_text, due_cells, renewed_cells, series, price_index,
                                            term_months, exact_discount_choices)
            truth_rows.append({"period": month_text, "vencen_unidades": true_cells["total_tr_units"].sum(),
                               "vencen_usd": true_cells["total_tr_usd"].sum(),
                               "renovadas_unidades": true_cells["total_renewed_units"].sum(),
                               "renovado_usd": true_cells["total_renewed_usd"].sum()})
        # the renewals the extract already carries: all of them up to the last complete month, part after
        if position > last_complete:
            renewed_cells = rng.binomial(renewed_cells, BOOKED_RESULTS.get(month_text, 0.0))
        # the marks the extract shows: final in the closed months, partly visible in the future ones
        months_ahead = position - last_complete
        if months_ahead >= 3:
            visibility = mark_visibility_matrix(months_ahead)
            shown = np.zeros_like(due_cells)
            for final_position in range(len(MARK_COMBINATIONS)):
                moved = rng.multinomial(due_cells[:, :, final_position, :], visibility[final_position])
                shown += np.moveaxis(moved, -1, 2)
            due_cells, renewed_cells = shown, np.zeros_like(shown)
        collected.append(cells_of_the_month(rng, position, month_text, due_cells, renewed_cells, series, price_index,
                                            term_months, exact_discount_choices, timing_rng=timing_rng))
    raw = finish_the_extract(pd.concat(collected, ignore_index=True), series)
    if return_truth:
        return raw, pd.DataFrame(truth_rows)
    return raw


def cells_of_the_month(rng, position, month_text, due_cells, renewed_cells, series, price_index, term_months,
                       exact_discount_choices, timing_rng=None) -> pd.DataFrame:
    """The non-empty cells of one month with their measures: due and renewed units and USD, the base of
    the renewers, the isolated renewals and their dispersion; and, with a timing_rng, the maturation columns of
    softcancel (softcancel_month_k: its units marked k months before due)."""
    series_position, origin_position, mark_position, bucket = np.nonzero(due_cells)
    due_units = due_cells[series_position, origin_position, mark_position, bucket].astype(float)
    renewed_units = renewed_cells[series_position, origin_position, mark_position, bucket].astype(float)
    known = bucket < DISCOUNT_BUCKETS
    exact_discount = np.where(known, exact_discount_choices[np.minimum(bucket, DISCOUNT_BUCKETS - 1),
                                                            rng.integers(0, 2, len(bucket))], np.nan)
    price_discount = np.where(known, exact_discount, IMPLIED_DISCOUNT_OF_UNKNOWN)
    bought_position = np.maximum(position - term_months[series_position], 0)
    tariff_when_bought = price_index[series_position, bought_position]
    tariff_now = price_index[series_position, position]
    due_price = series["_list_price"].to_numpy()[series_position] * tariff_when_bought * (1.0 - price_discount)
    increase_factor = tariff_now / tariff_when_bought
    noise = rng.lognormal(0.0, RENEWAL_PRICE_NOISE_SD, len(bucket))
    uplift = increase_factor * RENEWAL_PRICE_RELIEF[bucket] * noise
    due_usd = due_units * due_price
    renewers_base = np.minimum(renewed_units * due_price * rng.uniform(0.99, 1.01, len(bucket)), due_usd)
    # the isolated renewals: retentions without a softcancel, part of their renewed units
    retention = np.isin(origin_position, RETENTION_ORIGINS)
    softcancel = np.array([MARK_COMBINATIONS[mark][1] for mark in mark_position]) == 1
    eligible = retention & ~softcancel & (renewed_units > 0)
    isolated_units = np.where(eligible, np.round(renewed_units * rng.uniform(*ISOLATED_SHARE_OF_RENEWED, len(bucket))), 0.0)
    isolated_ratio = increase_factor * ISOLATED_PRICE_RELIEF[bucket] * rng.lognormal(0.0, 0.03, len(bucket))
    isolated_base = np.minimum(isolated_units * due_price, renewers_base)
    isolated_ratio = np.minimum(isolated_ratio, np.where(isolated_base > 0, renewed_units * due_price * uplift / np.maximum(isolated_base, 1e-9), 0.0))
    spread = rng.uniform(*ISOLATED_SPREAD, len(bucket))
    marks = np.array(MARK_COMBINATIONS)[mark_position]
    origins = np.array([(origin[0], origin[1]) for origin in ORIGINS])[origin_position]
    cells = series.iloc[series_position][[column for column in series.columns if not column.startswith("_")]].reset_index(drop=True)
    cells.insert(0, "period", month_text)
    cells["dormant"], cells["softcancel"], cells["not_installed"] = marks[:, 0], marks[:, 1], marks[:, 2]
    cells["net_new"], cells["prev_OperationGroup"] = origins[:, 0], origins[:, 1]
    cells["discount"] = discount_in_the_extract(exact_discount, series_position, bought_position, position, month_text,
                                                price_index)
    cells["total_tr_units"] = due_units
    cells["total_tr_usd"] = due_usd
    cells["total_renewed_units"] = renewed_units
    cells["total_renewed_usd"] = renewed_units * due_price * uplift
    cells["total_tr_usd_renewed"] = renewers_base
    not_renewed = np.maximum(due_units - renewed_units, 0).astype(int)
    reacquired = rng.binomial(not_renewed, REACQUIRED_SHARE) if month_text <= LAST_COMPLETE_MONTH else np.zeros(len(bucket))
    cells["total_reacquired_units"] = reacquired.astype(float)
    cells["total_reacquired_usd"] = reacquired * due_price * 0.9
    never_softcancel_share = np.where(softcancel, 0.0, rng.uniform(0.95, 0.99, len(bucket)))
    cells["total_tr_units_without_softcancel"] = due_units * never_softcancel_share
    cells["total_renewed_units_without_softcancel"] = isolated_units
    cells["total_renewed_usd_without_softcancel"] = isolated_base * isolated_ratio
    cells["total_tr_usd_renewed_without_softcancel"] = isolated_base
    cells["total_renewed_usd_sq_over_tr_isolated"] = isolated_base * (spread ** 2 + isolated_ratio ** 2)
    normal_cdf = np.vectorize(lambda z: 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))
    cumulative = [normal_cdf((edge - isolated_ratio) / spread) for edge in ISOLATED_BAND_EDGES]
    shares = [cumulative[0]] + [cumulative[i] - cumulative[i - 1] for i in range(1, len(ISOLATED_BAND_EDGES))] + [1 - cumulative[-1]]
    for band_name, share in zip(ISOLATED_BAND_NAMES, shares):
        cells[f"total_tr_usd_renewed_isolated_{band_name}"] = isolated_base * share
    if timing_rng is not None:
        add_softcancel_timing(cells, month_text, timing_rng)
    return cells


def discount_in_the_extract(exact_discount, series_position, bought_position, position, month_text, price_index):
    """The discount as the extract carries it. Against the MSRP at expiry: 1 − (1 − d) × tariff at purchase /
    tariff at expiry (the tariff of the snapshot for a month still to come), so a licence bought at list price
    with a 5 % increase in between carries 4.8 %. Null when unknown."""
    if not DISCOUNT_AGAINST_MSRP_AT_EXPIRY:
        return exact_discount
    months_after_snapshot = max((pd.Period(month_text, freq="M") - pd.Period(SNAPSHOT_MONTH, freq="M")).n, 0)
    reference_position = max(position - months_after_snapshot, 0)
    tariff_ratio = price_index[series_position, bought_position] / price_index[series_position, reference_position]
    return np.clip(1.0 - (1.0 - exact_discount) * tariff_ratio, 0.0, 1.0)


def add_softcancel_timing(cells: pd.DataFrame, month_text: str, timing_rng: np.random.Generator) -> None:
    """softcancel_month_0 … softcancel_month_12 on the cells with softcancel: their units spread by WHEN the mark was
    set. In a month after the snapshot only what was set by then can be in the extract: k ≥ the months between the
    snapshot and the due month (the rest are softcancels still to come, and they sit in the rows without the mark)."""
    marked = cells["softcancel"].to_numpy() == 1
    timing = np.full((len(cells), MATURATION_MONTHS + 1), np.nan)
    if marked.any():
        months_after_snapshot = (pd.Period(month_text, freq="M") - pd.Period(SNAPSHOT_MONTH, freq="M")).n
        first_possible = min(max(months_after_snapshot, 0), MATURATION_MONTHS)
        probabilities = SOFTCANCEL_TIMING.copy()
        probabilities[:first_possible] = 0.0
        timing[marked] = timing_rng.multinomial(cells.loc[marked, "total_tr_units"].astype(int).to_numpy(),
                                                probabilities / probabilities.sum())
    for months_before in range(MATURATION_MONTHS + 1):
        cells[f"softcancel_month_{months_before}"] = timing[:, months_before]


def finish_the_extract(raw: pd.DataFrame, series: pd.DataFrame) -> pd.DataFrame:
    """One row per combination of every column (the grain of the real extract: two cells that only
    differed in a mark not visible yet become one row), and the columns the real extract also carries."""
    timing_columns = [column for column in raw.columns if column.startswith("softcancel_month_")]
    measures = [column for column in raw.columns if column.startswith("total_")] + timing_columns
    keys = [column for column in raw.columns if column not in measures]
    raw["discount"] = raw["discount"].round(2)
    raw = raw.groupby(keys, dropna=False, as_index=False)[measures].sum()
    if timing_columns:                                  # the rows without the mark carry no date of it: null
        raw.loc[raw["softcancel"] != 1, timing_columns] = np.nan
    raw["sku"] = raw["tr_product_level_2"] + " · " + raw["tr_term"]
    # the term a renewal renews to (28-oct-2026): in this synthetic every licence renews to its own term, so the six
    # columns of the term transitions say exactly that (the real extract has transitions 2 → 1, 1 → 2…)
    years = pd.to_numeric(raw["tr_term"].str.extract(r"(\d+)")[0], errors="coerce")
    for destination_years in (1, 2, 3):
        same_term = years == destination_years
        raw[f"total_renewed_units_to_{destination_years}y"] = np.where(same_term, raw["total_renewed_units"], 0.0)
        raw[f"total_renewed_usd_to_{destination_years}y"] = np.where(same_term, raw["total_renewed_usd"], 0.0)
    raw["flag_time_series"] = 0
    raw["dummy_field"] = 1
    raw["_filter2"] = 1
    return raw.sort_values(["period"] + [key for key in keys if key != "period"]).reset_index(drop=True)
