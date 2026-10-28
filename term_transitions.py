"""
term_transitions.py — The term a renewal renews to, learned from the closed months (28-oct-2026).

A licence that falls due renews to a licence of 1, 2 or 3 years, not always of its own term:
  · 1 → 1     falls due again 12 months later: the pipeline of next year
  · 1 → 2, 3  more money this year (two or three years paid) and a GAP next year: it falls due 24 or 36 months later
  · 2, 3 → 1  less money per licence (the uplift of the transition) and pipeline next year that a projection of the
              1-year licences alone does not build
  · 2 → 2, 3 → 3 …  fall due beyond next year
The extract says it with six closed-month measures (renewed_units_to_1y_col … renewed_usd_to_3y_col): of the units and
USD that renewed in a row, those that renewed to every term.

THE RULE (the assumption of the forecast): the share of every destination is a reasonably constant proportion, the
sum of the last term_transition_window_months closed months, learned where it has support: every group takes the first
level of term_transition_levels (finest first, the origin term always in it) with term_transition_min_renewed_units
renewals in the window; the whole portfolio is the last resort. The share is learned in units and in USD apart, so the
pipeline a transition creates carries its own value (a 2 → 1 renewal is worth less than a 1 → 1).

Used by step 7.2 (the projected renewals of the simulation window).

The functions:
  term_years                 the years of a term written as text ('1 year' → 1)
  transition_columns         the declared columns: (short name, years, units column, USD column)
  learn_term_transitions     one row per level × group × origin term: renewals, the share of every destination,
                             whether it has support
  shares_of_rows             the share of every destination for some rows (the level that gave it, too)
  transitions_add_up         the six columns against the renewed units and USD of the closed months
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

from config import CALENDAR_ROLE_COLUMN, ROLE_PROJECTION, TERM_DESTINATIONS, Config

ORIGIN_YEARS_COLUMN = "plazo_origen_anos"
LEVEL_COLUMN = "nivel"
LEVEL_DIMS_COLUMN = "dims_del_nivel"
SUPPORT_COLUMN = "renovadas_unidades"
WITH_SUPPORT_COLUMN = "con_soporte"


def term_years(terms: pd.Series) -> pd.Series:
    """The years of a term written as text ('1 year', '2 year' → 1, 2); NaN when it says none."""
    return pd.to_numeric(terms.astype(str).str.extract(r"(\d+)")[0], errors="coerce")


def transition_columns(configuration: Config) -> list:
    """The declared columns of the term transitions: (short name, years, units column, USD column)."""
    if not configuration.term_transitions_declared:
        return []
    columns_by_destination = {"1y": (configuration.renewed_units_to_1y_col, configuration.renewed_usd_to_1y_col),
                              "2y": (configuration.renewed_units_to_2y_col, configuration.renewed_usd_to_2y_col),
                              "3y": (configuration.renewed_units_to_3y_col, configuration.renewed_usd_to_3y_col)}
    return [(short_name, years) + columns_by_destination[short_name] for short_name, years in TERM_DESTINATIONS]


def closed_rows_of_the_window(fine_table: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """The closed rows of the last term_transition_window_months months (the ones whose renewals are known)."""
    period = configuration.period_col
    closed = fine_table[fine_table[CALENDAR_ROLE_COLUMN] != ROLE_PROJECTION]
    if closed.empty:
        return closed
    last_closed = closed[period].max()
    first_month = last_closed - (configuration.term_transition_window_months - 1)
    return closed[closed[period] >= first_month]


def learn_term_transitions(fine_table: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """One row per level × group × origin term (the years of the licence that fell due): the renewals of the window
    with a known destination, the share of every destination in units and in USD, and whether the group has the
    support to use its own share. Empty when the transitions are not declared."""
    columns = transition_columns(configuration)
    if not columns:
        return pd.DataFrame()
    window = closed_rows_of_the_window(fine_table, configuration).copy()
    window[ORIGIN_YEARS_COLUMN] = term_years(window[configuration.term_column])
    window = window[window[ORIGIN_YEARS_COLUMN].notna()]
    unit_columns = [units_column for _, _, units_column, _ in columns]
    usd_columns = [usd_column for _, _, _, usd_column in columns]
    tables = []
    for level_number, level_dims in enumerate(configuration.term_transition_levels):
        dims = [dim for dim in level_dims if dim in window.columns]
        keys = dims + [ORIGIN_YEARS_COLUMN]
        grouped = window.groupby(keys, dropna=False)[unit_columns + usd_columns].sum().reset_index()
        renewed_units = grouped[unit_columns].sum(axis=1)
        renewed_usd = grouped[usd_columns].sum(axis=1)
        table = grouped[keys].copy()
        table[LEVEL_COLUMN] = level_number
        table[LEVEL_DIMS_COLUMN] = " × ".join(dims) if dims else "cartera"
        table[SUPPORT_COLUMN] = renewed_units
        table["renovado_usd"] = renewed_usd
        for short_name, _, units_column, usd_column in columns:
            table[f"parte_unidades_a_{short_name}"] = grouped[units_column] / renewed_units.replace(0, np.nan)
            table[f"parte_usd_a_{short_name}"] = grouped[usd_column] / renewed_usd.replace(0, np.nan)
        table[WITH_SUPPORT_COLUMN] = (renewed_units >= configuration.term_transition_min_renewed_units) | (not dims)
        tables.append(table)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def identity_shares(origin_years: pd.Series) -> dict:
    """Without anything learned for an origin term: every licence renews to its own term (what the forecast did until
    28-oct-2026)."""
    shares = {}
    for short_name, years in TERM_DESTINATIONS:
        same_term = (origin_years == years).astype(float)
        shares[f"parte_unidades_a_{short_name}"] = same_term
        shares[f"parte_usd_a_{short_name}"] = same_term
    return shares


def shares_of_rows(rows: pd.DataFrame, transitions: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """The share of every destination for every row (same index): from the first level, finest first, whose group of
    the row has support; the portfolio level always has it. The column nivel says which level gave it (−1: nothing
    learned for its origin term, every licence renews to its own term)."""
    share_columns = [f"parte_{kind}_a_{short_name}" for short_name, _ in TERM_DESTINATIONS for kind in ("unidades", "usd")]
    origin_years = term_years(rows[configuration.term_column])
    result = pd.DataFrame(index=rows.index)
    result[ORIGIN_YEARS_COLUMN] = origin_years
    for column, values in identity_shares(origin_years).items():
        result[column] = values.to_numpy()
    result[LEVEL_COLUMN] = -1
    if transitions is None or transitions.empty:
        return result
    pending = pd.Series(True, index=rows.index)
    for level_number, level_dims in enumerate(configuration.term_transition_levels):
        dims = [dim for dim in level_dims if dim in rows.columns]
        of_level = transitions[(transitions[LEVEL_COLUMN] == level_number) & transitions[WITH_SUPPORT_COLUMN]]
        if of_level.empty or not pending.any():
            continue
        keys = dims + [ORIGIN_YEARS_COLUMN]
        lookup = rows.loc[pending, dims].copy()
        lookup[ORIGIN_YEARS_COLUMN] = origin_years[pending]
        found = lookup.merge(of_level[keys + share_columns], on=keys, how="left")
        found.index = lookup.index
        matched = found[share_columns[0]].notna() | found[share_columns[1]].notna()
        matched_index = matched[matched].index
        for column in share_columns:
            result.loc[matched_index, column] = found.loc[matched_index, column].fillna(0.0).to_numpy()
        result.loc[matched_index, LEVEL_COLUMN] = level_number
        pending.loc[matched_index] = False
    return result


def transitions_add_up(fine_table: pd.DataFrame, configuration: Config) -> dict:
    """The six columns against the renewed units and USD of the closed months: the destination of how much of what
    renewed is known (1 = all of it; below, renewals to another term or not filled in)."""
    columns = transition_columns(configuration)
    closed = fine_table[fine_table[CALENDAR_ROLE_COLUMN] != ROLE_PROJECTION]
    renewed_units = float(closed[configuration.renewed_units_col].sum())
    renewed_usd = float(closed[configuration.renewed_usd_col].sum())
    known_units = float(sum(closed[units_column].sum() for _, _, units_column, _ in columns))
    known_usd = float(sum(closed[usd_column].sum() for _, _, _, usd_column in columns))
    return {"units": known_units / renewed_units if renewed_units else np.nan,
            "usd": known_usd / renewed_usd if renewed_usd else np.nan,
            "known_units": known_units, "renewed_units": renewed_units}
