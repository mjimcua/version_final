"""
test_explorer.py — The explorer of a run (sff_explorer.py) over the synthetic run.

A · every page builds from the results in memory, with every control in every value it offers
B · every page builds from the SQL tables of the same run, and says the same numbers
C · the widget opens every tab without an error (the tabs are built when opened)
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import contextlib
import io
import os
import tempfile

import plotly.graph_objects as go

from main import run
from sff_explorer import PAGES, SFFExplorer, Cards, Table
from test_helpers import check, console_of, finish, synthetic_with


def build_every_page(explorer: SFFExplorer) -> dict:
    """Every page with every value of its controls; the items of each, by a readable key."""
    pages = {}
    series = explorer.series_by_money()["s1_4_fs_id"]
    compositions = explorer.compositions_by_money()["composition_id"]
    pages["negocio"] = explorer.page("negocio")
    for dimension in explorer.breakdown_dimensions():
        pages[f"negocio · {dimension}"] = explorer.page("negocio", dimension=dimension, year=2027)
    pages["series"] = explorer.page("series")
    pages["series · filtradas"] = explorer.page("series", confidence=["low"], risk_levels=["B_prestado"], text="EU")
    for fs_id in series:
        for horizon in (1, 6):
            pages[f"ficha · {fs_id} · h{horizon}"] = explorer.page("ficha", fs_id=fs_id, exam_horizon=horizon)
    pages["ficha · desconocida"] = explorer.page("ficha", fs_id="NO|EXISTE")
    for composition_id in compositions:
        pages[f"escalera · {composition_id}"] = explorer.page("escalera", composition_id=composition_id)
        for horizon in (1, 6):
            pages[f"backtest · {composition_id} · h{horizon}"] = explorer.page("backtest", composition_id=composition_id, horizon=horizon)
    for horizon in (1, 6):
        pages[f"examen · h{horizon}"] = explorer.page("examen", horizon=horizon)
    for group in explorer.price_groups():
        pages[f"uplift · {group}"] = explorer.page("uplift", price_group=group)
    pages["estados"] = explorer.page("estados")
    pages["parametros"] = explorer.page("parametros")
    pages["parametros · todos · buscar"] = explorer.page("parametros", only_changed=False, text="uplift")
    return pages


def every_item_renders(pages: dict) -> bool:
    """Every item turns into HTML or is a figure that plotly can serialise."""
    for items in pages.values():
        for item in items:
            if isinstance(item, go.Figure):
                item.to_json()
            else:
                item.to_html()
    return True


def first_cards(items: list) -> list:
    return next(item.cards for item in items if isinstance(item, Cards))


def test_from_memory_and_sql() -> None:
    print("A · every page from the results in memory · B · the same from SQL")
    configuration = synthetic_with()
    results = {}
    console_of(lambda: results.update(run(configuration)))

    in_memory = SFFExplorer.from_results(results, configuration)
    pages = build_every_page(in_memory)
    check(len(pages) > 40 and every_item_renders(pages),
          f"{len(pages)} pages built from memory (every series, composition, horizon, dimension and price group): every item renders")
    business_cards = first_cards(pages["negocio"])
    check(any(label.startswith("Total 2026") for label, _, _ in business_cards)
          and any(label.startswith("Pipeline que vence en 2027") for label, _, _ in business_cards),
          "the business page answers how 2026 ends and what pipeline falls due in 2027")
    top_series = in_memory.series_by_money()["s1_4_fs_id"].iloc[0]
    card = pages[f"ficha · {top_series} · h1"]
    check(sum(isinstance(item, go.Figure) for item in card) >= 4,
          "the card of a series has its history, exam, techniques and money charts")
    expected = float(results["forecast_series"].set_index("s1_4_fs_id").loc[top_series, "s7_2_esperado_usd_total"])
    check(first_cards(card)[0][1] == "$" + f"{expected:,.0f}".replace(",", "."),
          "the card shows the expected USD of the dimension, formatted ($ and Spanish grouping)")
    check(any(isinstance(item, Table) and "role" in item.frame.columns and "uses" in set(item.frame["role"])
              for key, items in pages.items() if key.startswith("ficha · EU|A|0|0|0|0|tele") for item in items),
          "the card of a series that borrows shows who is in its composition (uses / lends)")
    unknown = pages["ficha · desconocida"]
    check(len(unknown) == 1 and "no está" in unknown[0].to_html(), "an unknown series says so, it does not break the page")

    from_sql = SFFExplorer.from_sql(configuration)
    sql_pages = build_every_page(from_sql)
    check(every_item_renders(sql_pages) and set(sql_pages) == set(pages),
          "the same pages build from the SQL tables of the run (periods as text, read once)")
    check(first_cards(sql_pages["negocio"]) == business_cards,
          "from SQL the business answers are the same numbers as from memory")
    check(first_cards(sql_pages[f"ficha · {top_series} · h1"]) == first_cards(card),
          "from SQL the card of the top series is the same as from memory")


def test_the_widget() -> None:
    print("C · the widget: every tab opens")
    configuration = synthetic_with()
    results = {}
    console_of(lambda: results.update(run(configuration)))
    explorer = SFFExplorer.from_results(results, configuration, figure_mode="html")
    with contextlib.redirect_stdout(io.StringIO()):
        widget = explorer.show()
        tab = widget.children[1]
        for index in range(len(PAGES)):
            tab.selected_index = index
        # the controls redraw their page: move every one of the card and the business
        card_controls = tab.children[2].children[0].children
        card_controls[1].value = 6
        business_controls = tab.children[0].children[0].children
        business_controls[0].value = explorer.breakdown_dimensions()[0]
    check(len(tab.children) == len(PAGES) and not explorer.page_errors,
          f"the {len(PAGES)} tabs open, and their controls redraw them, without an error {explorer.page_errors or ''}")
    exported = explorer.export_html(os.path.join(tempfile.mkdtemp(), "explorador.html"), top_series=3)
    content = open(exported, encoding="utf-8").read()
    check(content.count("plotly.js") >= 1 and "Ficha de serie" in content and "Backtest y técnica" in content,
          "export_html writes every page in one file (plotly embedded once)")


if __name__ == "__main__":
    test_from_memory_and_sql()
    test_the_widget()
    finish()
