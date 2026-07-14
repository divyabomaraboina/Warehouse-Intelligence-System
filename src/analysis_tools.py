import duckdb
from pathlib import Path
import glob
import os

os.chdir(Path(__file__).parent.parent)

# ── Gold parquet path ─────────────────────────────────────────────
train_files = glob.glob("data/processed/featured_train_*.parquet")
TRAIN_PATH  = max(train_files)
print(f"Using: {TRAIN_PATH}")


# ═══════════════════════════════════════════════════════════════
# CORE TOOLS
# ═══════════════════════════════════════════════════════════════

def get_total_sales():
    """Total warehouse and retail sales — overall revenue"""
    return duckdb.query(f"""
        SELECT
            ROUND(SUM("WAREHOUSE SALES"), 2) AS total_warehouse_sales,
            ROUND(SUM("RETAIL SALES"),    2) AS total_retail_sales
        FROM '{TRAIN_PATH}'
    """).df()


def get_top_suppliers(n: int = 10):
    """
    Top n suppliers by warehouse sales.
    Exact numbers — no LLM estimation.
    """
    return duckdb.query(f"""
        SELECT
            SUPPLIER,
            ROUND(SUM("WAREHOUSE SALES"), 2) AS total_warehouse_sales,
            ROUND(SUM("RETAIL SALES"),    2) AS total_retail_sales,
            ROUND(SUM("WAREHOUSE SALES") * 100.0 /
                  SUM(SUM("WAREHOUSE SALES")) OVER(), 2) AS share_pct
        FROM '{TRAIN_PATH}'
        WHERE "WAREHOUSE SALES" > 0
        GROUP BY SUPPLIER
        ORDER BY total_warehouse_sales DESC
        LIMIT {n}
    """).df()


def get_low_performing_suppliers(n: int = 10):
    """
    Bottom n suppliers by warehouse sales.
    Shows actual revenue — fixes hallucination issue.
    """
    return duckdb.query(f"""
        SELECT
            SUPPLIER,
            ROUND(SUM("WAREHOUSE SALES"), 2) AS total_warehouse_sales,
            ROUND(SUM("RETAIL SALES"),    2) AS total_retail_sales,
            COUNT(*) AS transaction_count,
            ROUND(SUM("WAREHOUSE SALES") * 100.0 /
                  SUM(SUM("WAREHOUSE SALES")) OVER(), 3) AS share_pct
        FROM '{TRAIN_PATH}'
        WHERE "WAREHOUSE SALES" > 0
        GROUP BY SUPPLIER
        ORDER BY total_warehouse_sales ASC
        LIMIT {n}
    """).df()


def get_mid_performing_suppliers():
    """
    Middle tier suppliers — 25th to 75th percentile revenue.
    Fixes: 'mid performer' query returning wrong data.
    """
    return duckdb.query(f"""
        WITH revenue_stats AS (
            SELECT
                SUPPLIER,
                ROUND(SUM("WAREHOUSE SALES"), 2) AS revenue
            FROM '{TRAIN_PATH}'
            WHERE "WAREHOUSE SALES" > 0
            GROUP BY SUPPLIER
        ),
        percentiles AS (
            SELECT
                PERCENTILE_CONT(0.25) WITHIN GROUP
                    (ORDER BY revenue) AS p25,
                PERCENTILE_CONT(0.75) WITHIN GROUP
                    (ORDER BY revenue) AS p75
            FROM revenue_stats
        )
        SELECT
            r.SUPPLIER,
            r.revenue AS total_warehouse_sales,
            ROUND(r.revenue * 100.0 /
                  SUM(r.revenue) OVER(), 2) AS share_pct
        FROM revenue_stats r, percentiles p
        WHERE r.revenue BETWEEN p.p25 AND p.p75
        ORDER BY r.revenue DESC
    """).df()


def get_suppliers_by_category(category: str, tier: str = "top", n: int = 5):
    """
    Suppliers filtered by product category AND tier.
    Fixes: follow-up questions like 'what about Wine suppliers?'

    Args:
        category: Beer | Wine | Liquor | Kegs | Non-Alcohol
        tier:     top | bottom | mid
        n:        number of results (top/bottom only)
    """
    if tier == "mid":
        query = f"""
        WITH revenue_stats AS (
            SELECT
                SUPPLIER,
                ROUND(SUM("WAREHOUSE SALES"), 2) AS revenue
            FROM '{TRAIN_PATH}'
            WHERE "WAREHOUSE SALES" > 0
              AND LOWER("ITEM TYPE") = LOWER('{category}')
            GROUP BY SUPPLIER
        ),
        percentiles AS (
            SELECT
                PERCENTILE_CONT(0.25) WITHIN GROUP
                    (ORDER BY revenue) AS p25,
                PERCENTILE_CONT(0.75) WITHIN GROUP
                    (ORDER BY revenue) AS p75
            FROM revenue_stats
        )
        SELECT
            r.SUPPLIER,
            r.revenue AS total_warehouse_sales,
            ROUND(r.revenue * 100.0 /
                  SUM(r.revenue) OVER(), 2) AS share_pct
        FROM revenue_stats r, percentiles p
        WHERE r.revenue BETWEEN p.p25 AND p.p75
        ORDER BY r.revenue DESC
        """
    else:
        order = "DESC" if tier == "top" else "ASC"
        query = f"""
        SELECT
            SUPPLIER,
            ROUND(SUM("WAREHOUSE SALES"), 2) AS total_warehouse_sales,
            ROUND(SUM("RETAIL SALES"),    2) AS total_retail_sales,
            ROUND(SUM("WAREHOUSE SALES") * 100.0 /
                  SUM(SUM("WAREHOUSE SALES")) OVER(), 2) AS share_pct
        FROM '{TRAIN_PATH}'
        WHERE "WAREHOUSE SALES" > 0
          AND LOWER("ITEM TYPE") = LOWER('{category}')
        GROUP BY SUPPLIER
        ORDER BY total_warehouse_sales {order}
        LIMIT {n}
        """

    result = duckdb.query(query).df()
    result.insert(0, 'category', category)
    result.insert(1, 'tier', tier)
    return result


def get_sales_by_month():
    """Monthly warehouse and retail sales"""
    return duckdb.query(f"""
        SELECT
            MONTH,
            ROUND(SUM("WAREHOUSE SALES"), 2) AS total_warehouse_sales,
            ROUND(SUM("RETAIL SALES"),    2) AS total_retail_sales
        FROM '{TRAIN_PATH}'
        GROUP BY MONTH
        ORDER BY MONTH ASC
    """).df()


def get_sales_by_item_type():
    """All item types ranked by warehouse sales — exact share percentages"""
    return duckdb.query(f"""
        SELECT
            "ITEM TYPE" AS item_type,
            ROUND(SUM("WAREHOUSE SALES"), 2) AS total_warehouse_sales,
            ROUND(SUM("RETAIL SALES"),    2) AS total_retail_sales,
            ROUND(SUM("WAREHOUSE SALES") * 100.0 /
                  SUM(SUM("WAREHOUSE SALES")) OVER(), 2) AS share_pct
        FROM '{TRAIN_PATH}'
        WHERE "WAREHOUSE SALES" > 0
        GROUP BY item_type
        ORDER BY total_warehouse_sales DESC
    """).df()


def get_items_ranked():
    """
    All items ranked highest to lowest with rank number.
    Fixes: underrated/overrated item queries.
    """
    return duckdb.query(f"""
        SELECT
            RANK() OVER (ORDER BY SUM("WAREHOUSE SALES") DESC) AS rank,
            "ITEM TYPE" AS item_type,
            ROUND(SUM("WAREHOUSE SALES"), 2) AS total_warehouse_sales,
            ROUND(SUM("RETAIL SALES"),    2) AS total_retail_sales,
            ROUND(SUM("WAREHOUSE SALES") * 100.0 /
                  SUM(SUM("WAREHOUSE SALES")) OVER(), 2) AS share_pct
        FROM '{TRAIN_PATH}'
        WHERE "WAREHOUSE SALES" > 0
        GROUP BY "ITEM TYPE"
        ORDER BY total_warehouse_sales DESC
    """).df()


def get_supplier_count():
    """Total unique supplier count"""
    return duckdb.query(f"""
        SELECT COUNT(DISTINCT SUPPLIER) AS total_suppliers
        FROM '{TRAIN_PATH}'
        WHERE "WAREHOUSE SALES" > 0
    """).df()


def generate_summary():
    """
    Complete business summary.
    Agent tool — full picture answer.
    """
    total    = get_total_sales()
    top      = get_top_suppliers()
    by_month = get_sales_by_month()
    by_item  = get_items_ranked()
    low      = get_low_performing_suppliers()
    mid      = get_mid_performing_suppliers()

    return {
        "total_sales"      : total,
        "top_suppliers"    : top,
        "low_suppliers"    : low,
        "mid_suppliers"    : mid,
        "monthly_trends"   : by_month,
        "item_performance" : by_item,
    }


# ═══════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n=== Total Sales ===")
    print(get_total_sales())

    print("\n=== Top Suppliers ===")
    print(get_top_suppliers(5))

    print("\n=== Bottom Suppliers ===")
    print(get_low_performing_suppliers(5))

    print("\n=== Mid Suppliers ===")
    print(get_mid_performing_suppliers().head(5))

    print("\n=== Beer Top Suppliers ===")
    print(get_suppliers_by_category("Beer", "top", 5))

    print("\n=== Beer Bottom Suppliers ===")
    print(get_suppliers_by_category("Beer", "bottom", 5))

    print("\n=== Wine Mid Suppliers ===")
    print(get_suppliers_by_category("Wine", "mid"))

    print("\n=== Items Ranked ===")
    print(get_items_ranked())

    print("\n=== Monthly Sales ===")
    print(get_sales_by_month())