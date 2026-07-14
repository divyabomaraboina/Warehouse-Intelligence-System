from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

DOCS_DIR = PROJECT_ROOT / "docs"
MODELS_DIR = PROJECT_ROOT / "models"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma_db"
MANIFEST_PATH = CHROMA_DIR / "rag_manifest.json"

COLLECTION_NAME = "warehouse_knowledge"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

CHROMA_DIR.mkdir(parents=True, exist_ok=True)

_embedding_model: SentenceTransformer | None = None
_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
_collection = _client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)


# 

# ─────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────
def get_embedding_model() -> SentenceTransformer:
    global _embedding_model

    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    return _embedding_model


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def source_files() -> list[Path]:
    candidates = [
        DOCS_DIR / "data_dictionary.md",
        DOCS_DIR / "project_log.md",
        MODELS_DIR / "ab_results.json",
        MODELS_DIR / "metrics.json",
    ]

    processed_dir = PROJECT_ROOT / "data" / "processed"
    if processed_dir.exists():
        candidates.extend(processed_dir.glob("*.parquet"))

    return sorted(path for path in candidates if path.exists())


def calculate_source_signature() -> str:
    digest = hashlib.sha256()

    for path in source_files():
        stat = path.stat()
        digest.update(str(path.relative_to(PROJECT_ROOT)).encode("utf-8"))
        digest.update(str(stat.st_size).encode("utf-8"))
        digest.update(str(stat.st_mtime_ns).encode("utf-8"))

    return digest.hexdigest()


def load_manifest() -> dict[str, Any]:
    return read_json(MANIFEST_PATH)


def save_manifest(signature: str, chunk_count: int) -> None:
    payload = {
        "signature": signature,
        "chunk_count": chunk_count,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "collection": COLLECTION_NAME,
    }
    MANIFEST_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ─────────────────────────────────────────────────────────────
# Verified project content
# ─────────────────────────────────────────────────────────────
def generate_key_findings() -> str:
    try:
        from src.analysis_tools import (
            get_items_ranked,
            get_low_performing_suppliers,
            get_mid_performing_suppliers,
            get_sales_by_item_type,
            get_sales_by_month,
            get_top_suppliers,
            get_total_sales,
        )

        total = get_total_sales()
        suppliers = get_top_suppliers(10)
        monthly = get_sales_by_month()
        items = get_sales_by_item_type()
        ranked_items = get_items_ranked()
        bottom_suppliers = get_low_performing_suppliers(5)
        middle_suppliers = get_mid_performing_suppliers()

        warehouse_total = float(total.iloc[0]["total_warehouse_sales"])
        retail_total = float(total.iloc[0]["total_retail_sales"])

        top_supplier = str(suppliers.iloc[0]["SUPPLIER"])
        top_supplier_sales = float(suppliers.iloc[0]["total_warehouse_sales"])
        top_supplier_share = (
            top_supplier_sales / warehouse_total * 100 if warehouse_total else 0
        )

        peak_row = monthly.loc[monthly["total_warehouse_sales"].idxmax()]
        peak_month_number = int(peak_row["MONTH"])
        peak_month_sales = float(peak_row["total_warehouse_sales"])

        month_names = {
            1: "January",
            2: "February",
            3: "March",
            4: "April",
            5: "May",
            6: "June",
            7: "July",
            8: "August",
            9: "September",
            10: "October",
            11: "November",
            12: "December",
        }
        peak_month = month_names.get(peak_month_number, str(peak_month_number))

        beer_rows = items[items["item_type"].str.lower() == "beer"]
        beer_sales = float(beer_rows.iloc[0]["total_warehouse_sales"]) if not beer_rows.empty else 0
        beer_share = beer_sales / warehouse_total * 100 if warehouse_total else 0

        experiment = read_json(MODELS_DIR / "ab_results.json")
        metrics = read_json(MODELS_DIR / "metrics.json")

        annual_opportunity = float(experiment.get("annual_opportunity", 0))
        p_value = float(experiment.get("p_value", 0))
        summer_average = float(experiment.get("summer_avg", 0))
        non_summer_average = float(experiment.get("non_summer_avg", 0))
        seasonal_items = experiment.get("seasonal_items", [])
        non_seasonal_items = experiment.get("non_seasonal_items", [])

        xgb_accuracy = float(metrics.get("xgb_accuracy", 0)) * 100
        logistic_accuracy = float(metrics.get("logistic_accuracy", 0)) * 100
        best_model = str(metrics.get("best_model", "XGBoost"))
        train_size = int(metrics.get("train_size", 0))
        test_size = int(metrics.get("test_size", 0))

        ranked_text = "\n".join(
            (
                f"{int(row['rank'])}. {row['item_type']}: "
                f"${float(row['total_warehouse_sales']):,.0f} in warehouse sales, "
                f"{float(row['share_pct']):.2f}% share."
            )
            for _, row in ranked_items.iterrows()
        )

        bottom_text = "\n".join(
            (
                f"{row['SUPPLIER']}: "
                f"${float(row['total_warehouse_sales']):,.0f} in warehouse sales."
            )
            for _, row in bottom_suppliers.iterrows()
        )

        middle_text = ", ".join(middle_suppliers.head(5)["SUPPLIER"].astype(str).tolist())

        return f"""
WAREIQ VERIFIED BUSINESS FINDINGS

TIME CONTEXT
- Historical data covers 2017 through 2020.
- The latest available observed data is from 2020.
- The app does not contain live warehouse data.
- Future values are model estimates, not observed facts.

OVERALL SALES
- Total warehouse sales: ${warehouse_total:,.0f}.
- Total retail sales: ${retail_total:,.0f}.
- Beer contributed about {beer_share:.1f}% of warehouse sales.

SUPPLIER PERFORMANCE
- {top_supplier} was the highest-selling supplier.
- {top_supplier} generated ${top_supplier_sales:,.0f}.
- {top_supplier} contributed about {top_supplier_share:.1f}% of warehouse sales.
- Highest-selling means highest warehouse sales only.
- Profit margin, delivery reliability, lead time, quality, and return rates are not available.

CATEGORY RANKING
{ranked_text}

LOWEST-SELLING SUPPLIERS
{bottom_text}

MIDDLE-TIER SUPPLIER EXAMPLES
- Middle tier is defined using the middle part of the warehouse-sales distribution.
- Examples: {middle_text}.

MONTHLY PATTERN
- {peak_month} was the highest warehouse-sales calendar month.
- Historical warehouse sales for {peak_month}: ${peak_month_sales:,.0f}.

SUMMER COMPARISON
- Summer is defined as May through August.
- Summer average monthly sales: ${summer_average:,.0f}.
- Non-summer average monthly sales: ${non_summer_average:,.0f}.
- Estimated annual summer revenue difference: ${annual_opportunity:,.0f}.
- Statistical comparison p-value: {p_value:.2e}.
- Seasonal categories: {", ".join(seasonal_items) if seasonal_items else "Beer and Liquor"}.
- Categories without a confirmed summer pattern: {", ".join(non_seasonal_items) if non_seasonal_items else "Wine and Kegs"}.
- A seasonal relationship does not prove that season alone caused every sales change.

MODEL RESULTS
- Best-performing classification model: {best_model}.
- XGBoost accuracy: {xgb_accuracy:.1f}%.
- Logistic Regression accuracy: {logistic_accuracy:.1f}%.
- Training rows: {train_size:,}.
- Test rows: {test_size:,}.

DATA QUALITY
- A suspicious $7,800 record was removed before trusted analysis.
- The project identified a major 2018 data-coverage gap.
- Three leakage risks were corrected before final model training.

SAFE BUSINESS LANGUAGE
- Do not call a supplier best or worst based only on warehouse sales.
- Do not recommend exact stock quantities without inventory, lead time, safety stock, storage, shelf life, and service-level information.
- Low sales alone does not prove that an item is underrated.
""".strip()

    except Exception as exc:
        return f"""
WAREIQ PROJECT STATUS

The verified business findings could not be regenerated.
Reason: {exc}

Run the data preparation, model training, experiment, and forecast steps before rebuilding RAG.
""".strip()


def build_experiment_document() -> str | None:
    data = read_json(MODELS_DIR / "ab_results.json")
    if not data:
        return None

    seasonal = data.get("seasonal_items", [])
    non_seasonal = data.get("non_seasonal_items", [])

    return f"""
SUMMER SALES COMPARISON

Historical period: 2017 through 2020.
Summer definition: May through August.

Observed comparison:
- Summer average monthly warehouse sales: ${float(data.get("summer_avg", 0)):,.0f}.
- Non-summer average monthly warehouse sales: ${float(data.get("non_summer_avg", 0)):,.0f}.
- Estimated annual difference: ${float(data.get("annual_opportunity", 0)):,.0f}.
- Statistical comparison p-value: {float(data.get("p_value", 0)):.2e}.
- Effect size: {float(data.get("monthly_cohens_d", 0)):.2f}.

Category result:
- Categories with a confirmed summer pattern: {", ".join(seasonal) if seasonal else "Beer and Liquor"}.
- Categories without a confirmed summer pattern: {", ".join(non_seasonal) if non_seasonal else "Wine and Kegs"}.

Plain-language interpretation:
Summer months showed meaningfully different warehouse sales in the historical data.
This supports seasonal planning, but exact inventory decisions still require current stock,
lead time, storage, shelf life, and service-level information.
""".strip()


def build_model_document() -> str | None:
    data = read_json(MODELS_DIR / "metrics.json")
    if not data:
        return None

    return f"""
DEMAND MODEL RESULTS

Purpose:
The model classifies whether a warehouse-sales observation belongs to the higher-sales group.

Results:
- XGBoost accuracy: {float(data.get("xgb_accuracy", 0)) * 100:.1f}%.
- Logistic Regression accuracy: {float(data.get("logistic_accuracy", 0)) * 100:.1f}%.
- Best-performing model: {data.get("best_model", "XGBoost")}.
- Training rows: {int(data.get("train_size", 0)):,}.
- Test rows: {int(data.get("test_size", 0)):,}.

Important:
Accuracy describes performance on the prepared test data.
It does not guarantee the same result on new business conditions.
""".strip()


def build_forecast_document() -> str | None:
    try:
        import pandas as pd
        from src.forecast import forecast_future, load_and_prepare, train_prophet

        history = load_and_prepare()
        model = train_prophet(history)
        forecast = forecast_future(model, periods=6)
        future = forecast.tail(6).copy()

        if future.empty:
            return None

        peak = future.loc[future["yhat"].idxmax()]
        low = future.loc[future["yhat"].idxmin()]

        monthly_lines = "\n".join(
            (
                f"- {row['ds'].strftime('%B %Y')}: "
                f"estimated ${float(row['yhat']):,.0f}, "
                f"reasonable range ${float(row['yhat_lower']):,.0f} "
                f"to ${float(row['yhat_upper']):,.0f}."
            )
            for _, row in future.iterrows()
        )

        return f"""
FUTURE WAREHOUSE SALES ESTIMATE

Time type:
These are future estimates produced from historical data.
They are not observed sales and they are not guaranteed outcomes.

Model:
Prophet monthly forecast trained on 2017 through 2020 historical data.

Six-month estimate:
{monthly_lines}

Summary:
- Highest estimated month: {peak['ds'].strftime('%B %Y')}.
- Highest estimated sales: about ${float(peak['yhat']):,.0f}.
- Lowest estimated month: {low['ds'].strftime('%B %Y')}.
- Lowest estimated sales: about ${float(low['yhat']):,.0f}.

Known limitation:
The historical dataset contains a major 2018 coverage gap.

Business use:
The forecast shows direction and a reasonable range.
It does not provide an exact order date or exact order quantity.
Current inventory, lead time, safety stock, storage capacity, shelf life,
minimum order quantity, and service-level goals are still required.
""".strip()

    except Exception:
        return None


def load_documents() -> list[dict[str, str]]:
    documents: list[dict[str, str]] = []

    documentation_files = [
        (DOCS_DIR / "data_dictionary.md", "data_dictionary"),
        (DOCS_DIR / "project_log.md", "project_log"),
    ]

    for path, source in documentation_files:
        if path.exists():
            try:
                content = path.read_text(encoding="utf-8")
            except OSError:
                continue

            if content.strip():
                documents.append(
                    {
                        "content": content,
                        "source": source,
                        "type": "documentation",
                        "time_scope": "reference",
                    }
                )

    documents.append(
        {
            "content": generate_key_findings(),
            "source": "verified_findings",
            "type": "business_findings",
            "time_scope": "historical_and_latest",
        }
    )

    experiment = build_experiment_document()
    if experiment:
        documents.append(
            {
                "content": experiment,
                "source": "summer_comparison",
                "type": "analysis",
                "time_scope": "historical",
            }
        )

    model = build_model_document()
    if model:
        documents.append(
            {
                "content": model,
                "source": "model_results",
                "type": "model",
                "time_scope": "historical",
            }
        )

    forecast = build_forecast_document()
    if forecast:
        documents.append(
            {
                "content": forecast,
                "source": "future_estimate",
                "type": "forecast",
                "time_scope": "future",
            }
        )

    return documents


# ─────────────────────────────────────────────────────────────
# Chunking and storage
# ─────────────────────────────────────────────────────────────
def chunk_documents(documents: list[dict[str, str]]) -> list[dict[str, str]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=140,
        separators=["\n\n", "\n", ". ", " "],
    )

    chunks: list[dict[str, str]] = []

    for document in documents:
        pieces = splitter.split_text(document["content"])

        for index, piece in enumerate(pieces):
            cleaned = piece.strip()
            if len(cleaned) < 60:
                continue

            chunks.append(
                {
                    "content": cleaned,
                    "source": document["source"],
                    "type": document["type"],
                    "time_scope": document["time_scope"],
                    "chunk_id": f"{document['source']}_{index}",
                }
            )

    return chunks


def rebuild_collection(chunks: list[dict[str, str]]) -> None:
    global _collection

    try:
        _client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    _collection = _client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    if not chunks:
        return

    model = get_embedding_model()
    texts = [chunk["content"] for chunk in chunks]
    embeddings = model.encode(
        texts,
        show_progress_bar=False,
        batch_size=32,
    ).tolist()

    _collection.add(
        ids=[chunk["chunk_id"] for chunk in chunks],
        documents=texts,
        embeddings=embeddings,
        metadatas=[
            {
                "source": chunk["source"],
                "type": chunk["type"],
                "time_scope": chunk["time_scope"],
            }
            for chunk in chunks
        ],
    )


def initialize_rag(force: bool = False) -> None:
    signature = calculate_source_signature()
    manifest = load_manifest()

    collection_ready = _collection.count() > 0
    signature_matches = manifest.get("signature") == signature
    model_matches = manifest.get("embedding_model") == EMBEDDING_MODEL_NAME

    if not force and collection_ready and signature_matches and model_matches:
        return

    documents = load_documents()
    chunks = chunk_documents(documents)
    rebuild_collection(chunks)
    save_manifest(signature=signature, chunk_count=len(chunks))


def reset_rag() -> None:
    initialize_rag(force=True)


# ─────────────────────────────────────────────────────────────
# Query understanding
# ─────────────────────────────────────────────────────────────
def detect_time_scope(query: str) -> str:
    text = query.lower()

    future_words = {
        "future",
        "forecast",
        "predict",
        "prediction",
        "next month",
        "next 6 months",
        "upcoming",
        "will happen",
        "may happen",
        "expect",
    }

    latest_words = {
        "current",
        "currently",
        "today",
        "now",
        "latest",
        "present",
        "right now",
    }

    past_words = {
        "past",
        "historical",
        "history",
        "previous",
        "before",
        "did",
        "was",
        "were",
        "happened",
        "2017",
        "2018",
        "2019",
        "2020",
    }

    if any(word in text for word in future_words):
        return "future"

    if any(word in text for word in latest_words):
        return "latest"

    if any(word in text for word in past_words):
        return "historical"

    return "general"


def temporal_guidance(scope: str) -> str:
    if scope == "future":
        return (
            "TIME GUIDANCE: Answer as a future estimate. "
            "Clearly say the result is model-generated, not observed or guaranteed."
        )

    if scope == "latest":
        return (
            "TIME GUIDANCE: The dataset is not live. "
            "Treat 'latest' or 'current' as the latest available observed data through 2020."
        )

    if scope == "historical":
        return (
            "TIME GUIDANCE: Answer using observed historical results from 2017 through 2020."
        )

    return (
        "TIME GUIDANCE: Distinguish historical facts from future estimates whenever both appear."
    )


def search_knowledge(query: str, n_results: int = 4) -> str:
    initialize_rag()

    if _collection.count() == 0:
        return (
            "No project knowledge is available yet. "
            "Run the project pipeline and rebuild the knowledge store."
        )

    scope = detect_time_scope(query)
    model = get_embedding_model()
    query_embedding = model.encode([query], show_progress_bar=False).tolist()

    requested = max(1, min(int(n_results), _collection.count()))

    where = None
    if scope == "future":
        where = {"time_scope": "future"}

    try:
        results = _collection.query(
            query_embeddings=query_embedding,
            n_results=requested,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
    except Exception:
        results = _collection.query(
            query_embeddings=query_embedding,
            n_results=requested,
            include=["documents", "metadatas", "distances"],
        )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    context_parts = [temporal_guidance(scope)]

    for document, metadata, distance in zip(documents, metadatas, distances):
        similarity = max(0.0, 1.0 - float(distance))
        source = metadata.get("source", "project")
        time_scope = metadata.get("time_scope", "reference")

        context_parts.append(
            (
                f"SOURCE: {source}\n"
                f"TIME TYPE: {time_scope}\n"
                f"RELEVANCE: {similarity:.0%}\n"
                f"{document}"
            )
        )

    context_parts.append(
        """
ANSWER STYLE
- Speak like a friendly business assistant.
- Use simple language first.
- Do not mention RAG, embeddings, ChromaDB, SQL, tool names, or function names.
- Do not expose internal reasoning.
- Do not show an Analysis basis section unless the user asks how the result was calculated.
- Do not display empty labels such as Metric: None or Filters: None.
- Use exact numbers only when they appear in the supplied context.
- Explain technical ideas using plain business examples.
- Clearly label future values as estimates.
""".strip()
    )

    return "\n\n---\n\n".join(context_parts)


if __name__ == "__main__":
    reset_rag()

    test_questions = [
        "Hi",
        "Which supplier sold the most?",
        "Which month was busiest?",
        "What does the latest available data show?",
        "What may happen in the next six months?",
        "What did the data quality checks find?",
    ]

    for question in test_questions:
        print("\n" + "=" * 72)
        print(question)
        print("-" * 72)
        print(search_knowledge(question, n_results=3)[:1500])