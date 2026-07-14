# WareIQ — Warehouse Sales Intelligence Assistant

WareIQ is a conversational warehouse analytics application that turns raw sales data into simple, traceable business answers.

Instead of forcing users to read technical dashboards, WareIQ lets them ask questions in plain language:

- Which supplier sold the most?
- Which month was the busiest?
- What changed during summer?
- What may happen in the next six months?
- What data quality issues were found?

The system combines:

- DuckDB analytics
- Medallion architecture
- Data validation
- Statistical testing
- Demand forecasting
- Machine learning
- RAG for project knowledge
- Tool-calling conversational AI
- Streamlit user interface

---

## 1. Project Goal

The goal was not only to build a model.

The goal was to create an end-to-end analytics product that follows the full data lifecycle:

```text
Ingestion
   ↓
Validation
   ↓
Transformation
   ↓
Modeling
   ↓
Semantic Layer
   ↓
Forecasting
   ↓
RAG + Tool Calling
   ↓
Conversational Interface
```

WareIQ is designed for non-technical users who need clear business answers without learning SQL, machine learning, or statistics.

---

## 2. Dataset

- Source: Montgomery County, Maryland government warehouse and retail sales data
- Size: 307K+ transactions
- Historical period: 2017–2020
- Categories:
  - Beer
  - Wine
  - Liquor
  - Kegs
  - Non-Alcohol
- Main business dimensions:
  - Supplier
  - Item type
  - Item description
  - Month
  - Warehouse sales
  - Retail sales

Important limitation:

- The dataset contains a major 2018 coverage gap
- The app does not contain live warehouse data
- “Latest” means the latest available record in the dataset

---

## 3. Key Business Findings

### Data quality

- Detected a major 2018 data gap
- Removed a suspicious USD 7,800 transaction before trusted analysis
- Fixed three leakage risks before final modeling

### Supplier performance

- Crown Imports was the highest-selling supplier by warehouse sales
- Supplier performance is ranked by sales only
- Margin, delivery reliability, lead time, quality, and return rates are not available

### Seasonality

- Summer was defined as May through August
- Beer and Liquor showed meaningful seasonal behavior
- Wine and Kegs did not show the same confirmed seasonal pattern
- Estimated annual summer revenue opportunity: about USD 271K

### Forecasting

- Built a six-month warehouse-sales forecast
- Forecasts include:
  - expected value
  - lower range
  - upper range
- Forecasts are treated as estimates, not guarantees

### Machine learning

- Logistic Regression was used as a baseline
- XGBoost improved classification performance
- Feature importance helped translate model behavior into business insight

---

## 4. Project Architecture

### Bronze Layer

Purpose:

- Preserve the raw source
- Add ingestion timestamp
- Add source-file lineage
- Create reproducible raw Parquet outputs

File:

```text
src/ingest.py
```

Output:

```text
data/raw/bronze_<timestamp>.parquet
```

---

### Silver Layer

Purpose:

- Standardize data types
- Clean supplier names
- Standardize item descriptions
- Standardize item categories
- Handle missing values
- Remove suspicious records
- Preserve transformation logic

File:

```text
src/clean.py
```

Output:

```text
data/cleaned/silver_<timestamp>.parquet
```

---

### Gold Layer

Purpose:

- Create business-ready data
- Prepare clean warehouse-sales records
- Support analytics, forecasting, and modeling

Files:

```text
src/clean.py
src/features.py
```

Output:

```text
data/processed/gold_<timestamp>.parquet
```

---

### Feature Layer

Purpose:

- Create model-ready variables
- Encode suppliers and item types
- Add calendar features
- Add lag features
- Add rolling sales features
- Build the target variable

Main features:

```text
MONTH
IS_SUMMER
IS_HOLIDAY
SUPPLIER_ENCODED
ITEM_TYPE_ENCODED
LAST_MONTH_SALES
LAST_3_MONTHS_SALES
```

File:

```text
src/features.py
```

---

### Analytics Layer

Purpose:

- Provide exact business metrics
- Use DuckDB for fast analytics
- Keep chatbot answers grounded in current query results
- Prevent the language model from inventing numbers

File:

```text
src/analysis_tools.py
```

Example functions:

```python
get_total_sales()
get_top_suppliers()
get_low_performing_suppliers()
get_mid_performing_suppliers()
get_suppliers_by_category()
get_sales_by_month()
get_sales_by_item_type()
get_items_ranked()
generate_summary()
```

---

### Modeling Layer

Purpose:

- Compare a simple baseline with a stronger model
- Predict high-sales observations
- Save trained model artifacts
- Track model performance

Models:

- Logistic Regression
- XGBoost

File:

```text
src/train.py
```

Outputs:

```text
models/logistic_regression.pkl
models/xgboost.pkl
models/metrics.json
models/xgb_feature_importance.png
```

---

### Forecasting Layer

Purpose:

- Estimate the next six months of warehouse sales
- Show a prediction range
- Communicate uncertainty clearly

Model:

- Prophet

File:

```text
src/forecast.py
```

Forecast outputs:

```text
ds
yhat
yhat_lower
yhat_upper
```

---

### Experimentation Layer

Purpose:

- Test whether summer sales are meaningfully different
- Avoid recommendations based only on assumptions
- Compare seasonal behavior by item type

Analysis levels:

1. Transaction-level comparison
2. Controlled analysis
3. Monthly aggregate analysis
4. Item-level segmentation
5. Effect-size calculation

File:

```text
src/ab_testing.py
```

Output:

```text
models/ab_results.json
```

---

### RAG Layer

Purpose:

- Store project knowledge
- Store definitions
- Store data limitations
- Store documentation
- Store methodology
- Preserve lessons and findings that are not simple SQL results

RAG is not the primary source for changing business numbers.

Data-source priority:

```text
1. Live function output
2. Current model or JSON output
3. RAG documentation
4. Never model memory
```

File:

```text
src/rag.py
```

RAG stack:

- Sentence Transformers
- ChromaDB
- Recursive text splitting
- Source-signature refresh logic

---

### Agent Layer

Purpose:

- Understand the user’s question
- Select the correct analytics function
- Read the latest returned values
- Explain the result in simple language
- Preserve conversation context

File:

```text
src/agent.py
```

Agent behavior:

```text
User question
   ↓
Intent understanding
   ↓
Tool selection
   ↓
DuckDB / forecast / experiment function
   ↓
Latest result
   ↓
Simple business explanation
```

Important rule:

The assistant must never answer changing numerical questions from memory.

---

### Experience Layer

Purpose:

- Provide a clean, conversational interface
- Explain analytics to general users
- Support Past, Latest, and Future views
- Show project proof and lessons

File:

```text
src/app.py
```

Main views:

- Assistant
- Visual Story
- Project Proof

---

## 5. Repository Structure

```text
food_intelligence/
│
├── data/
│   ├── raw/
│   ├── cleaned/
│   ├── processed/
│   └── chroma_db/
│
├── docs/
│   ├── data_dictionary.md
│   └── project_log.md
│
├── models/
│   ├── metrics.json
│   ├── ab_results.json
│   ├── logistic_regression.pkl
│   ├── xgboost.pkl
│   └── forecast.png
│
├── src/
│   ├── ingest.py
│   ├── clean.py
│   ├── features.py
│   ├── train.py
│   ├── forecast.py
│   ├── ab_testing.py
│   ├── analysis_tools.py
│   ├── rag.py
│   ├── agent.py
│   └── app.py
│
├── .env
├── requirements.txt
└── README.md
```

---

## 6. Installation

### Step 1 — Clone the repository

```bash
git clone <your-repository-url>
cd food_intelligence
```

---

### Step 2 — Create a virtual environment

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

---

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

Core packages include:

```text
pandas
numpy
duckdb
pyarrow
scikit-learn
xgboost
prophet
scipy
plotly
streamlit
groq
python-dotenv
chromadb
sentence-transformers
langchain
```

---

### Step 4 — Add the Groq API key

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
```

Recommended model:

```python
MODEL = "llama-3.1-8b-instant"
```

This model is sufficient because the actual calculations are handled by DuckDB, forecasting code, and statistical functions.

---

## 7. Run the Pipeline Step by Step

Run all commands from the project root.

### Step 1 — Add the raw CSV

Place the source CSV here:

```text
data/raw/Warehouse_and_Retail_Sales.csv
```

---

### Step 2 — Run ingestion

```bash
python3 src/ingest.py
```

Expected result:

```text
data/raw/bronze_<timestamp>.parquet
```

This step:

- loads the CSV
- profiles the dataset
- adds ingestion metadata
- preserves the original source

---

### Step 3 — Run cleaning

```bash
python3 src/clean.py
```

Expected result:

```text
data/cleaned/silver_<timestamp>.parquet
data/processed/gold_<timestamp>.parquet
```

This step:

- fixes data types
- creates the date column
- standardizes text
- handles missing supplier values
- removes the suspicious record
- prepares the clean business dataset

---

### Step 4 — Build features

```bash
python3 src/features.py
```

Expected result:

```text
data/processed/featured_train_<timestamp>.parquet
data/processed/featured_test_<timestamp>.parquet
```

This step:

- creates the target
- encodes suppliers
- encodes item types
- adds seasonality indicators
- adds lag features
- adds rolling sales features
- splits train and test data

---

### Step 5 — Train models

```bash
python3 src/train.py
```

Expected result:

```text
models/logistic_regression.pkl
models/xgboost.pkl
models/metrics.json
```

This step:

- trains Logistic Regression
- trains XGBoost
- compares model performance
- saves model artifacts
- saves evaluation metrics

---

### Step 6 — Run forecasting

```bash
python3 src/forecast.py
```

Expected result:

```text
models/forecast.png
```

This step:

- aggregates sales monthly
- trains Prophet
- forecasts six months
- generates lower and upper bounds
- creates a planning recommendation

---

### Step 7 — Run the seasonal experiment

```bash
python3 src/ab_testing.py
```

Expected result:

```text
models/ab_results.json
```

This step:

- compares summer and non-summer sales
- runs statistical tests
- calculates effect size
- evaluates seasonality by category
- estimates annual business opportunity

---

### Step 8 — Build or refresh RAG

```bash
python3 src/rag.py
```

This step:

- reads project documents
- generates current findings
- creates embeddings
- stores them in ChromaDB
- refreshes automatically when source files change

Expected result:

```text
data/chroma_db/
data/chroma_db/rag_manifest.json
```

---

### Step 9 — Test the agent

```bash
python3 src/agent.py
```

Example questions:

```text
Which supplier sold the most?
Which month was busiest?
What changed during summer?
What may happen in the next six months?
Which categories are seasonal?
What data quality problems were found?
```

---

### Step 10 — Launch the app

```bash
streamlit run src/app.py
```

Open the local URL shown in the terminal.

Usually:

```text
http://localhost:8501
```

---

## 8. Chatbot Behavior

The chatbot is designed for general users.

### Casual conversation

Examples:

```text
Hi
How are you?
Thank you
What can you do?
```

These should receive friendly responses without calling analytics functions.

---

### Analytical questions

For every numerical question, the agent must:

1. identify the intent
2. call the correct function
3. read the newest returned values
4. answer only from those values
5. explain the result in simple language

Example:

```text
User:
Which supplier sold the most?

Internal behavior:
get_top_suppliers()

Visible answer:
Crown Imports was the highest-selling supplier in the available data.

It generated about USD 1.29 million in warehouse sales and contributed roughly 20% of the total.
```

The user never sees:

```text
Tool call
Function name
SQL
DuckDB
RAG
ChromaDB
Internal reasoning
```

---

## 9. Past, Latest, and Future Logic

### Past

Uses observed historical data from 2017–2020.

Example:

```text
Which month was busiest historically?
```

---

### Latest

Uses the latest available record in the dataset.

Important:

```text
Latest does not mean live.
The dataset ends in 2020.
```

---

### Future

Reruns the forecast function and uses the latest generated values.

Future answers must include:

- forecast period
- expected value
- lower range
- upper range
- uncertainty explanation
- business limitation

---

## 10. Why DuckDB and RAG Are Both Used

### DuckDB

Used for:

- totals
- rankings
- comparisons
- monthly sales
- category sales
- supplier performance
- exact changing values

### RAG

Used for:

- documentation
- project notes
- data definitions
- methodology
- quality findings
- limitations
- lessons learned

Architecture:

```text
DuckDB = current analytical truth
RAG = project memory
LLM = question routing and explanation
```

---

## 11. What I Learned

### Data before model

I originally thought XGBoost would be the hardest and most important part.

It was not.

The biggest improvements came from understanding the dataset, finding quality issues, and defining reliable business metrics before modeling.

---

### Cleaning is analysis

The 2018 data gap and suspicious USD 7,800 transaction showed that cleaning was not a preprocessing checkbox.

Validation changed what could be trusted.

---

### Baselines matter

Logistic Regression created a reference point.

Without a baseline, higher accuracy has no context.

---

### Prove before recommending

Summer looked stronger, but the project did not stop at intuition.

The seasonal recommendation was supported by statistical comparison, effect size, and category-level segmentation.

---

### Forecasts are not promises

The first instinct was to focus on the predicted value.

The better approach was to communicate a range and make uncertainty visible.

---

### Design for the real user

The first dashboard was too technical.

The final design focuses on:

- simple language
- conversational answers
- clear business labels
- visual proof
- technical details only when requested

---

### RAG should not store changing truth

Hardcoded numbers become stale.

The final design uses live functions for business values and RAG for documentation and project memory.

---

## 12. Known Limitations

- Historical dataset ends in 2020
- 2018 has a major data gap
- No live inventory data
- No supplier lead-time data
- No product margin data
- No service-level data
- No shelf-life data
- No minimum-order-quantity data
- Forecasts are based only on historical sales patterns
- Sales alone cannot determine whether a supplier is truly “best” or “worst”

---

## 13. Future Improvements

- Connect a live warehouse database
- Add current inventory levels
- Add supplier lead times
- Add margin and profitability
- Add drift monitoring
- Add scheduled model refreshes
- Add role-based access
- Add downloadable reports
- Add chart generation inside chat
- Add user feedback and answer approval
- Add evaluation tests for tool routing
- Add deployment monitoring

---

## 14. Portfolio Value

WareIQ demonstrates:

- end-to-end data lifecycle ownership
- data quality validation
- medallion architecture
- DuckDB analytics
- reusable semantic functions
- baseline modeling
- XGBoost classification
- Prophet forecasting
- statistical experimentation
- RAG
- tool-calling agents
- conversational UX
- plain-language business communication

The most important result is not the chatbot or the model.

It is the ability to move from raw data to a trusted answer that a non-technical user can understand.

---

## 15. Run Everything

From the project root:

```bash
python3 src/ingest.py
python3 src/clean.py
python3 src/features.py
python3 src/train.py
python3 src/forecast.py
python3 src/ab_testing.py
python3 src/rag.py
streamlit run src/app.py
```

---

## 16. Final Principle

```text
Do not start with the model.

Start with the data.
Validate what can be trusted.
Build reusable business definitions.
Prove patterns before recommending action.
Use models to reduce uncertainty.
Explain the result so someone can use it.
```
