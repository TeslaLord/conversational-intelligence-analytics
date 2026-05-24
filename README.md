# Contact-Centre Insights Bot

## Setup

### Option 1: Prebuilt Docker image (analytics search only)

To skip data ingestion and just run the analytics chatbot against the pre-built index:

```bash
docker run --rm -p 8501:8501 \
  -e LLM_API_KEY=sk-proj-... \
  teslalord/cc-insights:latest
```

Then open <http://localhost:8501>.

### Option 2: Full application (ingestion + chatbot + eval)

```bash
git clone https://github.com/TeslaLord/conversational-intelligence-analytics.git
cd conversational-intelligence-analytics

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# fill in your OpenAI API key
# (optional for prototype) in the env file also replace the salt for security PII_HASH_SALT
```

Drop the dataset in the project root as `customer_support_data.csv`.

## Run

```bash
# Build the pipeline on a small batch of 100 conversations
python -m cc_insights.pipeline --batch-size 100

# Start the Streamlit chatbot
streamlit run cc_insights/ui.py

# Evaluate against the golden dataset (writes ./data/eval_results.json)
python -m eval.run_eval
```

Repository: <https://github.com/TeslaLord/conversational-intelligence-analytics>
