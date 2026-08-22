# Healthcare Claims Automation MVP

An end-to-end machine-learning pipeline built to predict healthcare claim
denials, streamline billing workflows, and flag risk areas using a Streamlit
dashboard.

## Project overview

- Phase 1 (Data): SQL data extraction and preprocessing pipeline.
- Phase 2 (Modeling): XGBoost classifier for denial-risk prediction.
- Phase 3 (API and Dashboard): FastAPI service and Streamlit interface.
- Phase 4 (CI/CD): Docker packaging and GitHub Actions automation.

## Repository structure

```text
arenaflex-claims-automation/
|-- .github/workflows/  - CI/CD automation
|-- data/                - Local claims data
|-- models/              - Trained model artifacts
|-- src/                 - Extraction, training, API, and dashboard code
|-- tests/               - API and model tests
|-- Dockerfile           - Container instructions
|-- requirements.txt     - Python dependencies
`-- README.md            - Project documentation
```

## Quickstart guide

1. Clone the repository:

   ```bash
   git clone https://github.com/UbokobongEmmanuel/arenaflex-claims-automation.git
   cd arenaflex-claims-automation
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the training script:

   ```bash
   python -m src.models.train
   ```

4. Start the prediction API:

   ```bash
   uvicorn src.app.main:app --reload
   ```

   API documentation is available at `http://localhost:8000/docs`.

5. In a second terminal, start the operations dashboard:

   ```bash
   streamlit run src/dashboard/app.py
   ```

## Prediction endpoint

Send a claim to `POST /predict`:

```json
{
  "charge_amount": 1250.0,
  "code_count": 2,
  "has_modifier_25": 1
}
```

The API requires `models/xgboost_denial_v1.joblib` and
`models/scaler_v1.joblib`. Run the training pipeline before requesting a
prediction.

## Docker

Build and run the API container:

```bash
docker build -t arenaflex-claims-api .
docker run --rm -p 8000:8000 arenaflex-claims-api
```
