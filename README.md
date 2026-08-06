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
