# MLOPS-2026 Mini Project 1

A machine learning project for house price prediction using linear regression.

## Features

- **Model Training**: Train a linear regression model on house data
- **FastAPI Web Service**: REST API for house price predictions
- **Streamlit Web App**: Interactive web interface for predictions
- **Docker Support**: Containerized deployment

## Project Structure

```
├── src/
│   ├── train_model.py    # Model training script
│   ├── model_api.py      # FastAPI web service
│   ├── model_app.py      # Streamlit web application
│   └── quantization_II.ipynb  # Jupyter notebook
├── data/
│   └── houses.csv        # Training data
├── deploy/
│   └── docker-compose.yml  # Docker deployment
├── main.py               # Main application entry point
├── pyproject.toml        # Project dependencies
└── Dockerfile            # Container configuration
```

## Requirements

- Python >=3.13
- Dependencies listed in `pyproject.toml`

## Installation

```bash
# Install dependencies
uv sync

# Or with pip
pip install -e .
```

## Usage

### Train the Model
```bash
python src/train_model.py
```

### Run FastAPI Service
```bash
uvicorn src.model_api:app --reload
```

### Run Streamlit App
```bash
streamlit run src/model_app.py
```

### Run Main Application
```bash
python main.py
```

## Docker Deployment

```bash
cd deploy
docker-compose up
```

## Development

This project uses:
- **uv** for dependency management
- **FastAPI** for web services
- **Streamlit** for web interfaces
- **scikit-learn** for machine learning
- **Docker** for containerization
