# SMARTA

### AI-Powered Industrial Technical Assistant

## About the Project

SMARTA is an AI-powered industrial maintenance assistant that helps engineers troubleshoot machine problems using information from MySQL databases, Excel spreadsheets, and PDF technical manuals.

It uses Retrieval-Augmented Generation (RAG) to retrieve relevant maintenance information and generate helpful troubleshooting responses using a locally running AI model.

The project also proposes a sensor-simulation feature that generates sample temperature and vibration readings, detects threshold violations, and combines alerts with technical guidance.

## Key Features

- **Unified Knowledge Base**: Retrieves maintenance information from databases, spreadsheets, and PDF manuals.
- **Semantic Search**: Finds relevant information based on the meaning of a question.
- **AI Troubleshooting**: Generates responses containing possible causes and recommended checks.
- **Local AI Processing**: Runs the language model using Ollama.
- **Sensor Simulation**: Proposed extension for generating synthetic machine readings.
- **Threshold Alerts**: Proposed extension for identifying abnormal simulated readings.

## Technology Stack

- **Frontend**: Flutter and Dart
- **Backend**: Python and FastAPI
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Search**: FAISS
- **Language Model**: Qwen2.5 3B
- **Local Inference**: Ollama
- **Data Sources**: MySQL, Excel, and PDF
