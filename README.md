# Delmia Apriso AI

This project is a custom AI model trained on content related to Delmia Apriso, built using Python, Transformers, and FastAPI.

## Setup Instructions

1. **Clone the repository:**
    ```bash
    git clone <repository-url>
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Run data collection:**
    ```bash
    python src/data_collection.py
    ```

4. **Preprocess the data:**
    ```bash
    python src/preprocessing.py
    ```

5. **Train the model:**
    ```bash
    python src/model_training.py
    ```

6. **Evaluate the model:**
    ```bash
    python src/evaluation.py
    ```

7. **Run the API:**
    ```bash
    uvicorn src.api:app --reload
    ```

## Usage

- **API Endpoint:** `/generate`
- **Request Format:**
    ```json
    {
        "prompt": "What is Delmia Apriso?"
    }
    ```
- **Response Format:**
    ```json
    {
        "response": "Delmia Apriso is..."
    }
    ```

## Folder Structure

