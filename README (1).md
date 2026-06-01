# Support Ticket Classification System

An AI-powered system to automatically classify customer support tickets into categories and assign priority levels. This helps support teams respond faster and more efficiently.

## Features

- **Text Preprocessing**: Cleans raw ticket text (lowercasing, stopword removal, lemmatization).
- **Category Classification**: Classifies tickets into:
  - Billing inquiry
  - Technical issue
  - Refund request
  - Cancellation request
  - Product inquiry
- **Priority Tagging**: Assigns priority levels:
  - Critical
  - High
  - Medium
  - Low
- **Performance Evaluation**: Generates accuracy metrics and confusion matrices.

## Installation

1. Ensure you have Python installed.
2. Install dependencies:
   ```bash
   pip install pandas scikit-learn nltk joblib matplotlib seaborn
   ```

## Usage

### 1. Training the Models

If you want to retrain the models on the dataset:

```bash
python train.py
```

### 2. Testing with Custom Tickets

Use the interactive classifier:

```bash
python app.py
```

### 3. Running Evaluation

To generate performance metrics and confusion matrices:

```bash
python evaluate.py
```

## Project Structure

- `app.py`: Main CLI application for classification.
- `train.py`: Training script for the ML models.
- `preprocess.py`: NLP preprocessing utilities.
- `evaluate.py`: Model evaluation and visualization.
- `eda.py`: Initial exploratory data analysis.
- `type_model.joblib`: Trained model for ticket type.
- `priority_model.joblib`: Trained model for ticket priority.
