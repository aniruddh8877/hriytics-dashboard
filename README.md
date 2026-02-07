# HRlytics: Human Resources Analytics

This repository contains tools and analyses for understanding HR data, predicting employee attrition, and visualizing key metrics.

## Project Structure

- `data/`: Raw and processed data (e.g., `hr_data.csv`).
- `notebooks/`: Jupyter notebooks for exploratory data analysis (EDA).
- `src/`: Source code for data processing, model training, and evaluation.
- `dashboard/`: Streamlit dashboard application (`app.py`).

## Getting Started

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/HRlytics.git
    cd HRlytics
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Streamlit Dashboard**:
    ```bash
    streamlit run dashboard/app.py
    ```

4.  **Run Model Training**:
    ```bash
    python src/model_training.py
    ```

## Dashboard Features

-   **Key Metrics**: Total employees, average salary, attrition rate.
-   **Visualizations**: Attrition by department, salary distribution, correlation heatmap.
-   **Interactive Filters**: Filter data by department.

## Data Source

The dataset simulates HR records including demographics, job role, satisfaction levels, and attrition status.
