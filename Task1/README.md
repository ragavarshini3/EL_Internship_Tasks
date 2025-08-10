# EL_TASK_1
Data cleaning and preprocessing of Netflix Movies and TV Shows dataset using Python (Google Colab).
# Task 1: Data Cleaning and Preprocessing - Netflix Dataset

## Dataset Used
Netflix Movies and TV Shows  
Source: [Kaggle Netflix Dataset](https://www.kaggle.com/datasets/shivamb/netflix-shows)

## Tools
Google Colab (Python, Pandas)

## Cleaning Summary:
- Dropped duplicate records
- Handled missing values:
  - Filled 'director', 'cast', 'country' with 'Unknown'
  - Filled 'date_added' and 'rating' with mode
- Standardized text and column names
- Converted 'date_added' to datetime and extracted year/month

## Output
- Cleaned Dataset: `cleaned_netflix_data.csv`

## Learnings
Gained hands-on experience with missing value imputation, standardization, and datetime handling using Pandas.
