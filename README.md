# Algorithmic Trading Strategy for Bitcoin

## Project Overview
This repository contains the implementation of an algorithmic trading strategy specifically tailored for Bitcoin, as part of the APS 1052: Artificial Intelligence in Finance. The project leverages historical data, machine learning, and sentiment analysis to design, test, and evaluate trading strategies aimed at maximizing returns and minimizing risks in the cryptocurrency market.

## Repository Structure
- `webscrapper_bitcoinfocharts.ipynb`: Script for scraping Bitcoin-related metrics.
- `bitcoin_data_bitcoininfocharts.csv`: Bitcoin metrics data.
- `webscrapper_finviz.ipynb`: Script for scraping News headlines from finviz.com.
- `webscrapper_googlenews.ipynb`: Script for scraping Google News headlines for sentiment analysis.
- `bitcoin_news_data.csv`: News headlines data.
- `bitcoin_news_data_sentiment_vader.csv`: Sentiment analysis results using VADER.
- `news_sentiment_analysis_bert.ipynb`: Script for performing sentiment analysis using a BERT transformer.
- `bitcoin_news_data_sentiment_finbert.csv`: BERT-based sentiment analysis results.
- `news_btc_correlation.ipynb`: Script for analyzing the correlation between market returns and news sentiment.
- `btc_usd_sentiment.csv`: Dataset for market returns and news sentiment correlation.
- `bitcoin_tsa.ipynb`: Time Series Analysis for Bitcoin.
- `bitcoin_feature_engineering.ipynb`: Feature engineering scripts for Bitcoin trading data.
- `bitcoin_techindicators_filtered_normalized.csv`: Processed dataset with technical indicators.
- `main.ipynb`: Main script for model evaluation and tuning.
- `main.csv`: Main dataset used for training and evaluations.


## Workflow
1. **Data Collection**: Collecting data from various sources including historical Bitcoin transactions, news headlines, and trading indicators.
2. **Data Preprocessing**: Cleaning and preparing data for analysis.
3. **Exploratory Data Analysis (EDA)**: Analyzing data to uncover trends and patterns.
4. **Feature Engineering**: Creating new features to improve model performance.
5. **Model Tuning and Evaluation**: Adjusting model parameters and evaluating their performance.
6. **Sentiment Analysis**: Analyzing news headlines to gauge market sentiment.
7. **Correlation Analysis**: Exploring the relationship between Bitcoin price movements and news sentiment.

## Key Results
- Developed a robust trading strategy that outperforms a simple buy-and-hold strategy.
- Implemented various models with systematic hyperparameter tuning and model validation.
- Demonstrated the importance of sentiment analysis in predicting market movements.

## Installation
To run the notebooks, you need to install the required libraries. THe following command will create a new conda environment with all the required libraries:
```bash
 conda env create -f environment.yml

