# PL-Prediction-Model

## Overview
This is a personal project aimed at predicting the outcomes of Premier League matches based on historical data from the 2022 season. The model utilizes logistic regression to analyze match statistics and generate predictions for home team victories.

## Project Features
- **Data Collection**: I pull historical match data from the API Sports football API, which provides comprehensive information about matches.
- **Data Preparation**: The data is processed and cleaned to create a structured dataset that can be used for training the model.
- **Model Training**: I trained a logistic regression model on the prepared dataset, allowing it to learn from past outcomes and make predictions.
- **User Interface**: A simple and intuitive web interface built with Flask enables users to select teams and view predictions.

## Technologies Used
- **Python**: Used for developing the model and web app.
- **Flask**: The web framework that powers the application.
- **Pandas**: Essential for data manipulation and analysis.
- **Scikit-Learn**: The machine learning library that includes the logistic regression model.
- **Requests**: Used for making API calls to fetch historical match data.
