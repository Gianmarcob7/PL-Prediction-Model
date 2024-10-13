from flask import Flask, request, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
import requests

app = Flask(__name__)

# base_url API key
api_key = "8a2efd035f2e63715e9c60a2cfe33bc6"
base_url = "https://v3.football.api-sports.io"

# historical match data
def get_historical_data(league_id, season):
    url = f"{base_url}/fixtures"
    headers = {"x-rapidapi-key": api_key}
    params = {"league": league_id, "season": season}
    response = requests.get(url, headers=headers, params=params)
    return response.json()

# list of teams
def get_teams(league_id):
    url = f"{base_url}/teams"
    headers = {"x-rapidapi-key": api_key}
    params = {"league": league_id, "season": season}
    response = requests.get(url, headers=headers, params=params)
    data = response.json()
    teams = data['response']
    team_names = [team['team']['name'] for team in teams]
    return team_names

# prepare the data
def prepare_data(data):
    fixtures = data.get('response', [])
    df = pd.json_normalize(fixtures)
    df = df[['teams.home.name', 'teams.away.name', 'score.fulltime.home', 'score.fulltime.away']]
    df.columns = ['home_team', 'away_team', 'fulltime_home', 'fulltime_away']

    # team names
    encoder = OneHotEncoder()
    teams_encoded = encoder.fit_transform(df[['home_team', 'away_team']])
    teams_encoded_df = pd.DataFrame(teams_encoded.toarray(), columns=encoder.get_feature_names_out(['home_team', 'away_team']))

    # combine the columns with the rest of the data
    df = pd.concat([teams_encoded_df, df[['fulltime_home', 'fulltime_away']]], axis=1)

    features = df.drop(columns=['fulltime_home'])
    target = df['fulltime_home'] > df['fulltime_away']  # 1 if home team wins, 0 otherwise

    return features, target, encoder

# train the model
def train_model(features, target):
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy}")
    return model

# predict the outcome of a match
def predict_match(model, home_team, away_team, encoder):
    team_data = pd.DataFrame([[home_team, away_team]], columns=['home_team', 'away_team'])
    teams_encoded = encoder.transform(team_data)
    teams_encoded_df = pd.DataFrame(teams_encoded.toarray(), columns=encoder.get_feature_names_out(['home_team', 'away_team']))
    example_data = pd.concat([teams_encoded_df, pd.DataFrame([[1]], columns=['fulltime_away'])], axis=1)  # Using placeholder for fulltime_away
    prediction = model.predict(example_data)
    return prediction

# initialize model and teams list once
league_id = 39
season = 2022
teams = get_teams(league_id)
historical_data = get_historical_data(league_id, season)
features, target, encoder = prepare_data(historical_data)
model = train_model(features, target)

@app.route('/')
def home():
    return render_template('home.html', teams=teams)

@app.route('/predict', methods=['POST'])
def predict():
    home_team = request.form['home_team']
    away_team = request.form['away_team']
    prediction = predict_match(model, home_team, away_team, encoder)
    result = 'Home team wins' if prediction[0] else 'Home team loses or draws'
    return render_template('result.html', prediction=result, home_team=home_team, away_team=away_team)

if __name__ == '__main__':
    app.run(debug=True)
