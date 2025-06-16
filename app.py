import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
import joblib

# Animated and enhanced CSS styling
st.markdown(
    """
    <style>
    /* Fade in animation for the app container */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px);}
        to { opacity: 1; transform: translateY(0);}
    }

    .reportview-container, .main {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #222222;
        background: linear-gradient(135deg, #f0f4f8, #d9e2ec);
        padding: 30px 60px;
        animation: fadeIn 1s ease forwards;
        min-height: 100vh;
    }

    /* Title styling with zoom-in */
    .css-1v3fvcr h1 {
        color: #2c3e50;
        font-weight: 700;
        font-size: 3.5rem;
        margin-bottom: 25px;
        text-align: center;
        animation: zoomIn 0.8s ease forwards;
    }

    @keyframes zoomIn {
        from { transform: scale(0.8); opacity: 0;}
        to { transform: scale(1); opacity: 1;}
    }

    /* Input boxes styling with shadow and focus glow */
    div.stTextInput > div > input, div.stNumberInput > div > input {
        font-size: 18px;
        padding: 14px 18px;
        border: 2.5px solid #4CAF50;
        border-radius: 12px;
        outline: none;
        transition: border-color 0.4s ease, box-shadow 0.4s ease;
        width: 100%;
        max-width: 420px;
        display: block;
        margin: 12px auto 24px auto;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
        background-color: #fff;
        font-weight: 600;
        color: #333;
    }
    div.stTextInput > div > input:focus, div.stNumberInput > div > input:focus {
        border-color: #27ae60;
        box-shadow: 0 0 12px #27ae60;
    }

    /* Button styling with hover and active animations */
    div.stButton > button {
        background: linear-gradient(45deg, #27ae60, #2ecc71);
        color: white;
        font-size: 19px;
        font-weight: 700;
        padding: 16px 48px;
        border-radius: 14px;
        border: none;
        cursor: pointer;
        display: block;
        margin: 0 auto 40px auto;
        transition: background 0.4s ease, transform 0.2s ease, box-shadow 0.3s ease;
        box-shadow: 0 6px 15px rgba(39, 174, 96, 0.5);
        user-select: none;
    }
    div.stButton > button:hover {
        background: linear-gradient(45deg, #2ecc71, #27ae60);
        box-shadow: 0 8px 20px rgba(46, 204, 113, 0.7);
        transform: translateY(-3px);
    }
    div.stButton > button:active {
        transform: translateY(1px);
        box-shadow: 0 4px 10px rgba(39, 174, 96, 0.5);
    }

    /* Result text styling with fade-in */
    .css-1v3fvcr h2, .css-1v3fvcr p {
        animation: fadeIn 1s ease forwards;
        text-align: center;
        color: #34495e;
        font-weight: 700;
    }
    .css-1v3fvcr h2 {
        font-size: 2.2rem;
        margin-bottom: 12px;
    }
    .css-1v3fvcr p {
        font-size: 20px;
        margin-top: 4px;
        color: #555;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load model and encoder
model = joblib.load("football_model.pkl")
le = joblib.load("label_encoder.pkl")

teams = ['Arsenal', 'Aston Villa', 'Bournemouth', 'Brentford', 'Brighton Hove',
         'Burnley', 'Chelsea', 'Crystal Palace', 'Everton', 'Fulham', 'Liverpool',
         'Luton Town', 'Man City', 'Man United', 'Newcastle', 'Nottingham',
         'Sheffield Utd', 'Tottenham', 'West Ham', 'Wolverhampton']

def predict_future_match(home_team_name, away_team_name, matchday):
    try:
        home_encoded = le.transform([home_team_name])[0]
        away_encoded = le.transform([away_team_name])[0]
    except:
        return "Error: Team not found", ""

    input_features = pd.DataFrame({
        'matchday': [matchday],
        'home_team_encoded': [home_encoded],
        'away_team_encoded': [away_encoded]
    })

    predicted_scores = model.predict(input_features)

    home_score = round(predicted_scores[0][0])
    away_score = round(predicted_scores[0][1])

    result = f"{home_team_name} {home_score} - {away_score} {away_team_name}"
    if home_score > away_score:
        winner = f"Predicted Winner: {home_team_name}"
    elif away_score > home_score:
        winner = f"Predicted Winner: {away_team_name}"
    else:
        winner = "Predicted Result: Draw"

    return result, winner

st.title("Football Score Predictor")

home_team = st.selectbox("Select Home Team", teams)
away_team = st.selectbox("Select Away Team", teams)
matchday = st.number_input("Enter Matchday", min_value=1, step=1)

if st.button("Predict"):
    scoreline, winner = predict_future_match(home_team, away_team, matchday)
    st.subheader(scoreline)
    st.write(winner)
