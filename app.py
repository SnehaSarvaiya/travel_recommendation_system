from click import prompt
from flask import Flask, render_template,request
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import numpy as np
from itinerary import generate_itinerary
from pathlib import Path
from flask import session
# import http.client

# app
app = Flask(__name__)
app.secret_key = "travel_ai_secret"
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
# Debug prints (can remove later)
print("BASE_DIR:", BASE_DIR)
print("DATA_DIR:", DATA_DIR)
print("DATA_DIR exists:", DATA_DIR.exists())
print("FILES:", list(DATA_DIR.glob("*")))
# model_path = DATA_DIR/ "model.pkl"
# label_encoders_path = DATA_DIR / "label_encoders.pkl"
# destinations_path = DATA_DIR / "Expanded_Destinations.csv"
# userhistory_path = DATA_DIR / "Final_Updated_Expanded_UserHistory.csv"
# df_path = DATA_DIR / "final_df.csv"
# print("BASE_DIR:", BASE_DIR)
# print("DATA_DIR:", DATA_DIR)
# print("DATA FILES:", list(DATA_DIR.glob("*")))
# File paths
# -----------------------------
model_path = DATA_DIR / "model.pkl"
label_encoders_path = DATA_DIR / "label_encoders.pkl"
destinations_path = DATA_DIR / "Expanded_Destinations.csv"
userhistory_path = DATA_DIR / "Final_Updated_Expanded_UserHistory.csv"
df_path = DATA_DIR / "final_df.csv"


# sanity checks
for p in (model_path, label_encoders_path, destinations_path, userhistory_path, df_path):
    if not p.exists():
        raise FileNotFoundError(f"Required file not found: {p}")

# Load datasets and models
features = ['Name_x', 'State', 'Type', 'BestTimeToVisit', 'Preferences', 'Gender', 'NumberOfAdults', 'NumberOfChildren']
# model = pickle.load(open(DATA_DIR / "model.pkl",'rb'))
# label_encoders = pickle.load(open(DATA_DIR / "label_encoders.pkl",'rb'))

# destinations_df = pd.read_csv(DATA_DIR / "Expanded_Destinations.csv")
# userhistory_df = pd.read_csv(DATA_DIR / "Final_Updated_Expanded_UserHistory.csv")
# df = pd.read_csv(DATA_DIR / "final_df.csv")
model = pickle.load(open(model_path, "rb"))
label_encoders = pickle.load(open(label_encoders_path, "rb"))
destinations_df = pd.read_csv(destinations_path)
userhistory_df = pd.read_csv(userhistory_path)
df = pd.read_csv(df_path)

# Collaborative Filtering Function
# Create a user-item matrix based on user history
user_item_matrix = userhistory_df.pivot(index='UserID', columns='DestinationID', values='ExperienceRating')

# Fill missing values with 0 (indicating no rating/experience)
user_item_matrix.fillna(0, inplace=True)

# Compute cosine similarity between users
user_similarity = cosine_similarity(user_item_matrix)


# Function to recommend destinations based on user similarity
def collaborative_recommend(user_id, user_similarity, user_item_matrix, destinations_df):
    """
    Recommends destinations based on collaborative filtering.

    Args:
    - user_id: ID of the user for whom recommendations are to be made.
    - user_similarity: Cosine similarity matrix for users.
    - user_item_matrix: User-item interaction matrix (e.g., ratings or preferences).
    - destinations_df: DataFrame containing destination details.

    Returns:
    - DataFrame with recommended destinations and their details.
    """
    # Find similar users
    similar_users = user_similarity[user_id - 1]

    # Get the top 5 most similar users
    similar_users_idx = np.argsort(similar_users)[::-1][1:6]

    # Get the destinations liked by similar users
    similar_user_ratings = user_item_matrix.iloc[similar_users_idx].mean(axis=0)

    # Recommend the top 5 destinations
    recommended_destinations_ids = similar_user_ratings.sort_values(ascending=False).head(5).index

    # Filter the destinations DataFrame to include detailed information
    recommendations = destinations_df[destinations_df['DestinationID'].isin(recommended_destinations_ids)][[
        'DestinationID', 'Name', 'State', 'Type', 'Popularity', 'BestTimeToVisit'
    ]]

    return recommendations

# Prediction system
def recommend_destinations(user_input, model, label_encoders, features, data):
    # Encode user input
    encoded_input = {}
    for feature in features:
        if feature in label_encoders:
            encoded_input[feature] = label_encoders[feature].transform([user_input[feature]])[0]
        else:
            encoded_input[feature] = user_input[feature]

    # Convert to DataFrame
    input_df = pd.DataFrame([encoded_input])

    # Predict popularity
    predicted_popularity = model.predict(input_df)[0]

    return predicted_popularity


# Route for the Home Page
@app.route('/')
def index():
    return render_template('index.html')
# Route for Travel Recommendation Page
@app.route('/recommendation')
def recommendation():
    return render_template('recommendation.html')
@app.route('/search')
def search():
    return render_template('search.html')
@app.route('/itinerary', methods=["GET","POST"])
def itinerary():

    bot_reply = None
    user_message = None

    if request.method == "POST":

        user_message = request.form.get("message", "")
        places = request.form.getlist("selected_places") 
        places = list(dict.fromkeys(places))
        if not places:
            places = session.get('recommended_places', [])
        days = request.form.get("days")
        budget = request.form.get("budget")
        transport = request.form.get("transport")
        food = request.form.get("food", "Veg")
        interests = request.form.get("interests")
        print("Selected places from form:", places)

        # # Example recommended places
        # recommended_places = session.get('recommended_places', []) 
        bot_reply = generate_itinerary(
        user_message=user_message,
        recommended_places=places,   
        days=days,
        budget=budget,
        transport=transport,
        food=food,
        interests=interests
)
        
        return render_template(
        "itinerary.html",
        user_message=user_message,
        bot_reply=bot_reply,
        recommended_places=session.get('recommended_places', [])
    )
# @app.route("/itinerary", methods=["POST"])
# def generate_plan():

#     recommended_places = session.get("recommended_places", [])

#     itinerary = generate_itinerary(prompt, recommended_places)

#     return render_template(
#         "itinerary.html",
#         bot_reply=itinerary,
#         recommended_places=recommended_places   # ✅ IMPORTANT
#     )
# Route for the recommendation
@app.route("/recommend", methods=['GET', 'POST'])
def recommend():
    if request.method == "POST":
        user_id = int(request.form['user_id'])

        user_input = {
            'Name_x': request.form['name'],
            'Type': request.form['type'],
            'State': request.form['state'],
            'BestTimeToVisit': request.form['best_time'],
            'Preferences': request.form['preferences'],
            'Gender': request.form['gender'],
            'NumberOfAdults': request.form['adults'],
            'NumberOfChildren': request.form['children'],
        }

        # Collaborative filtering recommendation
        recommended_destinations = collaborative_recommend(
            user_id, user_similarity, user_item_matrix, destinations_df
        )

        # Popularity prediction
        predicted_popularity = recommend_destinations(
            user_input, model, label_encoders, features, df
        )

        # Extract POI names
        poi_list = recommended_destinations['Name'].tolist()
        session['recommended_places'] = poi_list
        # Generate AI itinerary
        # itinerary = generate_itinerary(user_message="", recommended_places=poi_list)

        return render_template(
            
    "recommendation.html",
    recommended_destinations=recommended_destinations,
    predicted_popularity=predicted_popularity,
    itinerary=itinerary,
    recommended_places=poi_list   # ✅ ADD THIS LINE
)
        print("POI LIST:", poi_list)
        return render_template('recommendation.html')

if __name__ == '__main__':
    # Run the app in debug mode
    app.run(debug=True)
