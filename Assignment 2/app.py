from flask import Flask, render_template, request, jsonify
import pickle
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained models
size_model = load_model('size_recommendation_model.h5')  
with open('product_recommendation_model.pkl', 'rb') as file:
    product_model = pickle.load(file)  

# Load dummy data for recommendation
interaction_matrix = pd.read_csv('interaction_matrix.csv', index_col=0)
item_features = pd.read_csv('item_features.csv', index_col=0)

# Calculate item similarity
item_similarity = cosine_similarity(item_features)
item_similarity_df = pd.DataFrame(item_similarity, index=item_features.index, columns=item_features.index)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        # User inputs
        height = float(request.form['height'])
        weight = float(request.form['weight'])
        user_id = int(request.form['user_id'])
        item_id = int(request.form['item_id'])

        size_prediction = size_model.predict([[height, weight]])
        suggested_size = ['S', 'M', 'L', 'XL'][np.argmax(size_prediction)]

        similar_items = item_similarity_df[item_id].sort_values(ascending=False).index[1:6].tolist()
        user_recommendations = recommend_items_for_user(user_id, 5)
        hybrid_recommendations = list(set(similar_items).union(user_recommendations))[:5]

        return render_template('result.html', size=suggested_size, recommendations=hybrid_recommendations)
    except Exception as e:
        return str(e)

def recommend_items_for_user(user_id, k):
    user_similarity = cosine_similarity(interaction_matrix)
    user_similarity_df = pd.DataFrame(user_similarity, index=interaction_matrix.index, columns=interaction_matrix.index)
    similar_users = user_similarity_df[user_id].sort_values(ascending=False).index[1:k + 1]
    
    recommended_items = set()
    for similar_user in similar_users:
        items = interaction_matrix.loc[similar_user][interaction_matrix.loc[similar_user] > 0].index
        recommended_items.update(items)
    
    return list(recommended_items)

if __name__ == '__main__':
    app.run(debug=True)
