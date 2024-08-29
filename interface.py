import streamlit as st
import torch
from flask import Flask, request, jsonify
from threading import Thread

# API для рекомендаций на Flask
app = Flask(__name__)

@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = int(request.args.get('user_id'))
    user_tensor = torch.tensor([user_id], dtype=torch.long)
    item_ids = torch.arange(0, n_items)
    with torch.no_grad():
        predictions = model(user_tensor, item_ids).squeeze().numpy()
    top_items = predictions.argsort()[-5:][::-1]
    return jsonify(top_items.tolist())

# Интерфейс на Streamlit
def streamlit_app():
    st.title("Movie Recommendation System")
    st.write("Enter a User ID to get movie recommendations.")

    user_id = st.text_input("User ID")
    if st.button("Get Recommendations"):
        user_id = int(user_id)
        user_tensor = torch.tensor([user_id], dtype=torch.long)
        item_ids = torch.arange(0, n_items)
        with torch.no_grad():
            predictions = model(user_tensor, item_ids).squeeze().numpy()
        top_items = predictions.argsort()[-5:][::-1]
        st.write(f"Recommendations for User {user_id}:")
        for item_id in top_items:
            st.write(f"Item ID: {item_id}, Predicted Rating: {predictions[item_id]}")

# Загрузка модели
model = RecommenderNet(n_users=n_users, n_items=n_items, embedding_size=embedding_size)
model.load_state_dict(torch.load('model.pt'))
model.eval()

# Запуск Flask и Streamlit одновременно
if __name__ == '__main__':
    # Flask в отдельном потоке
    flask_thread = Thread(target=lambda: app.run(debug=True, use_reloader=False))
    flask_thread.start()

    # Streamlit UI
    streamlit_app()
