import streamlit as st
import joblib

# Load the trained model
model = joblib.load('sentiment_model.pkl')

# Title
st.title('Movie Review Sentiment Analysis')

# Text input
review = st.text_area('Enter the movie review:', '')

# Prediction button
if st.button('Analyze Sentiment'):
    if review:
        # Prediction
        prediction = model.predict([review])
        # Display result
        if prediction[0] == 'positive':
            st.success('Positive Sentiment')
        else:
            st.error('Negative Sentiment')
    else:
        st.warning('Please enter a review.')
