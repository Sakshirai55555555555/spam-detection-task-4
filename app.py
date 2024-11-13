from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# Load the trained model and vectorizer
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

with open('spam_classifier_model.pkl', 'rb') as f:
    spam_classifier_model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')  # Serve the HTML form

# Step 3: Define the Prediction Endpoint
@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']  # Get the message from the form

    if not message:
        return render_template('index.html', error="No message provided")  # Pass error message

    # Preprocess and transform the message
    message_tfidf = tfidf_vectorizer.transform([message])

    # Make the prediction
    prediction = spam_classifier_model.predict(message_tfidf)
    result = 'spam' if prediction[0] == 1 else 'not spam'
    
    # Define the color based on the prediction
    color = 'red' if result == 'spam' else 'green'

    return render_template('index.html', message=message, result=result, color=color)

# Step 4: Run the Flask App
if __name__ == '__main__':
    app.run(debug=True)
