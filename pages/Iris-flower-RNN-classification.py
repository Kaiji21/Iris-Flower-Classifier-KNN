import streamlit as st
from sklearn import datasets
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="Iris RNN classifier",
    page_icon="🤖",
)

# Load and prepare data
iris = datasets.load_iris()
data = iris.data
target = iris.target

# Scale the features
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Split the data
data_train, data_test, target_train, target_test = train_test_split(
    data_scaled, target, test_size=0.2, random_state=42
)

# Reshape data for RNN (samples, timesteps, features)
data_train_rnn = data_train.reshape(data_train.shape[0], 1, data_train.shape[1])
data_test_rnn = data_test.reshape(data_test.shape[0], 1, data_test.shape[1])

# Create RNN model
def create_rnn_model():
    model = Sequential([
        SimpleRNN(32, input_shape=(1, 4), activation='relu'),
        Dense(16, activation='relu'),
        Dense(3, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    return model

# Create and train model
@st.cache_resource
def train_rnn_model():
    model = create_rnn_model()
    model.fit(data_train_rnn, target_train, epochs=100, batch_size=32, verbose=0)
    return model

rnn_model = train_rnn_model()

# UI Elements
st.title("Iris Flower Classification using RNN")

col1, col2 = st.columns(2)
col1.image("iris_flower.jpg", caption="Iris Flower")

col2.write("### A neural network-based approach using Recurrent Neural Networks (RNN) to classify Iris flowers based on their characteristics.")

st.write("### Input Data")
st.caption("Enter the characteristics of an Iris flower below to predict its species (Setosa, Versicolour, or Virginica) using a Recurrent Neural Network.")

col3, col4 = st.columns(2)
sepal_length = col3.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
sepal_width = col4.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.0, step=0.1)
petal_length = col3.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.5, step=0.1)
petal_width = col4.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2, step=0.1)

if st.button("Predict Iris Class"):
    # Prepare input data
    user_input = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    user_input_scaled = scaler.transform(user_input)
    user_input_rnn = user_input_scaled.reshape(1, 1, 4)
    
    # Make prediction
    prediction_probs = rnn_model.predict(user_input_rnn, verbose=0)
    prediction = np.argmax(prediction_probs)
    
    flower_classes = {0: "Setosa", 1: "Versicolour", 2: "Virginica"}
    predicted_class = flower_classes[prediction]
    
    # Display prediction and probabilities
    st.write(f"### The predicted class is: **{predicted_class}**")
    
    st.write("### Prediction Probabilities:")
    for i, prob in enumerate(prediction_probs[0]):
        st.write(f"{flower_classes[i]}: {prob:.4f}")

# Model accuracy
test_predictions = np.argmax(rnn_model.predict(data_test_rnn, verbose=0), axis=1)
accuracy = accuracy_score(target_test, test_predictions)
st.write(f"Model accuracy: {accuracy * 100:.2f}%")

st.markdown(
    """
    <div style="text-align: center; font-size: 9px; color: gray;">
        <p>Created by ELHAIBA</p>
    </div>
    """,
    unsafe_allow_html=True
)