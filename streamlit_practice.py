import streamlit as st
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score





iris = datasets.load_iris()

data = iris.data

target =  iris.target

# 0 for the Setosa , 1 for Versicolour , 2 for Virginica

data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.2, random_state=42)
knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(data_train, target_train)



st.title("Iris Flower Classification")

col1,col2= st.columns(2)
col1.image("iris_flower.jpg", caption="Iris Flower")


col2.write("### A simple web application built with Streamlit that uses the K-Nearest Neighbors (KNN) algorithm to classify Iris flowers based on user-inputted characteristics.")

st.write("### Input Data")
st.caption("Enter the characteristics of an Iris flower below to predict its species (Setosa, Versicolour, or Virginica) using the K-Nearest Neighbors (KNN) algorithm.")

col3,col4= st.columns(2)
sepal_length = col3.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
sepal_width = col4.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.0, step=0.1)
petal_length = col3.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.5, step=0.1)
petal_width = col4.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2, step=0.1)

if st.button("Predict Iris Class"):

    user_input = [[sepal_length, sepal_width, petal_length, petal_width]]

    prediction = knn.predict(user_input)

    flower_classes = {0: "Setosa", 1: "Versicolour", 2: "Virginica"}
    predicted_class = flower_classes[prediction[0]]

    st.write(f"### The predicted class is: **{predicted_class}**")

target_pred = knn.predict(data_test)
accuracy = accuracy_score(target_test, target_pred)
st.write(f"Model accuracy: {accuracy * 100:.2f}%")

st.markdown(
    """
    <div style="text-align: center; font-size: 9px; color: gray;">
        <p>Created by ELHAIBA</p>
    </div>
    """,
    unsafe_allow_html=True
)