import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv("titanic.csv")

    # Drop unnecessary columns
    data.drop(columns=[
        'deck', 'embark_town', 'embarked', 'who',
        'adult_male', 'alive', 'alone', 'class'
    ], inplace=True)

    # Fill missing values
    data['age'] = data['age'].fillna(data['age'].mean())

    # Encode gender
    data['sex'] = data['sex'].map({'male': 1, 'female': 0})

    return data

data = load_data()

st.title("🚢 Titanic Survival Prediction App")
st.write("A Machine Learning app using **Random Forest** and **Titanic dataset**")

# Show data preview


# Features and Labels
X = data.drop(columns='survived')
Y = data['survived']

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Train model
model = RandomForestClassifier()
model.fit(X_train, Y_train)
Prediction = model.predict(X_test)

# Show metrics
#st.subheader("📊 Model Performance")
#st.write("**Accuracy Score:**", accuracy_score(Y_test, Prediction))
#st.text("Classification Report:")
#st.text(classification_report(Y_test, Prediction))

# User input section
st.subheader("🔍 Predict Survival")

class_No = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
Gender = st.radio("Gender", ['Male', 'Female'])
Age = st.slider("Age", 0, 80, 25)
Parch = st.number_input("Number of Parents/Children Aboard (parch)", min_value=0, max_value=10, value=0)
SibSp = st.number_input("Number of Siblings/Spouses Aboard (sibsp)", min_value=0, max_value=10, value=0)
Fare = st.number_input("Fare Paid", min_value=0.0, max_value=600.0, value=50.0)

# Encode gender
gender_encoded = 1 if Gender == 'Male' else 0

user_input = [[class_No, gender_encoded, Age, Parch, SibSp, Fare]]

if st.button("Predict"):
    predict1 = model.predict(user_input)
    if predict1[0] == 0:
        st.error("😢 The passenger did **NOT** survive.")
    else:
        st.success("🎉 The passenger **SURVIVED**!")

