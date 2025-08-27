import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.title("🚦 TraffixAI: Smart Traffic Management")

uploaded_file = st.file_uploader("Upload Traffic Dataset (CSV)", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.subheader("📊 Dataset Preview")
    st.write(data.head())

    # Graph
    if "hour" in data.columns and "vehicles" in data.columns:
        st.subheader("🚗 Traffic Flow by Hour")
        fig, ax = plt.subplots()
        data.groupby("hour")["vehicles"].sum().plot(kind="line", ax=ax)
        st.pyplot(fig)

    # ML Model
    if st.button("Run Model"):
        if "congestion" in data.columns:
            X = data.drop("congestion", axis=1)
            y = data["congestion"]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            model = DecisionTreeClassifier(random_state=2)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.success(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.2f}")
            st.write("### Confusion Matrix")
            st.write(confusion_matrix(y_test, y_pred))
            st.write("### Classification Report")
            st.text(classification_report(y_test, y_pred))
        else:
            st.error("Dataset me 'congestion' column chahiye")