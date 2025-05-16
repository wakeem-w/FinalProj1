import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Title for the App
st.title("House Price Prediction App")

# Automatically load the attached dataset
file_path = "house_prices_records.csv"  # Replace with the actual name of your file
try:
    # Load the dataset
    df = pd.read_csv(file_path)
    st.write("### Dataset Preview:")
    st.dataframe(df.head())  # Display dataframe preview

    # Handle missing values
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Correlation heatmap
    if st.checkbox("Show Correlation Heatmap"):
        st.write("### Correlation Heatmap:")
        plt.figure(figsize=(12, 8))
        heatmap = sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.5)
        st.pyplot(plt)

    # Select features for predictions
    st.write("### Choose Features for Predictions:")
    all_features = list(df.columns)
    numeric_features = [col for col in all_features if pd.api.types.is_numeric_dtype(df[col])]

    features = st.multiselect("Select the features to use for training the model:", numeric_features,
                              default=["GrLivArea", "OverallQual", "YearBuilt"])
    target = st.selectbox("Select the target variable:", numeric_features,
                          index=numeric_features.index("SalePrice") if "SalePrice" in numeric_features else 0)

    if features and target:
        X = df[features]
        y = df[target]

        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        model = RandomForestRegressor()
        model.fit(X_train, y_train)

        # Evaluate model
        predictions = model.predict(X_test)
        rmse = mean_squared_error(y_test, predictions, squared=False)

        st.write("### Model Evaluation:")
        st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")

        # Allow user input for predictions
        st.write("### Make Predictions:")
        # Generate input form dynamically based on selected features
        user_inputs = {}
        for feature in features:
            user_inputs[feature] = st.number_input(f"Enter value for {feature}:", value=float(df[feature].mean()))

        # Prediction results
        if st.button("Predict"):
            user_input_df = pd.DataFrame([user_inputs])
            user_prediction = model.predict(user_input_df)
            st.write(f"### Predicted {target}: **{user_prediction[0]:,.2f}**")
except FileNotFoundError:
    st.error(f"File '{file_path}' not found. Make sure the file is in the proper directory.")
except pd.errors.EmptyDataError:
    st.error("The dataset file is empty or invalid.")
except Exception as e:
    st.error(f"An error occurred: {e}")