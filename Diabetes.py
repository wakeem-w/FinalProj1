import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Title for the App
st.title("House Price Prediction App")
file_path = "path/to/student_performance.csv"
# Automatically load the attached dataset
file_path = "student_performance.csv"  # Replace with the actual name of your file
try:
    # Load the dataset
    df = pd.read_csv(file_path)
    st.write("### Student Performance Dataset Preview:")
    st.dataframe(df.head())  # Display student dataset preview

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

    # Calculate correlation of numeric features with the target variable (FinalGrade)
    if "FinalGrade" in numeric_features:
        corr_with_target = df.corr(numeric_only=True)["FinalGrade"].sort_values(ascending=False)
        detailed_features = corr_with_target.index[
                            1:6].tolist()  # Select top 5 correlated features excluding FinalGrade
        default_features = detailed_features
    else:
        default_features = ["StudyHours", "MidtermScore", "ClassParticipation"]  # Fallback default features

    features = st.multiselect("Select the features to use for training the model (correlation with target shown):",
                              numeric_features,
                              default=default_features,
                              help="Features pre-populated based on their correlation with the target variable.")
    target = st.selectbox("Select the target variable:", numeric_features,
                          index=numeric_features.index("FinalGrade") if "FinalGrade" in numeric_features else 0)

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
        st.write("### Predict Future Performance:")
        # Generate input form dynamically based on selected features
        user_inputs = {}
        for feature in features:
            user_inputs[feature] = st.number_input(f"Enter value for {feature} (e.g., StudyHours, MidtermScore):",
                                                   value=float(df[feature].mean()))

        # Form for user inputs and predictions
        with st.form("prediction_form"):
            st.write("### Enter Values for Prediction")
            user_inputs = {}
            for feature in features:
                user_inputs[feature] = st.number_input(f"Enter value for {feature}:", value=float(df[feature].mean()))

            # Submit button to trigger prediction
            predict_button = st.form_submit_button("Predict")

        # Prediction logic after form submission
        if predict_button:
            try:
                user_input_df = pd.DataFrame([user_inputs])
                user_prediction = model.predict(user_input_df)
                st.success(f"### Predicted {target} (Future Performance): **{user_prediction[0]:,.2f}**")
            except Exception as ex:
                st.error(f"An error occurred during prediction: {ex}")
except FileNotFoundError:
    st.error(f"File '{file_path}' not found. Make sure the file is in the proper directory.")
except pd.errors.EmptyDataError:
    st.error("The dataset file is empty or invalid.")
except Exception as e:
    st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    st._is_running_with_streamlit = True  # Ensure it's compatible with Streamlit runtime
    from streamlit.web import cli as stcli
    import sys

    sys.argv = ["streamlit", "run", sys.argv[0]]
    sys.exit(stcli.main())
