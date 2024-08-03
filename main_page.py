import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt

from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os

st.set_page_config(page_title="Unmasking the Churn", layout="wide")
main, banner = st.columns(spec=[7, 2], gap="large")
main.title("Unmasking the Churn")
st.sidebar.image("static/social-customer-churn-analysis.jpg", use_column_width=True)


# Load models
class ModelWrapper(BaseEstimator):
    def __init__(self, model_path, features_path):
        self.model = joblib.load(model_path)
        self.trained_features = joblib.load(features_path)

    def predict(self, X):
        return self.model.predict(X[self.trained_features])

@st.cache_resource
def load_model(model_path, features_path):
    print(f"Loading model from {model_path}")
    return ModelWrapper(model_path, features_path)

random_forest_model = load_model(
    'models/random_forest_model.pkl',
    'features_selected/rf_features.pkl'
)
svm_model = load_model(
    'models/svm_model.pkl',
    'features_selected/svm_features.pkl'
)
logistic_regression_model = load_model(
    'models/logistic_regression_model.pkl',
    'features_selected/logistic_regression_features.pkl'
)

adaboost_model = load_model(
    'models/adaboost_model.pkl',
    'features_selected/adaboost_features.pkl'
)
gaussian_nb_model = load_model(
    'models/gaussian_nb_model.pkl',
    'features_selected/gaussian_nb_features.pkl'
)
knn_model = load_model(
    'models/knn_model.pkl',
    'features_selected/knn_features.pkl'
)


# Model selection dropdown
selected_model = st.sidebar.selectbox(
    label="Select a model",
    options=["Random Forest", "SVM", "Logistic Regression", "AdaBoost", "Gaussian Naive Bayes", "K-Nearest Neighbors"],
    on_change=lambda: st.session_state.update(analyze=False)
)


# Load stats
@st.cache_resource
def load_stats(file_path):
    df = pd.read_csv(file_path, index_col=0)
    return df

stats = load_stats("static/evaluation.csv")


# Model selection logic
if selected_model == "Random Forest":
    model = random_forest_model
    st.session_state.update(model_name="Random Forest")
    st.session_state.update(model_stats=[0.84, 0.88, 0.86, 0.95])

elif selected_model == "SVM":
    model = svm_model
    st.session_state.update(model_name="SVM")
    st.session_state.update(model_stats=[0.72, 0.82, 0.77, 0.92])

elif selected_model == "Logistic Regression":
    model = logistic_regression_model
    st.session_state.update(model_name="Logistic Regression")
    st.session_state.update(model_stats=[0.62, 0.74, 0.68, 0.89])

elif selected_model == "AdaBoost":
    model = adaboost_model
    st.session_state.update(model_name="AdaBoost")
    st.session_state.update(model_stats=[0.76, 0.90, 0.82, 0.94])

elif selected_model == "Gaussian Naive Bayes":
    model = gaussian_nb_model
    st.session_state.update(model_name="Gaussian Naive Bayes")
    st.session_state.update(model_stats=[0.53, 0.69, 0.60, 0.85])

elif selected_model == "K-Nearest Neighbors":
    model = knn_model
    st.session_state.update(model_name="K-Nearest Neighbors")
    st.session_state.update(model_stats=[0.70, 0.77, 0.74, 0.91])
else:
    st.sidebar.write("Invalid model selection")

# Load individual model stats
@st.cache_resource
def load_chart(series):
    df = series.to_frame().T
    df = df.melt(var_name='metrics', value_name='value')

    bar = alt.Chart(df).mark_bar(size=25).encode(
        alt.Y('metrics:N', title=None),
        alt.X('value:Q', title=None),
        text=alt.Text('value:Q')
    ).properties(height=200)

    text = bar.mark_text(
        align='right',
        baseline='middle',
        dx=-3
    ).encode(
        text='value:Q'
    )

    chart = (bar + text)
    return chart

st.sidebar.write("Model Performance")
st.sidebar.altair_chart(load_chart(stats.loc[st.session_state.get('model_name')]), use_container_width=True)


# Load advertisement videos
advert_toggle = st.sidebar.toggle(label="Show Advertisement", value=False,
                                  on_change=lambda: st.session_state.update(advert=not advert_toggle))

if st.session_state.get("advert"):
    banner.write("Advertisement")
    video_files = [file for file in os.listdir("static") if file.endswith(".mp4")]
    np.random.shuffle(video_files)
    for video_file in video_files:
        banner.video(f"static/{video_file}", format="video/mp4", loop=True, autoplay=True, muted=True)


# Function to process the data and feed it into the ML model
def process_data(file):
    with main:
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(file)
        df = df.drop(columns=['Attrition_Flag'], errors='ignore')
        st.write(df)
        st.button(label="Analyze", on_click=lambda: st.session_state.update(analyze=True))

        if st.session_state.get("analyze"):
            X = df.copy()
            # Convert True/False to 1/0 value
            X['Marital_Status_Divorced'] = X['Marital_Status_Divorced'].replace({True: 1, False: 0})
            X['Marital_Status_Married'] = X['Marital_Status_Married'].replace({True: 1, False: 0})
            X['Marital_Status_Single'] = X['Marital_Status_Single'].replace({True: 1, False: 0})
            X['Marital_Status_Unknown'] = X['Marital_Status_Unknown'].replace({True: 1, False: 0})

            # Scale the data
            scaling_columns = [
                'Dependent_count', 'Education_Level',
                'Income_Category', 'Months_on_book',
                'Total_Relationship_Count', 'Months_Inactive_12_mon',
                'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
                'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
                'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1'
            ]
            std_scaler = StandardScaler()
            min_max_scaler = MinMaxScaler()
            X['Customer_Age'] = std_scaler.fit_transform(X[['Customer_Age']])
            X[scaling_columns] = min_max_scaler.fit_transform(X[scaling_columns])

            # Make predictions
            y = model.predict(X)

            # Merge predictions array with original table
            prediction_df = pd.concat([df, pd.DataFrame(y, columns=['Prediction'])], axis=1)

            # Create a window with multiple tabs
            tabs = st.tabs(["Data", "Statistics"])

            # Display prediction_df in the first tab
            with tabs[0]:
                st.write(f"Customers predicted to be close to churning by {st.session_state.get('model_name')}:")
                churned_df = prediction_df[prediction_df['Prediction'] == 1]
                st.write(churned_df)
                st.download_button(label="Save CSV", data=churned_df.to_csv().encode("utf-8"),
                                   file_name="churned.csv", mime="text/csv")

            # Display statistics of prediction_df in the second tab
            with tabs[1]:
                st.write("Statistics of prediction:")

                churned_df = prediction_df[prediction_df['Prediction'] == 1]
                st.write("Churned customers:" + str(churned_df.shape[0]) + " customers")
                not_churned_df = prediction_df[prediction_df['Prediction'] == 0]
                st.write("Retained customers:" + str(not_churned_df.shape[0]) + " customers")

                # Get the names of the first two features in model.trained_features
                feature1 = 'Total_Trans_Ct'
                feature2 = 'Total_Trans_Amt'

                # Create an Altair chart
                chart = alt.Chart(prediction_df).mark_circle().encode(
                    x=feature1,
                    y=feature2,
                    color='Prediction',
                    tooltip=['Prediction', feature1, feature2]
                ).interactive()

                # Display the Altair chart
                st.altair_chart(chart, use_container_width=True)


# Allow the user to drop a file
file = main.file_uploader("Upload a CSV file", type="csv")

if file is not None:
    if file.name != st.session_state.get("file_name"):
        st.session_state.update(file_name=file.name)
        st.session_state.update(analyze=False)
    process_data(file)

