# Unmasking the Churn: A demo application

This is a demo web application for our Churn Prediction project. The idea is to
predict the subset of customers who are likely to churn from a CSV dataset of
customer information.

## Setting up the app

Python 3.12 recommended, `python` environment variable is a must. Navigate to the
app directory (where `main_page.py` is located), open the terminal there and
run the following commands:

Setting up virtual environment (recommended):
```bash
python -m venv venv
```
Activate virtual environment for Windows:
```bash
source venv/Scripts/activate
```
Activate virtual environment for MacOS/Linux:
```bash
source venv/bin/activate
```
Install required packages:
```bash
pip install --no-cache-dir -r requirements.txt
```
Finally, run the app!
```bash
streamlit run main_page.py
```

## Using the app

In the main panel, you can upload a CSV file containing customer information and
preview the data. Click ```Analyze``` to start the churn prediction process. The
result will be displayed and can be downloaded as a CSV file.

In the sidebar, you can select the model to use for prediction.