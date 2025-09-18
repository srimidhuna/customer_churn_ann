Customer Churn Prediction with ANN

This project builds an Artificial Neural Network (ANN) model to predict customer churn (whether a customer will exit or not). The workflow includes data preprocessing, model training, evaluation, and deployment using Gradio.

📂 Project Structure
├── Churn_Modelling_3000_Top8.csv   # Dataset
├── app.py                          # Gradio web app
├── model.pkl                       # Trained ANN model (optional)
├── requirements.txt                # Dependencies
└── README.md                       # Project documentation

📊 Dataset

The dataset used is Churn_Modelling_3000_Top8.csv.
It contains customer details like:

CreditScore

Geography

Gender

Age

Tenure

Balance

NumOfProducts

EstimatedSalary

Exited (target: 1 = churn, 0 = not churn)

⚙️ Preprocessing

The following preprocessing steps were applied:

Handled categorical variables

Converted categorical features (e.g., Geography, Gender) using OneHotEncoder.

Scaled numerical variables

Standardized numeric columns using StandardScaler.

Train-test split

Split dataset into training and testing sets.

🤖 Model: Artificial Neural Network (ANN)

Built using TensorFlow / Keras.

Network architecture:

Input layer: number of selected features

Hidden layers: Dense layers with ReLU activation

Output layer: single neuron with sigmoid activation

Metrics: accuracy

📈 Evaluation

The trained ANN was evaluated on the test set using:

Accuracy

Precision

Recall

F1-score

ROC-AUC

Confusion matrix and classification report were also generated.

🌐 Deployment with Gradio

A Gradio web app (app.py) was built to interact with the model.

Features of the app:

User can either:

Enter a row index to test prediction against actual label from dataset.

Enter feature values manually to get churn prediction.

Outputs:

Prediction: Will Churn / Will Not Churn

Probability of churn

Notes: Actual value (if row index is used)

Run the app:

python app.py

📦 Installation

Clone this repository.

Install dependencies:

pip install -r requirements.txt


Run training or directly start the app:

python app.py