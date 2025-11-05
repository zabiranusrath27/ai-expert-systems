# AI-Enabled Expert System for Process Optimization in Manufacturing SMEs

This project builds a machine-learning based expert system to predict downtime/failures and support decision-making in manufacturing SMEs.  
It uses machine-learning analytics (Random Forest / XGBoost) along with explainable metrics.

---

## 📁 Project Structure

ML-Project/
│
├── data/ # CSV dataset (raw file placed here)
├── artifacts/ # Generated artifacts (saved models, metrics)
│ ├── preprocessed_data.pkl
│ ├── RandomForest.pkl
│ └── RandomForest_metrics.txt
│
├── dataprep.py # Cleans data, handles preprocessing & splitting
├── train_models.py # Trains Random Forest (and optionally XGBoost)
├── app.py (optional) # Streamlit dashboard (model visualization)
├── main.py (optional) # Pipeline executor
│
└── README.md # Project documentation

step 1: open a directory and clone the repository
step 2: in the folder loaction start a virtual environment using "python -m venv venv"
step 3: activate the virtual environment using " .\venv\Scripts\activate"
step 4: install the packages required for the project using the command "pip install -r requirements.txt
"
step 5: take the control to src folder and run the data_prep.py using the command "python data_prep.py"
step 6:now train the model by giving the command "python train_models.py"
step 7: take the control out of the src folder by giving "cd.."
step 8: deploy the dashboard using the command "streamlit run dashboard/app.py"


for SHAP analysis python 3.12 is required hence new venv is created and then python 3.12 is downloaded then
command for opening virtual environment in python 3.12.3 "python3.12 -m venv mlenv"
to install shap "pip install shap --only-binary :all:  "
