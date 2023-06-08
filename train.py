from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import pandas as pd

#  Checking whether the dataset is balanced or not

DATA_PATH = "Data/Training.csv"
data = pd.read_csv(DATA_PATH).dropna(axis=1)
disease_counts = data["prognosis"].value_counts()
temp_df = pd.DataFrame(
    {"Disease": disease_counts.index, "Counts": disease_counts.values}
)

encoder = LabelEncoder()
data["prognosis"] = encoder.fit_transform(data["prognosis"])

X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=24
)


# Initializing Models
models = {"Random Forest": RandomForestClassifier(random_state=18)}


# Training and testing Random Forest Classifier
rf_model = RandomForestClassifier(random_state=18)
rf_model.fit(X_train, y_train)
preds = rf_model.predict(X_test)

final_rf_model = RandomForestClassifier(random_state=18)
final_rf_model.fit(X, y)

symptom_index = {}
for index, value in enumerate(X.columns.values):
    symptom_index[value] = index

data_dict = {"predictions_classes": encoder.classes_, "symptom_index": symptom_index}

model = {
    "final_model": final_rf_model,
    "symptoms": X.columns.values,
    "data_dict": data_dict,
}

filename = "finalized_model.sav"
pickle.dump(model, open(filename, "wb"))
