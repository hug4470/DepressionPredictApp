from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import pickle
from sklearn.metrics import recall_score


df_test = pd.read_csv("../data/test/df_test.csv", sep=",")

x_test = df_test.drop(columns=['Target'])
y_test = df_test['Target']


with open('../models/otros/best_model_py.pkl', 'rb') as archivo:
    modelo_importado = pickle.load(archivo)

x_test = x_test.reindex(columns=modelo_importado.feature_names_in_, fill_value=0)
y_pred = modelo_importado.predict(x_test)

print("Recall: ", modelo_importado.best_score_)
print("Classification Report:")
print(classification_report(y_test, y_pred))