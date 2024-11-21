import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier

df_procesado = pd.read_csv('../data/processed/df_procesado.csv', index_col=0)

def modelo_entreno(df):
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
    train_df.to_csv('../data/train/df_train.csv', index=False)
    test_df.to_csv('../data/test/df_test.csv', index=False)

    df_train = pd.read_csv("../data/train/df_train.csv", sep=",") 
    
    x = df_train.drop(columns=['Target'])  
    y = df_train['Target']   
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    smote = SMOTE(random_state=42)
    x_resampled, y_resampled = smote.fit_resample(x_train, y_train)

    pipeline_sgd = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=5)),
    ('classifier', SGDClassifier(random_state=42))
    ])

    param_grid_sgd = {
    'pca__n_components': [4, 5, 6], 
    'classifier__loss': ['hinge'], 
    'classifier__alpha': [0.005, 0.03, 0.05],
    'classifier__penalty': ['elasticnet'],
    }
    grid_search_sgd = GridSearchCV(
        pipeline_sgd,
        param_grid_sgd,
        cv=5,
        scoring='recall',
        n_jobs=-1,
        verbose=1
    )

    grid_search_sgd.fit(x_resampled, y_resampled)

    return grid_search_sgd

best_model = modelo_entreno(df_procesado)

with open('../models/otros/best_model_py.pkl', 'wb') as archivo:
    pickle.dump(best_model, archivo)