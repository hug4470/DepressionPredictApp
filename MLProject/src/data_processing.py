import pandas as pd
import numpy as np

df = pd.read_csv('../data/raw/dataset.csv/depression_data.csv')

def procesador(df):
    df['Chronic Medical Conditions'] = df['Chronic Medical Conditions'].replace({'Yes': 1, 'No': 0})
    df['History of Mental Illness'] = df['History of Mental Illness'].replace({'Yes': 1, 'No': 0})
    df['Family History of Depression'] = df['Family History of Depression'].replace({'Yes': 1, 'No': 0})
    df['History of Substance Abuse'] = df['History of Substance Abuse'].replace({'Yes': 1, 'No': 0})

    def clasificar_edad(df):
        def categorizar_edad(i):
            if 18 <= i < 40:
                return 'Young Adulthood'
            elif 40 <= i < 60:
                return 'Adulthood'
            elif i >= 60:
                return 'Eld'
            
        df['Age_Group'] = df['Age'].apply(categorizar_edad)
        return df

    df = clasificar_edad(df)
    df.drop(columns='Age', inplace=True)

    dummies_age = pd.get_dummies(df['Age_Group'], prefix='Age')
    df = pd.concat([df, dummies_age], axis=1)
    df.drop(columns='Age_Group', inplace=True)

    df['Physical Activity Level'] = df['Physical Activity Level'].replace({'Active': 0, 'Sedentary': 10, 'Moderate': 5})
    df['Smoking Status'] = df['Smoking Status'].replace({'Non-smoker': 0, 'Current': 10, 'Former': 5})
    df['Alcohol Consumption'] = df['Alcohol Consumption'].replace({'Low': 0, 'Moderate': 5, 'High': 10})
    df['Dietary Habits'] = df['Dietary Habits'].replace({'Unhealthy': 10, 'Moderate': 5, 'Healthy': 0})
    df['Sleep Patterns'] = df['Sleep Patterns'].replace({'Poor': 10, 'Fair': 5, 'Good': 0})

    dummies = pd.get_dummies(df['Marital Status'], prefix='Status')
    df = pd.concat([df, dummies], axis=1)
    df.drop(columns='Marital Status', inplace=True)

    df['Log_Income'] = np.log1p(df['Income'])

    df.drop(columns = ['Income'], inplace = True)  

    dummies_education = pd.get_dummies(df['Education Level'], prefix='Education')
    df = pd.concat([df, dummies_education], axis=1)
    df.drop(columns='Education Level', inplace=True)

    df['Employment Status'] = df['Employment Status'].replace({'Unemployed': 1, 'Employed': 0})
    
    df.drop(columns='Name', inplace=True)

    df['Target'] = df['Chronic Medical Conditions']
    df.drop(columns='Chronic Medical Conditions', inplace=True)

    return df

df = procesador(df)

df.to_csv('../data/processed/df_procesado.csv', index=False)
