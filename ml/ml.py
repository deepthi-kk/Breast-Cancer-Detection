import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class MachineLearningModel:
    def __init__(self) -> None:
        self.model = None
        self._fit2()
        print("Model initialized")

    def _fit2(self) -> None:
        with open('./ml/models/model.pkl', 'rb') as f:
            self.model = pickle.load(f)

    def _fit(self) -> None:
        df = pd.read_csv('./ml/data.csv')
        df = df.rename(
            columns={'concave points_mean': 'concave_points_mean', 'concave points_worst': 'concave_points_worst'})
        df.drop(columns='Unnamed: 32', axis=1, inplace=True)
        df.drop(columns='id', axis=1, inplace=True)
        le = LabelEncoder()
        labels = le.fit_transform(df['diagnosis'])
        df['target'] = labels
        df.drop(columns='diagnosis', axis=1, inplace=True)
        df_copy = df.copy()
        hashmap = {}
        for column in df_copy.columns:
            correlation = df_copy[column].corr(df_copy["target"])
            if correlation >= 0.65:
                hashmap[column] = correlation

        column_list = []
        for keys in hashmap.keys():
            column_list.append(keys)

        df = pd.DataFrame(df_copy, columns=column_list)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        self.model = SVC(C=10, kernel='linear').fit(X_train, y_train)
        print('score: ', self.model.score(X_test, y_test))


    def _get_scaled_data(self, input):
        df = pd.read_csv('./ml/data.csv')
        df = df.rename(
            columns={'concave points_mean': 'concave_points_mean', 'concave points_worst': 'concave_points_worst'})
        df.drop(columns='Unnamed: 32', axis=1, inplace=True)
        df.drop(columns='id', axis=1, inplace=True)
        le = LabelEncoder()
        labels = le.fit_transform(df['diagnosis'])
        df['target'] = labels
        df.drop(columns='diagnosis', axis=1, inplace=True)
        df_copy = df.copy()
        hashmap = {}
        for column in df_copy.columns:
            correlation = df_copy[column].corr(df_copy["target"])
            if correlation >= 0.65:
                hashmap[column] = correlation

        column_list = []
        for keys in hashmap.keys():
            column_list.append(keys)

        df = pd.DataFrame(df_copy, columns=column_list)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        sc = StandardScaler()
        sc.fit_transform(X_train)
        return sc.transform(input)

    def predict(self, radius_mean, perimeter_mean, area_mean, concavity_mean, concave_points_mean, radius_worst,
                perimeter_worst, area_worst, concavity_worst, concave_points_worst):

        x = self._get_scaled_data(
            np.array([[radius_mean, perimeter_mean, area_mean, concavity_mean, concave_points_mean, radius_worst,
                       perimeter_worst, area_worst, concavity_worst, concave_points_worst]]))
        result = self.model.predict(x)
        if result[0] == 1:
            return "M"
        return "B"
