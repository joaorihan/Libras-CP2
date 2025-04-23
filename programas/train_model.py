import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
import joblib

class ModelTrainer:
    def __init__(self, data_dir: str, model_path: str):
        self.data_dir = Path(data_dir)
        self.model_path = Path(model_path)
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )

    def load_data(self):
        dataframes = []
        for csv_file in self.data_dir.glob("*.csv"):
            df = pd.read_csv(csv_file)
            dataframes.append(df)
        
        return pd.concat(dataframes, ignore_index=True)

    def train(self):
        # Carrega e prepara dados
        data = self.load_data()
        X = data.drop('label', axis=1)
        y = data['label']

        # Split dados
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # Treina modelo
        self.model.fit(X_train, y_train)

        # Avalia
        cv_scores = cross_val_score(self.model, X, y, cv=5)
        print(f"Cross-validation scores: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")
        
        # Avalia no conjunto de teste
        y_pred = self.model.predict(X_test)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Salva modelo
        joblib.dump(self.model, self.model_path) 