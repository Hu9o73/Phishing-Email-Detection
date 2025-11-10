import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, accuracy_score


class ModelManager:
    
    def __init__(self, datamanager):
        self.dm = datamanager
        self.model = None
        self.model_name = None
        self.models_dir = os.path.join("models")
        os.makedirs(self.models_dir, exist_ok=True)

    # -----------------------------
    # Data preparation
    # -----------------------------
    def get_features_and_labels(self):
        
        if getattr(self.dm, "X_processed", None) is None:
            raise ValueError("Data not vectorized yet. Please preprocess first.")
        if "Label" not in self.dm.df.columns:
            raise ValueError("No 'Label' column found in dataset.")

        X = self.dm.X_processed
        y = self.dm.df["Label"]
        return X, y

    # -----------------------------
    # Training
    # -----------------------------
    def train_model(self, model_type="RandomForest"):

        #Train a classical ML model (RandomForest or SGDClassifier)."""
        X, y = self.get_features_and_labels()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model_type = model_type.lower().replace(" ", "_")

        if model_type == "random_forest":
            self.model_name = "RandomForest"
            self.model = RandomForestClassifier(
                n_estimators=100, random_state=42, n_jobs=-1
            )
        elif model_type == "sgd":
            self.model_name = "SGDClassifier"
            self.model = SGDClassifier(loss="log_loss", max_iter=1000, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        print(f"\nTraining {self.model_name}...")
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        print(f"\n✅ Accuracy: {acc:.4f}\n")
        print(classification_report(y_test, y_pred))

    # -----------------------------
    # Saving and loading
    # -----------------------------
    def save_model(self):
        
        if self.model is None:
            print("No model trained yet.")
            return

        path = os.path.join(self.models_dir, f"{self.model_name}.joblib")
        joblib.dump(self.model, path)
        print(f"\n💾 Model saved to {path}")

    def load_model(self, model_name):
    
        path = os.path.join(self.models_dir, f"{model_name}.joblib")

        if not os.path.exists(path):
            print(f"No saved model found at {path}")
            return

        self.model = joblib.load(path)
        self.model_name = model_name
        print(f"\n✅ Loaded model: {model_name}")

    # -----------------------------
    # Continue training
    # -----------------------------
    def continue_training(self):
        
        if self.model is None:
            print("No model loaded or trained.")
            return

        X, y = self.get_features_and_labels()
        print(f"\nContinuing training of {self.model_name} on full data...")
        self.model.fit(X, y)
        print("🔁 Model updated.")
