from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

from ml.training_data import training_data

class ResearchClassifier:

    def __init__(self):
        self.model = Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("classifier", LogisticRegression(max_iter=1000))
        ])

    def train(self):
        texts = [item[0] for item in training_data]
        labels = [item[1] for item in training_data]

        self.model.fit(texts, labels)

        print("ML model trained successfully.")
    

    def predict(self, text):
        prediction = self.model.predict([text])[0]

        probabilities = self.model.predict_proba([text])[0]

        confidence = max(probabilities)

        return {
            "category": str(prediction),
            "confidence": round(float(confidence), 3)
        }


if __name__ == "__main__":

    classifier = ResearchClassifier()
    classifier.train()

    examples = [
        "The model requires 24GB GPU memory",
        "The model supports external function calls",
        "Inference performance reaches 50 tokens per second",
        "Multiple agents coordinate their tasks"
    ]

    for example in examples:
        result = classifier.predict(example)

        print(f"\nText: {example}")
        print(f"Prediction: {result}")