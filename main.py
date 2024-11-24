import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
import joblib
from colorama import Fore
import logging

logger = logging.basicConfig(level=logging.INFO)
VERBOSE: bool = False


class SpamClassifier:
    def __init__(self, data_path, params=None, model_path=None, test_size: int = 0.2):
        """
        Initializes the SpamClassifier class.

        :param data_path: Path to the CSV file containing the data.
        :param params: List of dictionaries containing parameters for the vectorizer.
        :param model_path: Path to the model file to load if it exists already.
        """
        self.df = pd.read_csv(data_path)
        # encode the categorical vars
        self.df["label"] = self.df["label"].map({"ham": 0, "spam": 1})
        self.X = self.df["Message"]
        self.y = self.df["label"]
        self.test_size = test_size
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=19
        )
        self.params = params or [
            {
                "ngram_range": (1, 1),
                "stop_words": "english",
                "max_df": 1.0,
                "min_df": 1,
            },
            {
                "ngram_range": (1, 2),
                "stop_words": "english",
                "max_df": 0.9,
                "min_df": 2,
            },
            {"ngram_range": (1, 3), "stop_words": None, "max_df": 1.0, "min_df": 1},
            {
                "ngram_range": (1, 2),
                "stop_words": "english",
                "max_df": 0.8,
                "min_df": 5,
            },
        ]
        self.vectorizer = None
        self.clf = None
        self.model_path = model_path
        # load model if path is provided
        if model_path and self._is_model_saved():
            self.load_model()

    def _is_model_saved(self):
        """
        Check if the model already exists in the specified path.
        """
        try:
            joblib.load(self.model_path)
            return True
        except (FileNotFoundError, EOFError):
            return False

    def train(self):
        """
        Trains the model with different vectorizer configurations, stores the results, and saves the best model.
        """
        results = []
        for i, param in enumerate(self.params):
            logging.info(
                f"{Fore.BLUE}Running experiment {i + 1}/{len(self.params)} with params: {param}{Fore.RESET}"
            )
            vectorizer = CountVectorizer(**param)
            X_train_vec = vectorizer.fit_transform(self.X_train)
            X_test_vec = vectorizer.transform(self.X_test)
            clf = MultinomialNB()
            clf.fit(X_train_vec, self.y_train)
            y_pred = clf.predict(X_test_vec)
            f1 = f1_score(self.y_test, y_pred, pos_label=1)
            results.append({"params": param, "f1_score": f1})
        # keep track of the results for benchmarking
        self.results_df = pd.DataFrame(results)
        if VERBOSE:
            logger.info(self.results_df)

        # get the best params a.k.a with the highest F1-score
        best_result = self.results_df.loc[self.results_df["f1_score"].idxmax()]
        self.vectorizer = CountVectorizer(**best_result["params"])
        self.clf = MultinomialNB()
        # train the model with the best params found
        X_train_vec = self.vectorizer.fit_transform(self.X_train)
        self.clf.fit(X_train_vec, self.y_train)
        self.save_model()

    def save_model(self):
        """
        Saves the trained model to the specified file.
        """
        if not self.clf:
            raise ValueError(
                f"{Fore.RED}Model is not trained yet. Please call train() first.{Fore.RESET}"
            )
        # save the models (classifier and vectorizer)
        joblib.dump((self.vectorizer, self.clf), self.model_path)
        if VERBOSE:
            logger.info(f"{Fore.BLUE}Model saved to {self.model_path}{Fore.RESET}")

    def load_model(self):
        """
        Loads a previously trained model from the specified file.
        """
        if self.model_path and self._is_model_saved():
            if VERBOSE:
                logger.info(
                    f"{Fore.BLUE}Loading model from {self.model_path}{Fore.RESET}"
                )
            self.vectorizer, self.clf = joblib.load(self.model_path)
            if VERBOSE:
                logger.info(f"{Fore.GREEN}[+] Model loaded successfully.{Fore.RESET}")
        else:
            logger.info(
                f"{Fore.BLUE}No saved model found. Proceeding with training.{Fore.RESET}"
            )

    def evaluate(self):
        """
        Evaluates the model with the best parameters.
        """
        if not self.clf:
            raise ValueError("Model is not trained yet. Please call train() first.")
        X_test_vec = self.vectorizer.transform(self.X_test)
        y_pred = self.clf.predict(X_test_vec)
        f1 = f1_score(self.y_test, y_pred, pos_label=1)
        if VERBOSE:
            logger.info(f"F1 Score on Test Data: {f1:.4f}")

    def inference(self, new_sentence):
        """
        Makes predictions on new sentences.

        :param new_sentence: The sentence to classify.
        :return: Predicted label and probability scores.
        """
        if not self.clf:
            raise ValueError(
                f"{Fore.RED}Model is not trained yet. Please call train() first.{Fore.RESET}"
            )
        new_sentence_vec = self.vectorizer.transform([new_sentence])
        predicted_label = self.clf.predict(new_sentence_vec)[0]
        predicted_prob = self.clf.predict_proba(new_sentence_vec)[0]
        result = {
            "sentence": new_sentence,
            "predicted_label": "Spam" if predicted_label == 1 else "Not Spam",
            "spam_probability": predicted_prob[1],
            "ham_probability": predicted_prob[0],
        }

        logger.info(result)
        return result

    def run_all_inferences(self, new_sentence):
        """
        Runs inference using all parameter sets from the results and returns a detailed summary.

        :param new_sentence: The sentence to classify.
        :return: DataFrame with inference results for each parameter set.
        """
        inference_results = []

        for result in self.results_df.to_dict("records"):
            param = result["params"]
            logger.info(f"Using parameters: {param}")
            vectorizer = CountVectorizer(**param)
            X_train_vec = vectorizer.fit_transform(self.X_train)
            clf = MultinomialNB()
            clf.fit(X_train_vec, self.y_train)
            new_sentence_vec = vectorizer.transform([new_sentence])
            predicted_label = clf.predict(new_sentence_vec)[0]
            predicted_prob = clf.predict_proba(new_sentence_vec)[0]
            inference_results.append(
                {
                    "params": param,
                    "predicted_label": "Spam" if predicted_label == 1 else "Not Spam",
                    "spam_probability": predicted_prob[1],
                    "ham_probability": predicted_prob[0],
                }
            )
        inference_results_df = pd.DataFrame(inference_results)
        logger.info(inference_results_df)
        return inference_results_df


if __name__ == "__main__":
    model_path = "best_spam_model.joblib"
    classifier = SpamClassifier("data/spam_data.csv", model_path=model_path)
    if not classifier._is_model_saved():
        classifier.train()
    classifier.evaluate()
    sentence = "Congratulations, you've won a free vacation!"
    classifier.inference(sentence)
    classifier.run_all_inferences(sentence)
