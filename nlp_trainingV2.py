#packages
import spacy
from spacy.training import Example
from spacy.scorer import Scorer
from os import system
from time import sleep
import csv
import random

"""
NLP Training Script V2

Create custom models using Spacy, using the pipelines NER and CAT. 
Only works with Python 3.12

"""

#base scripts

def open_csv(path):
    """
    Opens a CSV file and returns its contents as a list of dictionaries.

    Args:
        path (str): The file path to the CSV file.

    Returns:
        list: A list of dictionaries, where each dictionary represents a row in the CSV.
              Returns an empty list if an error occurs.
    """
    try:
        with open(path, newline="") as f: 
            print(f"Successfully readed: {path}")
            return list(csv.DictReader(f)) 

    except Exception as e:
        print(f"Error opening CSV file: {e}")
        return []

def save_model(nlp, nlp_name):
    """
    Saves the trained NLP model to disk.

    Args:
        nlp: The trained spaCy NLP model.
        nlp_name (str): The directory path where the model will be saved.

    Returns:
        None
    """
    try:
        nlp.to_disk(nlp_name)
        print("NLP Saved!")
    except Exception as e:
        print(f"Error saving model: {e}")

#NER Training INCOMPLETE

def create_ner_model(ner_label):
    """
    Creates a blank spaCy model with a Named Entity Recognition (NER) pipeline.

    Args:
        ner_label (str): The label for the NER entity to be recognized.

    Returns:
        nlp: The initialized spaCy model with NER pipeline, or None if an error occurs.
    """
    try:
        nlp = spacy.blank("en")
        ner = nlp.add_pipe("ner")
        ner.add_label(ner_label)
        return nlp
    except Exception as e:
        print(f"Error creating NER model: {e}")
        return None

def ner_treat_training_csv(label, reader):
    """
    Processes training data for NER from a CSV reader. (Currently incomplete)

    Args:
        reader: A CSV reader object containing training data.

    Returns:
        list: A list of processed training data (currently empty).
    """
    try:
        training_data = []
        for row in reader:
            phrase = row["phrase"]
            highlight_object = row["highlight"]

            highlight_lengh = len(highlight_object)
            highlight_start = phrase.find(highlight_object)
            highlight_end = highlight_lengh + highlight_start

            training_data.append(
                (phrase, {"entities": [(highlight_start, highlight_end, label)]})
            )
        return training_data
    except Exception as e:
        print(f"Error in NER treat training CSV: {e}")
        return []

def ner_model_training(nlp, number_of_interactions, training_data):
    try:
        print("Starting model training")
        nlp.initialize()
        losses = {}

        patience = 30
        last_avg = float("inf")
        loss_history = []
        window = 20

        current_iteration = 0
        percentage = 100 / number_of_interactions
        current_percentage = 0

        for _ in range(number_of_interactions):
            current_iteration += 1

            if patience <= 0:
                print("Patience limit reached! Ending training...")
                break

            random.shuffle(training_data)
            losses = {}

            for text, annotations in training_data:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                nlp.update([example], losses=losses, drop=0.2)

            current_loss = losses["ner"]
            loss_history.append(current_loss)

            if len(loss_history) > window:
                loss_history.pop(0)

            avg_loss = sum(loss_history) / len(loss_history)

            display_perc = int(current_percentage)
            current_percentage += percentage

            system("clear")
            print(f"Training {display_perc}% | Iteration {current_iteration}")
            print(f"Loss: {current_loss:.6f} | Avg({window}): {avg_loss:.6f} | Patience: {patience}")

            if len(loss_history) == window:
                if last_avg - avg_loss < 0.0001:
                    patience -= 1
                #else: #uncomment for a better model but more time training
                #    patience = 30
                last_avg = avg_loss

        print("Training completed!")
        return nlp
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None
    
def ner_model_testing(nlp, reader, label):
    """
    Tests the trained NER model using test data and computes evaluation metrics.

    Args:
        nlp: The trained spaCy model.
        reader: List of dictionaries containing test data.
        label: The NER label to evaluate.

    Returns:
        dict: Dictionary of evaluation scores (precision, recall, F1, etc.).
    """
    try:
        scorer = Scorer()
        for row in reader:
            phrase = row["phrase"]
            highlight = row["highlight"]
            highlight_start = phrase.find(highlight)
            highlight_end = highlight_start + len(highlight)
            
            doc = nlp(phrase)
            gold_entities = [(highlight_start, highlight_end, label)]
            example = Example.from_dict(doc, {"entities": gold_entities})
            scorer.score_spans(example, "ner")
        
        scores = scorer.scores
        return scores
    except Exception as e:
        print(f"Error in NER testing: {e}")
        return {}

def ner_main(label, number_of_interactions, training_csv_path, testing_csv_path):
    
    nlp = create_ner_model(label)
    training_data = ner_treat_training_csv(label, open_csv(training_csv_path))
    trained_nlp = ner_model_training(nlp, number_of_interactions, training_data)
    if trained_nlp is None:
        print("Training failed, aborting.")
        return
    scores = ner_model_testing(trained_nlp, open_csv(testing_csv_path), label)

    print("NER Evaluation Scores:")
    for key, value in scores.items():
        print(f"{key}: {value:.4f}")
    
    save_nlp = input("Want to save the model? Type anything if yes | Press ENTER without typing anything if no\n>> ")

    if save_nlp:
        trained_nlp.to_disk("NER_MODEL_2")
        print("Model saved!")

#CAT Training

def create_cat_model(intention_list):
    """
    Creates a blank spaCy model with a text categorization (textcat) pipeline for multilabel classification.

    Args:
        intention_list (list): A list of intention labels for categorization.

    Returns:
        nlp: The initialized spaCy model with textcat pipeline, or None if an error occurs.
    """
    try:
        nlp = spacy.blank("en")
        textcat = nlp.add_pipe("textcat_multilabel")
        for intention in intention_list:
            textcat.add_label(intention)
        textcat.add_label("none") 
        return nlp
    except Exception as e:
        print(f"Error creating cat model: {e}")
        return None

def cat_treat_training_csv(intention_list, reader):
    """
    Processes training data for text categorization from a CSV reader.

    Args:
        intention_list (list): A list of intention labels.
        reader: A CSV reader object containing training data with columns: phrase, intention, score.

    Returns:
        list: A list of tuples, each containing (phrase, {"cats": cats_dict}).
    """
    cat_training_data = []
    try:
        for row in reader:  # ← removed the redundant outer intention loop
            phrase = row["phrase"]
            csv_intention = row["intention"]
            score = int(row["score"])  # ← cast to int

            cats = {}
            for intention in intention_list:
                cats[intention] = 1.0 if intention == csv_intention and score == 1 else 0.0
            cats["none"] = 1.0 if score == 0 else 0.0

            cat_training_data.append((phrase, {"cats": cats}))
        return cat_training_data
    
    except Exception as e:
        print(f"Training failed! Error: {e}")
        return []
    
def cat_model_training(nlp, cat_training_data, number_of_interactions):
    """
    Trains the text categorization model using the provided training data.

    Args:
        nlp: The initialized spaCy model with textcat pipeline.
        cat_training_data (list): List of training examples as (text, annotations) tuples.
        number_of_interactions (int): Number of training epochs.

    Returns:
        nlp: The trained spaCy model, or None if an error occurs.
    """
    try:
        print("Starting model training, this can take a while.")
        optimizer = nlp.begin_training()
        losses = {}

        patience = 10
        last_loss = float("inf")
        current_loss = float("inf")
        first_epoch = True

        for _ in range(number_of_interactions):
            if patience <= 0:
                print("Patience limit reached! Ending training...")
                break

            random.shuffle(cat_training_data)
            last_loss = current_loss
            losses = {}

            for text, annotations in cat_training_data:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                nlp.update([example], sgd=optimizer, losses=losses, drop=0.2)

            current_loss = losses["textcat_multilabel"]
            system("clear")

            if first_epoch:
                print(f"Loss: {current_loss:.4f} | Last: N/A | Patience: {patience}")
                first_epoch = False

            elif last_loss - current_loss < 0.001:
                patience -= 1                
                print(f"Loss: {current_loss:.4f} | Last: {last_loss:.4f} | Patience: {patience}")
                print("Not enough difference between errors, -1 patience point")
                patience -= 1

            else: 
                print(f"Loss: {current_loss:.4f} | Last: {last_loss:.4f} | Patience: {patience}")
                

        print("Training completed!")
        return nlp
    except Exception as e:
        print(f"Error during model training: {e}")
        return None

def test_cat_model(nlp, reader, intention_list):
    """
    Tests the trained text categorization model and computes evaluation metrics.

    Args:
        nlp: The trained spaCy model.
        reader: List of dictionaries containing test data.
        intention_list: List of intention labels.

    Returns:
        dict: Dictionary of evaluation scores for textcat_multilabel.
    """
    try:
        examples = []
        for row in reader:
            intention = row["intention"]
            true_phrase = row["true_phrase"]
            false_phrase = row["false_phrase"]
            
            # Create gold cats for true phrase: intention=1, others=0, none=0
            true_cats = {int: 1.0 if int == intention else 0.0 for int in intention_list}
            true_cats["none"] = 0.0
            
            # Create gold cats for false phrase: all=0, none=1
            false_cats = {int: 0.0 for int in intention_list}
            false_cats["none"] = 1.0
            
            # Create examples
            true_doc = nlp(true_phrase)
            true_example = Example.from_dict(true_doc, {"cats": true_cats})
            examples.append(true_example)
            
            false_doc = nlp(false_phrase)
            false_example = Example.from_dict(false_doc, {"cats": false_cats})
            examples.append(false_example)
        
        scorer = Scorer()
        scorer.score_cats(examples, "cats", labels=intention_list + ["none"])
        scores = scorer.scores
        return scores
    except Exception as e:
        print(f"Error in CAT testing: {e}")
        return {}

def cat_main(intention_list, number_of_interactions, training_csv_path, testing_csv_path):
    """
    Orchestrates the complete model training and testing workflow.

    Args:
        intention_list (list): List of intention labels for the model.
        number_of_interactions (int): Number of training epochs.
        training_csv_path (str): Path to the training CSV file.
        testing_csv_path (str): Path to the testing CSV file.

    Returns:
        None
    """
    try:
        training_reader = open_csv(training_csv_path)
        nlp = create_cat_model(intention_list)
        if nlp is None:
            print("Failed to create model.")
            return
        cat_training_data = cat_treat_training_csv(intention_list, training_reader)
        trained_nlp = cat_model_training(nlp, cat_training_data, number_of_interactions)
        if trained_nlp is None:
            print("Failed to train model.")
            return
        
        testing_reader = open_csv(testing_csv_path)
        scores = test_cat_model(trained_nlp, testing_reader, intention_list)

        print("CAT Evaluation Scores:")
        for key, value in scores.items():
            print(f"{key}: {value:.4f}")

        save_model(trained_nlp, "spotify-v2")
        question_save_model = str(input("Save model? Y/N: "))
        if question_save_model == "y":
            model_name = str(input("Please type the model name: "))
            save_model(trained_nlp, model_name)
    except Exception as e:
        print(f"Error in train_model: {e}")

#dev template
"""
Development template: Example usage of the train_model function.
Loads training and testing data from CSV files, trains a model on Spotify-related intentions,
and optionally saves the trained model.
"""

try:
    training_csv = "spotify_training.csv"
    testing_csv = "spotify_test.csv"
    intention_list = [
        "resume_music",
        "next_track",
        "pause_music",
        "shuffle",
        "repeat",
        "get_current_music",
        "play_music",
        "play_playlist",
        "search_music",
        "search_playlist"
    ]

    cat_main(intention_list, 10, training_csv, testing_csv)
except Exception as e:
    print(f"Error in main execution: {e}")



# ner_main("playlist_name", 10, "playlist_ner_train.csv", "playlist_ner_test.csv")
