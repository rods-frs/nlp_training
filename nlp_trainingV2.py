#packages
import spacy
from spacy.training import Example
from os import system
#from os import system
from time import sleep
import csv
import random

"""
NLP Training Script V2

This script provides functionality for training and testing NLP models using spaCy.
It includes tools for Named Entity Recognition (NER) and text categorization (textcat).
The main workflow involves loading training data from CSV files, creating models,
training them, and testing their performance.
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
    print(path)
    try:
        with open(path, newline="") as f: 
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

def ner_treat_training_csv(reader):
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
                (phrase, {"entities": [(highlight_start, highlight_end, "MUSIC_NAME")]})
            )
        return training_data
    except Exception as e:
        print(f"Error in NER treat training CSV: {e}")
        return []

def ner_model_training(nlp, number_of_interactions, training_data):

    #Patience limit removed because the NER model training losses was too variable, making the patience stop the training too early

    try:
        print("Starting model training")
        nlp.initialize()
        losses = {}

        #patience = 30
        #last_lost = float("inf")

        #main training

        current_iteration = 0
        percentage = 100 / number_of_interactions
        current_percentage = 0

        for _ in range(number_of_interactions):

            current_iteration += 1

            #if patience <= 0:
            #    print("Patience limit reached! Ending training... ")
            #    break

            random.shuffle(training_data)

            losses = {}

            for text, annotations in training_data:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
            nlp.update([example], losses=losses, drop=0.2)

            display_perc = int(current_percentage)
            system("cls")
            print(f"Training {display_perc}% completed")

            current_percentage += percentage
            current_loss = losses["ner"]
            print(f"Current loss count: {current_loss:.6f}")
            
            #if current_loss < last_lost:
            #    last_lost = current_loss
            #    patience = 10
            #else:
            #    patience -= 1
            print(f"Iteration number {current_iteration}")

        print("Training completed!")
        return nlp
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None

def ner_model_testing(nlp, reader):

    failures = 0
    fail_highlights = []

    for row in reader:
        phrase = row["phrase"]
        highlight = row["highlight"]
        doc = nlp(phrase)

        for entity in doc.ents:
            if entity.text != highlight:
                failures += 1
                fail_highlights.append(highlight)
    
    return failures, fail_highlights

def ner_main(number_of_interactions, training_csv_path, testing_csv_path):
    
    nlp = create_ner_model("MUSIC_NAME")
    training_data = ner_treat_training_csv(open_csv(training_csv_path))
    trained_nlp = ner_model_training(nlp, number_of_interactions, training_data)
    if trained_nlp is None:
        print("Training failed, aborting.")
        return
    failures, fail_list = ner_model_testing(trained_nlp, open_csv(testing_csv_path))

    fail_index = 0

    for fails in fail_list:
        fail_index += 1
        print(f"Fail number {fail_index} | Fail: {fails}")
    print(f"Total failures: {failures}")
    
    save_nlp = input("Want to save the model? Type anything if yes | Press ENTER without typing anything if no\n>> ")

    if save_nlp:
        trained_nlp.to_disk("NER_MODEL")

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
    print("Entering CSV treating...")
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
        print("Finished CSV treating")
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
        print("Starting model training")
        optimizer = nlp.begin_training()
        losses = {}

        patience = 10
        last_lost = float("inf")

        #main training

        for _ in range(number_of_interactions):

            if patience <= 0:
                print("Patience limit reached! Ending training... ")
                break

            random.shuffle(cat_training_data)

            losses = {}

            for text, annotations in cat_training_data:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                nlp.update([example], sgd=optimizer, losses=losses, drop=0.2)

            current_loss = losses["textcat_multilabel"]
            print(f"Current loss count: {current_loss}")
            
            if current_loss - last_lost < 0.001:
                patience += 1
            else: patience -= 1

        print("Training completed!")
        return nlp
    except Exception as e:
        print(f"Error during model training: {e}")
        return None

def test_cat_model(nlp, reader):
    """
    Tests the trained text categorization model using test data from a CSV reader.

    Args:
        nlp: The trained spaCy model.
        reader: A CSV reader object containing test data with columns: intention, true_phrase, false_phrase.

    Returns:
        dict: A dictionary with test results for each intention, including success flags and probabilities.
    """
    try:
        #the true and false phrase are phrases that you want to test if the NLP will correctly recognize
        #Returns the probability of truness and the result of the test
        
        test_results = {}

        for row in reader:
            intention = row["intention"]
            true_phrase = row["true_phrase"]
            false_phrase = row["false_phrase"]
            true_doc = nlp(true_phrase)
            false_doc = nlp(false_phrase)

            #True phrase verification
            true_phrase_probability = true_doc.cats[intention]
            if true_phrase_probability > 0.5:
                success_test_1 = True
            else:
                success_test_1 = False
                
            #False phrase verification
            false_phrase_probability = false_doc.cats[intention]
            if false_phrase_probability > 0.5:
                success_test_2 = False
            else:
                success_test_2 = True

            test_results[intention] = {
                "test_1" : success_test_1,
                "test_1_prob" : true_phrase_probability,
                "test_2" : success_test_2,
                "test_2_prob" : false_phrase_probability
            }
        
        return test_results
    except Exception as e:
        print(f"Error testing model: {e}")
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
        test_results = test_cat_model(trained_nlp, testing_reader)

        for intention, results in test_results.items():
            print(f"Intention: {intention}, Test 1: {results['test_1']}, Prob: {results['test_1_prob']} | Test 2: {results['test_2']}, Prob: {results['test_2_prob']}")

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

"""
try:
    training_csv = "/home/rodrigo/Documents/GitHub/nlp_training/spotify_all_intentions.csv"
    testing_csv = "/home/rodrigo/Documents/GitHub/nlp_training/spotify_test_phrases.csv"
    intention_list = [
        "resume_music",
        "next_track",
        "pause_music",
        "shuffle",
        "repeat",
        "get_current_music"
    ]

    train_model(intention_list, 100, training_csv, testing_csv)
except Exception as e:
    print(f"Error in main execution: {e}")
"""

ner_main(5000, "ner_train_augmented.csv", "ner_test_improved.csv")
