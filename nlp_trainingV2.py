#packages
import spacy
from spacy.training import Example
from os import system
#from os import system
from time import sleep
import csv
import random

#base scripts

def open_csv(path):
    print(path)
    with open(path, newline="") as f: 
        return list(csv.DictReader(f)) 
        #returns the reader

def save_model(nlp, nlp_name):
    nlp.to_disk(nlp_name)
    print("NLP Saved!")

#NER Training INCOMPLETE

def create_ner_model(ner_label):
    nlp = spacy.blank("en")
    ner = nlp.add_pipe("ner")
    ner.add_label(ner_label)
    return nlp

def ner_treat_training_csv(reader):
    training_data = []
    for row in reader:
        phrase = row["phrase"]
        pass

#CAT Training

def create_cat_model(intention_list):
    nlp = spacy.blank("en")
    textcat = nlp.add_pipe("textcat_multilabel")
    for intention in intention_list:
        textcat.add_label(intention)
    textcat.add_label("none") 
    return nlp

def cat_treat_training_csv(intention_list, reader):
    print("Entering CSV treating...")
    cat_training_data = []
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

def cat_model_training(nlp, cat_training_data, number_of_interactions):

    print("Starting model training")
    optimizer = nlp.begin_training()
    losses = {}
    fixed_percentage = 100 / number_of_interactions 
    not_int_current_percentage = 0

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
        system("clear")
        int_current_percentage = int(not_int_current_percentage)
        print(f"NLP training {int_current_percentage}% completed | Current loss count: {current_loss}")
        not_int_current_percentage += fixed_percentage
        
        if current_loss - last_lost < 0.001:
            patience += 1
        else: patience -= 1

    print("Training completed!")
    return nlp

def test_cat_model(nlp, reader):

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

def train_model(intention_list, number_of_interactions, training_csv_path, testing_csv_path):

    training_reader = open_csv(training_csv_path)
    nlp = create_cat_model(intention_list)
    cat_training_data = cat_treat_training_csv(intention_list, training_reader)
    trained_nlp = cat_model_training(nlp, cat_training_data, number_of_interactions)
    
    testing_reader = open_csv(testing_csv_path)
    test_results = test_cat_model(trained_nlp, testing_reader)

    for intention, results in test_results.items():
        print(f"Intention: {intention}, Test 1: {results['test_1']}, Prob: {results['test_1_prob']} | Test 2: {results['test_2']}, Prob: {results['test_2_prob']}")

    question_save_model = str(input("Save model? Y/N: "))
    if question_save_model == "y":
        model_name = str(input("Please type the model name: "))
        save_model(trained_nlp, model_name)

#dev template
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