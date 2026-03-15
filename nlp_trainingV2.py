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

def create_cat_model(intention, not_intention):
    nlp = spacy.blank("en")
    textcat = nlp.add_pipe("textcat")
    textcat.add_label(intention)
    textcat.add_label(not_intention)
    return nlp

#the scripts bellow can be used to train a already trained NLP, just give the nlp of the loaded NLP

def cat_treat_training_csv(intention, not_intention, reader):
    cat_training_data = []
    for row in reader:
        for row in reader:
            phrase = row["phrase"]
            score = row["score"]
            if score == "0":
                cat_training_data.append((phrase, {"cats": {intention: 0.0, not_intention: 1.0}}))
            else:
                cat_training_data.append((phrase, {"cats": {intention: 1.0, not_intention: 0.0}}))
        return cat_training_data

def cat_model_training(nlp, cat_training_data, number_of_interactions):

    optimizer = nlp.begin_training()
    losses = {}
    fixed_percentage = 100 / number_of_interactions 
    not_int_current_percentage = 0

    #main training

    for _ in range(number_of_interactions):
        random.shuffle(cat_training_data)
        for text, annotations in cat_training_data:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            nlp.update([example], sgd=optimizer, losses=losses)

        system("clear")
        int_current_percentage = int(not_int_current_percentage)
        print(f"NLP training {int_current_percentage}% completed")
        not_int_current_percentage += fixed_percentage

    print("Training completed!")
    return nlp

def test_cat_model(intention, nlp, true_phrase, false_phrase):

    #the true and false phrase are phrases that you want to test if the NLP will correctly recognize
    #Returns the probability of truness and the result of the test

    for _ in range(2):

        #True phrase verification
        doc = nlp(true_phrase)
        true_phrase_probability = doc.cats[intention]
        if true_phrase_probability > 0.5:
            success_test_1 = True
        else:
            success_test_1 = False
        
        #False phrase verification
        doc = nlp(false_phrase)
        false_phrase_probability = doc.cats[intention]
        if false_phrase_probability > 0.5:
            success_test_2 = False
        else:
            success_test_2 = True
    
    return true_phrase_probability, success_test_1, false_phrase_probability, success_test_2

def train_blank_model(intention, not_intention, number_of_interactions, training_csv_path, true_phrase, false_phrase):

    reader = open_csv(training_csv_path)
    nlp = create_cat_model(intention, not_intention)
    cat_training_data = cat_treat_training_csv(intention, not_intention, reader)
    trained_nlp = cat_model_training(nlp, cat_training_data, number_of_interactions)
    
    true_phrase_probability, success_test_1, false_phrase_probability, success_test_2 = test_cat_model(intention, trained_nlp, true_phrase, false_phrase)

    print(f"Test 1: {success_test_1}, Prob: {true_phrase_probability} | Test 2: {success_test_2}, Prob: {false_phrase_probability}")

    question_save_model = int(input("Save model? 1- Yes | Anything else- NO: "))
    if question_save_model == 1:
        model_name = str(input("Please type the model name: "))
        save_model(trained_nlp, model_name)

def train_model(nlp, intention, not_intention, number_of_interactions, training_csv_path, true_phrase, false_phrase):

    reader = open_csv(training_csv_path)
    cat_training_data = cat_treat_training_csv(intention, not_intention, reader)
    trained_nlp = cat_model_training(nlp, cat_training_data, number_of_interactions)
    
    true_phrase_probability, success_test_1, false_phrase_probability, success_test_2 = test_cat_model(intention, trained_nlp, true_phrase, false_phrase)

    print(f"Test 1: {success_test_1}, Prob: {true_phrase_probability} | Test 2: {success_test_2}, Prob: {false_phrase_probability}")

    question_save_model = int(input("Save model? 1- Yes | Anything else- NO: "))
    if question_save_model == 1:
        model_name = str(input("Please type the model name: "))
        save_model(trained_nlp, model_name)

#dev template
csv_path = "/home/rodrigo/Documents/GitHub/nlp_training/spotify_resume_phrases.csv"

train_blank_model("play_music", "not_play_music", 200, csv_path, "please play my music", "What day is today?")