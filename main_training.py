#packages
import spacy
from spacy.training import Example
from os import system
from time import sleep
import csv
import threading
import random

#main functions

def open_csv(path):
    with open(path, newline="") as f: 
        return list(csv.DictReader(f))

#CAT training

def cat_analytics():
    pass

def c_cat_model(it, n_it):
    nlp = spacy.blank("en")
    textcat = nlp.add_pipe("textcat")
    textcat.add_label(it)
    textcat.add_label(n_it)
    return nlp

def cat_get_td(it1, n_it1, reader):
    cat_td = []
    for row in reader:
        phrase = row["phrase"]
        score = row["score"]
        if score == "0":
            cat_td.append((phrase, {"cats": {it1: 0.0, n_it1: 1.0}}))
        else:
            cat_td.append((phrase, {"cats": {it1: 1.0, n_it1: 0.0}}))
    return cat_td

def cat_training(nlp, cat_td, interations):
    optimizer = nlp.begin_training()
    losses = {}
    fix_perc = 100 / interations
    perc = 0
    int_n = 0
    for _ in range(interations):
        ex_n = 0
        random.shuffle(cat_td)
        for text, annotations in cat_td:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            nlp.update([example], sgd=optimizer, losses=losses)
            ex_n += 1
        system("clear")
        int_perc = int(perc)
        print(f"NLP training {int_perc}% completed")
        perc += fix_perc
        int_n += 1
    print("Training completed!")
    return nlp

def b_check_cat_nlp(intention, trained_nlp, t_text, f_text):

    for _ in range(2):

        #True phrase verification
        doc = trained_nlp(t_text)
        t_it_prob1 = doc.cats[intention]
        if t_it_prob1 > 0.5:
            test1 = True
        else:
            test1 = False
        
        #False phrase verification
        doc = trained_nlp(t_text)
        f_it_prob1 = doc.cats[intention]
        if f_it_prob1 > 0.5:
            test2 = False
        else:
            test2 = True
    
    return t_it_prob1, test1, f_it_prob1, test2

#testing

it1 = "update"
n_it1 = "n_update"
interations = 200
intention = "update"
t_text = "please update my computer"
f_text = "what date is today?"

b_nlp = c_cat_model(it1, n_it1)

reader = open_csv("/home/morsdesuper/Documents/GitHub/nlp_training/update_t_data.csv")

cat_td = cat_get_td(it1, n_it1, reader)

trained_nlp = cat_training(b_nlp, cat_td, interations)

t_prob, test1, f_prob, test2 = b_check_cat_nlp(intention, trained_nlp, t_text, f_text)

print(t_prob, test1, f_prob, test2)

