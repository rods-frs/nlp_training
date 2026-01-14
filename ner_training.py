#imports
import spacy
import csv
from spacy.training import Example
from os import system

#main variables
csv_path = "/home/morsdesuper/Documents/joseh/Joseh/open_p_t_data.csv"
it = "program"

#start the model
blank_nlp = spacy.blank("en")
ner = nlp.add_pipe("ner")
ner.add_label(it)
interactions = 100

#get the csv with the phrases and the name of the program



with open(csv_path, newline="") as f: 
    reader = list(csv.DictReader(f))

#extract the csv and treat it

training_phrase = []

for row in reader:
    phrase = row["phrase"]
    p_name = row["p_name"]
    p_start = phrase.find(p_name)
    p_end = start + len(p_name)

    master_list.append(
        (phrase, {"entities": [(p_start, p_end, p_name)]})
    )

#train the model

optimizer = nlp.begin_training()
examples = []
losses = {}
n_entr = 1
fix_perc = 100 / interations
perc = 0

for text, annotations in master_list:
    doc = nlp.make_doc(text)
    examples.append(Example.from_dict(doc, annotations))

for epoch in range(interations):
    nlp.update(examples, sgd=optimizer, losses=losses)
    system("clear")
    print(f"NLP training {perc}% completed, total losses{losses}")
    perc += fix_perc
print("Training completed!")
