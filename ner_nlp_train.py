import spacy
import csv
from spacy.training import Example
from os import system

def open_t_data_csv(csv_path):
    with open(csv_path, newline="") as f: 
        return list(csv.DictReader(f))

def create_model(it):
    nlp = spacy.blank("en")
    ner = nlp.add_pipe("ner")
    ner.add_label(it)
    return nlp

def training_data(nlp, reader):
    master_list = []

    for row in reader:
        phrase = row["phrase"]
        p_name = row["p_name"]

        start = phrase.find(p_name)
        if start == -1:
            print(f"WARNING: '{p_name}' not found in '{phrase}'")
            continue

        end = start + len(p_name)

        master_list.append(
            (phrase, {"entities": [(start, end, p_name)]})
        )

    for idx in master_list:
        print(idx)
    return master_list

    return master_list

def train_model(nlp, training_data, interations):
    optimizer = nlp.begin_training()
    examples = []
    losses = {}
    n_entr = 1
    fix_perc = 100 / interations
    perc = 0

    for text, annotations in training_data:
        doc = nlp.make_doc(text)
        examples.append(Example.from_dict(doc, annotations))

    for epoch in range(interations):
        nlp.update(examples, sgd=optimizer, losses=losses)
        system("clear")
        print(f"NLP training {perc}% completed, total losses{losses}")
        perc += fix_perc
    print("Training completed!")

    return nlp

def get_ent_name(nlp, ent, txt):
    doc = nlp(txt)
    for e in doc.ents:
        if e.label_ == ent:
            return e.text
    return None


def main(ent, path, ent_name, true_t_phrase, false_t_phrase):
    nlp = create_model(ent)
    reader = open_t_data_csv(path)
    train_data = training_data(nlp, reader)
    final_nlp = train_model(nlp, train_data, 200)

    t_nlp_ent_name = get_ent_name(final_nlp, ent, true_t_phrase)
    f_nlp_ent_name = get_ent_name(final_nlp, ent, false_t_phrase)

    if t_nlp_ent_name == ent_name:
        print(f"NLP recognized the entity name: {f_nlp_ent_name}")
    else:
        print(f"NLP failed to recognized the entity name. Name gave: {ent_name}, Name returned: {f_nlp_ent_name}")
    
    print(f_nlp_ent_name)

main("program", "/home/morsdesuper/Documents/joseh/Joseh/open_p_t_data.csv", "chrome", "Please open chrome", "dont open anything")
