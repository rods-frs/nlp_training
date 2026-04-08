#packages
import spacy
from spacy.training import Example
from os import system
from time import sleep
import csv
import random

#main functions

def open_csv(path):
    print(path)
    with open(path, newline="") as f: 
        return list(csv.DictReader(f))

#NER Training

def b_ner_model(ner_label):
    nlp = spacy.blank("en")
    ner = nlp.add_pipe("ner")
    ner.add_label(ner_label)
    return nlp

def ner_treat_td(reader):
    master_list = []
    for row in reader:
        phrase = row["phrase"]
        p_name = row["pname"]

        ph_len = len(p_name)
        p_start = phrase.find(p_name)
        p_end = ph_len + p_start

        master_list.append(
            (phrase, {"entities": [(p_start, p_end, "PROGRAM")]})
        )
    return master_list

def ner_training(nlp, int_range, TRAIN_DATA):

    # 5. Initialize the model
    nlp.initialize()

    perc = 100 / int_range
    
    current_perc = 0

    # 6. Training loop
    for i in range(int_range):  # small number of iterations for demo
        losses = {}
        for text, annotations in TRAIN_DATA:
            example = Example.from_dict(nlp.make_doc(text), annotations)
            nlp.update([example], losses=losses)
        display_perc = int(current_perc)
        system("clear")
        print(f"Training {display_perc}% completed")
        current_perc += perc
    
    print("Training completed!")
    return nlp

def ner_test(nlp, t_csv, ):
    failures = 0
    fail_name = []
    success = 0
    suc_name = []
    row_counter = 0

    reader = open_csv(t_csv)
    for row in reader:
        row_counter += 1
        phrase = row["phrase"]
        p_name = row["name"]
        doc = nlp(phrase)
        for ent in doc.ents:
            if ent.text != p_name:
                failures += 1
                fail_name.append(p_name)
            elif ent.text == p_name:
                success += 1
                suc_name.append(p_name)
    for row in reader:
        p_name = row["name"]
        if p_name not in suc_name:
            fail_name.append(p_name)
    print("=" * 10)
    total_rows = success + failures

    return failures, fail_name, success, suc_name, total_rows, row_counter


#CAT training

def cat_analytics():
    pass

def create_cat_model(it, n_it):
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
            test1 = "Passed"
        else:
            test1 = "Failed"
        
        #False phrase verification
        doc = trained_nlp(f_text)
        f_it_prob1 = doc.cats[intention]
        if f_it_prob1 > 0.5:
            test2 = "Failed"
        else:
            test2 = "Passed"
    
    return t_it_prob1, test1, f_it_prob1, test2

#testing

def cat_main(it1, n_it1, interations, t_text, f_text, cat_csv_path):

    b_nlp = create_cat_model(it1, n_it1)

    reader = open_csv(cat_csv_path)

    cat_td = cat_get_td(it1, n_it1, reader)

    trained_nlp = cat_training(b_nlp, cat_td, interations)

    t_prob, test1, f_prob, test2 = b_check_cat_nlp(it1, trained_nlp, t_text, f_text)

    print(f"True phrase test: {test1} with the prob being: {t_prob} | False phrase test: {test2} with the prob being: {f_prob} ")
    
    input("Press ENTER to save the model as 'open_program_model'")
    trained_nlp.to_disk("open_program_model")

def ner_main(label, csv_path, t_csv, n_int):

    b_nlp = b_ner_model(label)

    reader = open_csv(csv_path)

    ner_training_data = ner_treat_td(reader)

    t_nlp = ner_training(b_nlp, n_int, ner_training_data)

    failures, fail_name, success, suc_name, total_rows, row_counter = ner_test(t_nlp, t_csv)

    if failures > 0:
        print(f"{failures} programs from the testing CSV where not recognized")
        print(f"This programs where: ")
        for name in fail_name:
            print(name)
        
        while True:
            ex_option = str(input("Would you like to export the trained NLP? ")).lower()
            if ex_option == "yes":
                nlp_name = input("Whats the name of the NLP? ")
                t_nlp.to_disk(nlp_name)
                break
            elif ex_option == "no":
                print("Sure thing, exiting...")
                break
            else:
                print(f"{ex_option} is not a known command... ")

    else: t_nlp.to_disk("SUCCESS_NLP001")

if __name__ == "__main__":
    #ner_main("PROGRAM", "/home/morsdesuper/Documents/GitHub/nlp_training/program_phrases_mixed.csv", "csv.csv", 250)
    cat_main("open_program", "not_open_program", 250, "Could you please open up Word?", "What time is it?", "/home/morsdesuper/Documents/GitHub/nlp_training/update_t_data.csv")

    