#packages
import spacy
from spacy.tokens import DocBin
from spacy.training import Example
import csv
import logging
import random
import nltk
import nlpaug.augmenter.word as naw

#libraries configuration

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler("treinamento.log"),
        logging.StreamHandler()                
    ]
)

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')
aug = naw.SynonymAug(aug_src='wordnet')

#Basic scripts

def create_blank_model():
    nlp = spacy.blank("en")
    return nlp

def open_csv(csv_path):
    try:
        logging.debug(f"Trying to open {csv_path}")
        with open(csv_path, newline="") as f: 
            reader = list(csv.DictReader(f))
            columns = list(reader[0].keys())
            logging.info(f"Readed {csv_path}!")
            return reader, columns
    except Exception as e:
        raise(logging.error(f"Error while opening the CSV: {e}"))

def save_docbin(docbin, save_path):
    try:
        logging.info(f"Saving DocBin to the path {save_path}")
        docbin.to_disk(save_path)
    except Exception as e:
        logging.error(f"Error while saving DocBin to {save_path}: {e}")

def get_doc_labels(db, vocab):
    try:
        labels = []
        for doc in db.get_docs(vocab):
            for ent in doc.ents:
                if ent.label_ not in labels:
                    labels.append(ent.label_)
        return labels
    except Exception as e:
        logging.error(f"Error while getting the labels from the doc: {e}")

#Data treatment for NER and CAT models

def treat_ner_csv(nlp, csv_path):
    try:
        logging.debug("Opening DocBin")
        ner_db = DocBin()

        expected_columns = ["phrase", "word", "label"]
        raw_csv, csv_columns = open_csv(csv_path)

        logging.debug("Verifing columns")
        for column in csv_columns:
            if column not in expected_columns:
                raise ValueError(f"The column {column} was wrote wrong or was not supposed to be here...")

        for row in raw_csv:

            logging.debug("Extracting data from CSV")
            phrase = row["phrase"]
            word = row["word"]
            phrase_label = row["label"]

            doc = nlp.make_doc(phrase)
            word_lengh = len(word)
            word_start = phrase.find(word)
            word_end = word_lengh + word_start

            aug_phrase = aug.augment(phrase, stopwords=[word])[0]
            aug_doc = nlp.make_doc(aug_phrase)
            aug_word_start = aug_phrase.find(word)
            aug_word_end = word_lengh + aug_word_start

            logging.debug("Creating span")
            span = doc.char_span(word_start, word_end, label=phrase_label)
            aug_span = aug_doc.char_span(aug_word_start, aug_word_end, label=phrase_label)

            if span is not None:
                doc.ents = [span]
                ner_db.add(doc)
            else:
                logging.error(f"Falha ao alinhar palavra {word} da phrase original")

            if aug_span is not None:
                aug_doc.ents = [aug_span]
                ner_db.add(aug_doc)

            else:
                logging.error(f"Falha ao alinhar palavra {word} da phrase augumentada")

        return ner_db, nlp

    except Exception as e:
        logging.error(f"Error while treating the data from the CSV: {e}")

def treat_cat_csv(nlp, csv_path):

    logging.info("Starting data treating... This should not take a lot of time")

    """
    The first line of the CSV needs to have the row "phrase" to be the number of labels that the model will have
    After the first line, the phrase row needs to be the name of a label until the line in the first row.
    Example:
    phrase, intention
    3,
    age,
    name,
    location,
    my name is elison, name
    im 10 years old, age
    im in brazil, location
    """

    idx = 1
    label_numbers = 0
    label_list = []
    cat_db = DocBin()

    try:
        logging.debug("Checking columns of the csv...")
        expected_columns = ["phrase", "intention"]
        raw_csv, csv_columns = open_csv(csv_path)
        for column in (csv_columns):
            if column not in expected_columns:
                raise ValueError(f"The column {column} was wrote wrong or was not supposed to be here...")
    except Exception as e:
        logging.error(f"Error while checking the columns: {e}")

    logging.debug("Starting CSV treating...")

    try:
        for row in raw_csv:
            phrase = row["phrase"]
            intention = row["intention"]

            if idx == 1:
                logging.debug("First CSV line detected! Checking label number...")
                phrase = int(phrase)
                label_numbers = phrase
            else:
                if idx-1 <= label_numbers:
                    logging.debug(f"Getting the label {idx-1} out of {label_numbers}")
                    label_list.append(phrase)
                
                else:
                    logging.debug("Starting to fill cat_db...")

                    #unagumented phrase
                    logging.debug("Making doc for unagumented phrase")
                    doc = nlp.make_doc(phrase)
                    logging.debug("Putting labels on the row...")
                    doc.cats = {label: 0.0 for label in label_list}
                    if intention in doc.cats:
                        doc.cats[intention] = 1.0
                    cat_db.add(doc)

                    #augumented phrase
                    try:
                        logging.debug("Making doc for agumented phrase")
                        aug_phrase = aug.augment(phrase)[0]
                        aug_doc = nlp.make_doc(aug_phrase)
                        logging.debug("Putting labels on the row...")
                        aug_doc.cats = {label: 0.0 for label in label_list}
                        if intention in aug_doc.cats:
                            aug_doc.cats[intention] = 1.0
                        cat_db.add(aug_doc)
                    except Exception as e:
                        logging.error(f"Error while creating the augumented phrase: {e}")
            idx += 1
        logging.info("Data treated! Returning gattered data...")
        return cat_db, nlp, label_list
            
    except Exception as e:
        logging.error(f"Couldn't treat the data: {e}")

#Training models

def train_ner_model(nlp, db, iterations, window, patience):

    try:

        #preparation for the pacience system
        loss_history = []
        last_average = None

        logging.debug("Creating blank model")
        nlp = create_blank_model()
        ner = nlp.add_pipe("ner")

        logging.debug("Getting labels")
        labels = get_doc_labels(db, nlp.vocab)
        for label in labels:
            ner.add_label(label)

        logging.debug("Opening DocBin")
        train_data = list(db.get_docs(nlp.vocab))

        logging.debug("Starting training")
        optimizer = nlp.begin_training()

        epoches = 0
        for epoch in range(iterations):
            random.shuffle(train_data)
            losses = {}
            epoches += 1

            for doc in train_data:
                example = Example.from_dict(nlp.make_doc(doc.text), {"entities": [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]})
                nlp.update([example], sgd=optimizer, losses=losses)

            logging.debug("Executing patience calculations")
            loss_history.append(losses['ner'])
            if len(loss_history) > window:
                current_average = sum(loss_history) / len(loss_history)
                if last_average is None:
                    logging.debug("First average detected! Creating last_average")
                    last_average = current_average
                else:
                    if last_average - current_average < 0.0001:
                        patience -= 1
                        logging.info(f"No relevant change. Patience lost 1 point, {patience} points left")
                    last_average = current_average
                loss_history = []

            logging.info(f"Epoch number {epoches} | {iterations - epoches} to go.")

            if patience == 0:
                logging.info("Patience limit reached! Finishing training...")
                break
        
        logging.info("Training finished! Returning training NLP...")
        return nlp

    except Exception as e:
        logging.error(f"Error while training the NER model: {e}")
        
def train_cat_model(nlp, iterations, window, patience, csv_path):

    logging.debug("Starting model training function.")

    try:
        
        last_average = None
        loss_history = []

        cat = nlp.add_pipe('textcat')
        
        cat_db, nlp, label_list = treat_cat_csv(nlp, csv_path)
        for label in label_list:
            cat.add_label(label)
        
        logging.info("Data fed! Starting training...")
        train_data = list(cat_db.get_docs(nlp.vocab))

        optimizer = nlp.begin_training()
        epoches = 0

        logging.debug("Starting training loop")
        for epoches in range(iterations):
            logging.debug(f"Making epoch number {epoches}")
            random.shuffle(train_data)
            losses = {}
            epoches += 1

            logging.debug("Starting secondery loop of the training...")
            for doc in train_data:
                example = Example.from_dict(nlp.make_doc(doc.text), {"cats": doc.cats})
                nlp.update([example], sgd=optimizer, losses=losses)
            
            logging.debug("Executing patience calculations")
            loss_history.append(losses['textcat'])
            if len(loss_history) > window:
                current_average = sum(loss_history) / len(loss_history)
                if last_average is None:
                    logging.debug("First average detected! Creating last_average")
                    last_average = current_average
                else:
                    if last_average - current_average < 0.0001:
                        patience -= 1
                        logging.info(f"No relevant change. Patience lost 1 point, {patience} points left")
                    last_average = current_average
                loss_history = []

            logging.info(f"Epoch number {epoches} | {iterations - epoches} to go.")

            if patience == 0:
                logging.info("Patience limit reached! Finishing training...")
                break

        logging.info("Training finished! Returning NLP...")
        return nlp

    except Exception as e:
        logging.error(f"Failed to train the textcat model: {e}")

#Model testing

def test_ner_model():
    pass

def test_cat_model(nlp):

    test_phrases_dictionary = {
        "test1": {
            "phrase": "Hey, can you turn on my computer for me?",
            "label": "start_computer"
        },
        "test2": {
            "phrase": "What is the temperature outside right now?",
            "label": "weather_inquiry"
        },
        "test3": {
            "phrase": "Start up my PC, I need to get to work.",
            "label": "start_computer"
        },
        "test4": {
            "phrase": "Is it going to rain this afternoon?",
            "label": "weather_inquiry"
        },
        "test5": {
            "phrase": "Wake up the workstation and boot the system.",
            "label": "start_computer"
        }
    }

    fail_list = []
    success_list = []

    for key, content in test_phrases_dictionary.items():
        phrase = content["phrase"]
        label = content["label"]
        doc = nlp(phrase)
        if doc.cats[label] > 0.5:
            success_list.append(phrase)
        else:
            fail_list.append(phrase)
    
    return fail_list, success_list

#Main 

def main_ner_training(CSV_FILE, iterations, window, patience, test_phrases):
    nlp_init = create_blank_model()
    DB_FILE = treat_ner_csv(nlp_init, CSV_FILE)

    trained_nlp = train_ner_model(DB_FILE, iterations, window, patience)

    print("\n" + "="*40)
    print(" NER MODEL RESULTS ")
    print("="*40)
    

    for phrase in test_phrases:
        doc = trained_nlp(phrase)
        entidades = [(ent.text, ent.label_) for ent in doc.ents]
        print(f"Input: '{phrase}'")
        print(f"Detected: {entidades if entidades else 'No entity found'}")
        print("-" * 40)

def main_cat_training(csv_path, iterations, window, patience):
    nlp = create_blank_model()
    trained_nlp = train_cat_model(nlp, iterations, window, patience, csv_path)
    fail_list, success_list = test_cat_model(trained_nlp)
    logging.info("Training was a success! Testing it and showing results...")
    print("Fail list:")
    for fail in fail_list:
        print(fail)
    print("Success list:")
    for success in success_list:
        print(success)

main_cat_training(r"C:\Users\rodrigo.fsilva61\Documents\GitHub\nlp_training\textcat_training_data.csv", 50, 10, 10)