#packages
import spacy
from spacy.tokens import DocBin
from spacy.training import Example
import csv
import logging
import random

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler("treinamento.log"),
        logging.StreamHandler()                
    ]
)


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
    labels = []
    for doc in db.get_docs(vocab):
        for ent in doc.ents:
            if ent.label_ not in labels:
                labels.append(ent.label_)
    return labels

#Data treatment for NER and CAT models

def treat_ner_csv(nlp, csv_path):

    try:
        logging.debug("Opening DocBin")
        ner_db = DocBin()

        expected_columns = ["phrase", "word", "label"]
        raw_csv, csv_columns = open_csv(csv_path)

        logging.debug("Verifing columns")
        for column, expected_column in zip(csv_columns, expected_columns):
            if column != expected_column:
                raise ValueError(f"Column name expected: {expected_column}, got {column}")
            
        for row in raw_csv:

            logging.debug("Extracting data from CSV")
            phrase = row["phrase"]
            word = row["word"]
            phrase_label = row["label"]

            doc = nlp.make_doc(phrase)

            word_lengh = len(word)
            word_start = phrase.find(word)
            word_end = word_lengh + word_start

            logging.debug("Creating span")
            span = doc.char_span(word_start, word_end, label=phrase_label)

            if span is not None:
                doc.ents = [span]
                ner_db.add(doc)
            else:
                logging.warning(f"Falha ao alinhar palavra {word}")

        return ner_db

    except Exception as e:
        logging.error(f"Error while treating the data from the CSV: {e}")

def treat_cat_csv(nlp, csv_path)
    pass

#Training models

def train_ner_model(db, iterations, window, patience):

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
                if not last_average:
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
        