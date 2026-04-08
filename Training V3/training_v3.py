#packages
import spacy
from spacy.tokens import DocBin
import csv
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler("treinamento.log"),
        logging.StreamHandler()                
    ]
)

#Put this inside the NER main
ner_db = DocBin()

def open_csv(csv_path):
    try:
        logging.debug(f"Trying to open {csv_path}")
        with open(csv_path, newline="") as f: 
            reader = csv.DictReader(f)
            columns = reader.fieldnames
            logging.info(f"Readed {csv_path}!")
            return reader, columns
    except Exception as e:
        logging.error(f"Error while opening the CSV: {e}")

def treat_ner_csv(nlp, csv_path, label):

    #the expected columns for the NER training is: phrase, word, label. the incorrect name of the columns will result in the code failing.

    try:
        pass

        expected_columns = ["phrase", "word", "label"]
        raw_csv, csv_columns = open_csv(csv_path)

        for column, expected_column in csv_columns, expected_columns:
            if column != expected_column:
                raise ValueError(f"Column name expected: {expected_column}, got {column}")
        
        for row in raw_csv:

            phrase = row["phrase"]
            word = row["word"]
            phrase_label = row["label"]

            nlp.make_doc(phrase)

            word_lengh = len(word)
            word_start = phrase.find(word)
            word_end = word_lengh + word_start

            span = doc.char_span(word_start, word_end, label=phrase_label)

            if span is not None:
                doc.ents = [span]
                ner_db.add(doc)
            else:
                logging.warning(f"Falha ao alinhar palavra {word}")
        
    except Exception as e:
        logging.error(f"Error while treating the data from the CSV: {e}")
