import spacy

nlp = spacy.load("NER_MODEL")
doc = nlp("Just start playing Joias da Familia")
for ent in doc.ents:
    if ent.label_ == "MUSIC_NAME":
        print(ent.text)