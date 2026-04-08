import logging
from training_v3 import create_blank_model, treat_ner_csv, train_ner_model, save_docbin

# Configuração básica de log para ver o progresso no terminal
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- CONFIGURAÇÕES DO TESTE ---
CSV_FILE = "/home/rsilva/Documents/nlp_training/Training V3/teste_ner.csv" 

def executar_teste():
    # 1. Preparar o terreno (Gera o DocBin)
    logging.info("Iniciando tratamento de dados...")
    nlp_init = create_blank_model()
    DB_FILE = treat_ner_csv(nlp_init, CSV_FILE)

    # 2. Treinar o modelo
    # Parâmetros: iterations=100, window=5, patience=3
    logging.info("Iniciando treinamento...")
    nlp_treinado = train_ner_model(DB_FILE, 100, 5, 3)

    # 3. Testar a inteligência da Jose
    print("\n" + "="*40)
    print(" RESULTADOS DO MODELO NER ")
    print("="*40)
    
    frases_para_testar = [
        "Joseh, tocar Linkin Park por favor",
        "Pode pausar a musica agora",
        "Eu quero ouvir Taylor Swift"
    ]

    for frase in frases_para_testar:
        doc = nlp_treinado(frase)
        entidades = [(ent.text, ent.label_) for ent in doc.ents]
        print(f"Entrada: '{frase}'")
        print(f"Detectado: {entidades if entidades else 'Nenhuma entidade encontrada'}")
        print("-" * 40)

if __name__ == "__main__":
    executar_teste()