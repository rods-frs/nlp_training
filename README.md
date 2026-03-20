# NLP Training V2

A Python script for training and testing NLP models using spaCy, focused on text categorization for intent recognition.

## Overview

This project provides tools for:
- Loading training data from CSV files
- Creating and training spaCy models for text categorization (multilabel)
- Testing model performance
- Saving trained models

The main use case demonstrated is training a model to recognize Spotify control intentions from user phrases.

## Requirements

- Python 3.7 ~ 3.12
- spaCy
- Required spaCy model: `en_core_web_sm` (or similar English model)

Install dependencies:
```bash
pip install spacy
python -m spacy download en_core_web_sm
```

## File Structure

```
nlp_trainingV2.py          # Main training script
spotify_all_intentions.csv  # Training data (phrases with intentions and scores)
spotify_test_phrases.csv    # Test data (true/false phrases for each intention)
spotify_*.csv              # Various CSV files for different intentions
*_model/                   # Saved model directories
```

## Key Functions

### Data Loading
- `open_csv(path)`: Loads CSV data into a list of dictionaries

### Model Creation
- `create_cat_model(intention_list)`: Creates a blank spaCy model with textcat_multilabel pipeline
- `create_ner_model(ner_label)`: Creates a blank spaCy model with NER pipeline (incomplete)

### Data Processing
- `cat_treat_training_csv(intention_list, reader)`: Processes CSV data into training format for categorization
- `ner_treat_training_csv(reader)`: Processes CSV data for NER (incomplete)

### Training & Testing
- `cat_model_training(nlp, cat_training_data, number_of_interactions)`: Trains the categorization model
- `test_cat_model(nlp, reader)`: Tests the model on validation data

### Main Workflow
- `train_model(intention_list, number_of_interactions, training_csv_path, testing_csv_path)`: Orchestrates the complete training and testing process

## Usage

### Basic Training

```python
from nlp_trainingV2 import train_model

# Define intentions
intention_list = [
    "resume_music",
    "next_track", 
    "pause_music",
    "shuffle",
    "repeat",
    "get_current_music"
]

# Train model
train_model(
    intention_list=intention_list,
    number_of_interactions=100,
    training_csv_path="spotify_all_intentions.csv",
    testing_csv_path="spotify_test_phrases.csv"
)
```

### CSV Data Format

#### Training Data (spotify_all_intentions.csv)
```csv
phrase,intention,score
"play some music",resume_music,1
"stop the song",pause_music,1
"what's playing",get_current_music,1
"random phrase",none,0
```

#### Test Data (spotify_test_phrases.csv)
```csv
intention,true_phrase,false_phrase
resume_music,"start playing music","stop the music"
pause_music,"pause the song","play next track"
```

## Model Training Process

1. **Data Loading**: CSV files are read and parsed
2. **Model Creation**: Blank spaCy model with textcat_multilabel pipeline
3. **Data Preparation**: Training examples formatted for spaCy
4. **Training Loop**: 
   - Shuffles data each epoch
   - Updates model with examples
   - Tracks loss and implements early stopping with patience
   - Displays progress percentage
5. **Testing**: Evaluates model on test phrases
6. **Saving**: Optionally saves trained model to disk

## Error Handling

The code includes comprehensive try-except blocks to catch and display errors:
- File I/O errors
- Model creation failures
- Training errors
- Testing errors

## Notes

- The script uses hardcoded file paths in the development template - update these for your environment
- NER functionality is currently incomplete
- Training includes a patience mechanism to prevent overfitting
- Models are saved in spaCy's native format

## Output

During training, you'll see progress like:
```
NLP training 25% completed | Current loss count: 0.123
NLP training 50% completed | Current loss count: 0.089
...
Training completed!
```

Test results show:
```
Intention: resume_music, Test 1: True, Prob: 0.87 | Test 2: True, Prob: 0.12
```</content>
<parameter name="filePath">c:\Users\rodrigo.fsilva61\Downloads\nlp_training-dev\nlp_training-dev\README.md