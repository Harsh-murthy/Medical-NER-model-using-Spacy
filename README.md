# Medical NER Project using SpaCy

This project trains a Named Entity Recognition (NER) model to identify medical entities such as diseases, symptoms, and medications from clinical text using SpaCy.

## 📁 Dataset
The dataset used was from (https://www.kaggle.com/datasets/finalepoch/medical-ner) annotated with medical entity labels.

## 🧠 Model
The model was trained using SpaCy's pipeline with custom-labeled entities. Loss decreased steadily over training epochs, indicating learning.

## 🔬 Example Inference
```python
import spacy
nlp = spacy.load("model")
doc = nlp("Patient was prescribed paracetamol for high fever.")
for ent in doc.ents:
    print(ent.text, ent.label_)
