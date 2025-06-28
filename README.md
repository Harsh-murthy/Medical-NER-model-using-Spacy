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

doc = nlp_trained_model('''
The patient was prescribed Aspirin for their heart condition.
The doctor recommended Ibuprofen to alleviate the patient's headache.
The patient is suffering from diabetes, and they need to take Metformin regularly.
After the surgery, the patient experienced some post-operative complications, including infection.
The patient is currently on a regimen of Lisinopril to manage their high blood pressure.
The antibiotic course for treating the bacterial infection should be completed as prescribed.
The patient's insulin dosage needs to be adjusted to better control their blood sugar levels.
The physician suspects that the patient may have pneumonia and has ordered a chest X-ray.
The patient's cholesterol levels are high, and they have been advised to take Atorvastatin.
The allergy to penicillin was noted in the patient's medical history.
''')


spacy.displacy.render(doc, style="ent", jupyter=True)

