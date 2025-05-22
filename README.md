# AI-Powered NLP Assignment 

This project was developed to focus on using RNNs for text classification and word generation.


## Project Folder Structure

```
├── educational_dataset.csv
├── generate_word.py
├── inference.py
├── preprocess.py
├── README.md
├── scrap_wikipedia.py
├── text_generation_preprocess.py
├── train_rnn.py
├── train_text_generator.py

```
## Task 1: Text Classification with RNN (Mandatory)

### Objective
Classify short educational texts into one of the following categories:
- Math
- Science
- History
- English

### How to Run
```bash
python preprocess.py
python train_rnn.py
python inference.py
```


### Sample Input
> "As an academic discipline, it analyses and interprets evidence to construct narratives about what happened and explain why it happened."

### Predicted Label
> History

---

## Task 2: Next Word Generation

### Objective
Generate the next 20 words given a seed sentence of 10 or more words.

### How to Run
```bash
python text_generation_preprocess.py
python train_text_generator.py
python generate_word.py
```

### Sample Input
> "the industrial revolution"

### Sample Output
> the industrial revolution and the establishment of civilization regional enough ancient perspectives themes termed historical elements putting instance scope on the culture past

---

## Technologies Used

- Python 3.x  
- TensorFlow / Keras  
- Pandas, NumPy  
- Wikipedia API  
- Scikit-learn  

---

## Notes

- All scripts can be run directly after cloning or downloading the repository.
- You can also run the code in Google Colab or Jupyter Notebook.
- Please ensure required dependencies are installed (e.g., TensorFlow, scikit-learn).

---

