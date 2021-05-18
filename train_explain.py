from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer
from datasets import load_dataset
import datasets
import pickle
import sys
import torch

MODEL_NAME = 'TurkuNLP/bert-base-finnish-cased-v1'
    
def train():

    with open("momaf.pkl","rb") as f:
        dataset=pickle.load(f)

    labels=sorted(set(dataset['train']['label']))
    labeling=datasets.ClassLabel(names=labels)
    num_labels = len(labels)
    print("labels:",labels)

    def number_labels(d):
        d["label"]=labeling.str2int(d["label"])
        return d

    dataset=dataset.map(number_labels)

    print(dataset)
    print(f'number of distinct labels: {num_labels}')

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def encode_dataset(d):
      return tokenizer(d['sentence'], truncation="only_first")

    encoded_dataset = dataset.map(encode_dataset)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)


    def compute_accuracy(pred):
        y_pred = pred.predictions.argmax(axis=1)
        y_true = pred.label_ids
        return { 'accuracy': sum(y_pred == y_true) / len(y_true) }


    train_args = TrainingArguments(
        'momaf-decade-model',    # output directory for checkpoints and predictions
        load_best_model_at_end=True,
        evaluation_strategy='epoch',
        logging_strategy='steps',
        logging_steps=5,
        learning_rate=1e-4,
        per_device_train_batch_size=8,
        num_train_epochs=30,
        gradient_accumulation_steps=16,
        save_total_limit=10
    )

    trainer = Trainer(
          model,
          train_args,
          train_dataset=encoded_dataset['train'],
          eval_dataset=encoded_dataset['validation'],
          tokenizer=tokenizer,
          compute_metrics=compute_accuracy
    )

    trainer.train()

    results = trainer.evaluate()
    print(f'Accuracy: {results["eval_accuracy"]}')

    #make sure you make this directory
    torch.save(trainer.model,"momaf_decades.pt")
    with open("momaf_decades_labels.pkl","wb") as f:
        pickle.dump(labeling,f)


if __name__=="__main__":
    #this whole code is ugly, since it originates in a notebook
    train()
