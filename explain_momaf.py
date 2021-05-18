from captum.attr import visualization as viz
from captum.attr import IntegratedGradients, LayerConductance, LayerIntegratedGradients
from captum.attr import configure_interpretable_embedding_layer, remove_interpretable_embedding_layer
import torch
import transformers
from transformers import AutoTokenizer
import captum
import re
import pickle
import explain

if __name__=="__main__":
    tokenizer = AutoTokenizer.from_pretrained("TurkuNLP/bert-base-finnish-cased-v1")
    model = torch.load("momaf_decades.pt")
    model.to('cuda')

    with open("momaf.pkl","rb") as f:
        dataset=pickle.load(f)


    for item in dataset["train"]:
        target,aggregated=explain.explain([item["sentence"]],model,tokenizer)
        target=target[0]
        aggregated=aggregated[0]
        for tok,a_val in aggregated:
            print(f"movie_{item['id']}",item["label"],int(target),tok,a_val,sep="\t")
