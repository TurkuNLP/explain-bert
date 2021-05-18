import datasets
import pickle

d=datasets.load_dataset("json",data_files="class_data.jsonl",split={"train":'train[:80%]',"validation":'train[80%:90%]',"test":'train[90%:]'})
with open("momaf.pkl","wb") as f:
    pickle.dump(d,f)
    
