import sys
import json
import transformers
import csv
import re
import random

def normtext(t):
    t=re.sub("<i>|</i>|<br ?/>"," ",t)
    t=re.sub("[0-9]","N",t)
    return t

def proc_text(d):
    d["text"]=" ".join(normtext(d[f]) for f in ("name","synopsis","contentdescription")).strip()
    

def csv2dict(inp):
    r=csv.DictReader(inp,dialect="excel-tab",fieldnames="filmiri,year,name,synopsis,contentdescription".split(","))
    for i,d in enumerate(r):
        if d["synopsis"]=="None" or d["contentdescription"]=="None":
            continue
        proc_text(d)
        decade=int(d["year"])//10*10
        new_d={"sentence":d["text"], "label":str(decade), "id":str(i)}
        yield new_d
        
if __name__=="__main__":
    a=[]
    for d in csv2dict(sys.stdin):
        a.append(d)
    random.shuffle(a)
    for d in a:
        print(json.dumps(d,ensure_ascii=False,sort_keys=True))
        
