from captum.attr import visualization as viz
from captum.attr import IntegratedGradients, LayerConductance, LayerIntegratedGradients
from captum.attr import configure_interpretable_embedding_layer, remove_interpretable_embedding_layer
import torch
import transformers
from transformers import AutoTokenizer
import captum
import re


# # Forward on the model -> data in, prediction out, nothing fancy really
def predict(model, inputs, token_type_ids=None, attention_mask=None):
    pred=model(inputs, token_type_ids=token_type_ids, attention_mask=attention_mask)
    return pred.logits #return the output of the classification layer

def blank_reference_input(tokenized_input, blank_token_id): #b_encoding is the output of HFace tokenizer
    """
    makes a tuple of blank (input_ids, token_type_ids, attention_mask)
    right now token_types_ids, position_ids, and attention_mask simply point to tokenized_input
    """
    
    #1) let us leave token_type_ids unchanged
    #2) let us leave position embeddings unchanged

    blank_input_ids=tokenized_input.input_ids.clone().detach()
    blank_input_ids[tokenized_input.special_tokens_mask==0]=blank_token_id #blank out everything which is not special token
    return blank_input_ids, tokenized_input.token_type_ids, tokenized_input.attention_mask

def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions

def aggregate(inp,attrs,tokenizer):
    """detokenize and merge attributions"""
    detokenized=[]
    for l in inp.input_ids.cpu().tolist():
        detokenized.append(tokenizer.convert_ids_to_tokens(l))
    attrs=attrs.cpu().tolist()

    aggregated=[]
    for token_list,attr_list in zip(detokenized,attrs): #One text from the batch at a time!
        res=[]
        for token,a_val in zip(token_list,attr_list):
            if token.startswith("##"):
                #This is a continuation. We need to pool by absolute value, i.e. pick the most extreme one
                current_tok,current_a_val=res[-1] #this is what we have so far
                if abs(current_a_val)>abs(a_val): #what we have has larger absval
                    res[-1]=(res[-1][0]+token[2:],res[-1][1])
                else:
                    res[-1]=(res[-1][0]+token[2:],a_val) #the new value had a large absval, let's use that
            else:
                res.append((token,a_val))
        aggregated.append(res)
    return aggregated
    

def explain(text,model,tokenizer,wrt_class="winner"):
    """
    wrt_class: which class should this explaination refer to? either an integer, or "winner"
    """
    #1) tokenize
    inp=tokenizer(text,return_tensors="pt",return_special_tokens_mask=True,truncation=True).to(model.device)
    #2) prepare blank input
    b_input_ids, b_token_type_ids, b_attention_mask=blank_reference_input(inp, tokenizer.convert_tokens_to_ids("-"))

    def predict_f(inputs, token_type_ids=None, attention_mask=None):
        return predict(model,inputs,token_type_ids,attention_mask)
    
    lig = LayerIntegratedGradients(predict_f, model.bert.embeddings)
    if wrt_class=="winner":
        prediction=predict(model,inp.input_ids, inp.token_type_ids, inp.attention_mask)
        target=torch.argmax(prediction,axis=-1)
    else:
        assert False, "not implemented"

    attrs, delta= lig.attribute(inputs=(inp.input_ids,inp.token_type_ids,inp.attention_mask),
                                 baselines=(b_input_ids,b_token_type_ids,b_attention_mask),
                                 return_convergence_delta=True,target=target,internal_batch_size=1)
    attrs_sum = attrs.sum(dim=-1)
    attrs_sum = attrs_sum/torch.norm(attrs_sum)
    aggregated=aggregate(inp,attrs_sum,tokenizer)
    return target,aggregated

def print_aggregated(target,aggregated):
    print("<html><body>")
    for tg,inp_txt in zip(target,aggregated): #one input of the batch
        x=captum.attr.visualization.format_word_importances([t for t,a in inp_txt],[a for t,a in inp_txt])
        print(f"<b>{tg}</b>")
        print(f"""<table style="border:solid;">{x}</table>""")
    print("</body></html>")

    

if __name__=="__main__":
    tokenizer = AutoTokenizer.from_pretrained("TurkuNLP/bert-base-finnish-cased-v1")
    model = torch.load("momaf_decades.pt")
    model.to('cuda')


    txt=["Tyttö kuunsillalta Matti Kassilan ohjaama ja käsikirjoittama draama  Tyttö kuunsillalta (1956) pohjautuu Juhani Tervapään (Hella Wuolijoki) kuunnelmaan. Kun tohtori ja kahden aikuisen lapsen leskiäiti Pepi Varala (Ansa Ikonen) näkee konsertissa nuoruudenrakkautensa Erik Rambergin (Joel Rinne) kyllästyneen oloisena, päättää hän herättää tämän vuorineuvokseksi kohonneen miehen opiskeluaikaisen elämänhalun. Pepi soittaa Erikille salaten henkilöllisyytensä esitellen itsensä tytöksi kuunsillalta. Hän tunnustaa myöhemmin lämpimät tunteensa, joihin rakkaudettomaan avioliittoonsa ja vaimonsa Flooran (Kerstin Nylander) alkoholismiin kyllästynyt mies oitis vastaa. Tohtori Pepi Varala näkee vuorineuvos Erik Rambergin vaimonsa ja rouva Lindmanin seurassa itsenäisyyspäivän konsertissa. Vuorineuvos poistuu konsertista väliajalla kotiin kyllästyttyään ystävättärien juopuneeseen ja äänekkääseen juoruiluun. Konsertin jälkeen Pepi soittaa vuorineuvokselle paljastamatta henkilöllisyyttään ja palauttaa miehen mieleen osakunnan kesäjuhlat 26 vuotta aikaisemmin, jolloin Ramberg haastoi erään tytön kävelemään kanssaan kuunsillalle. Rambergilla on vaikeuksia muistaa tapahtumaa, Pepille kokemus on ollut unohtumaton: \"Todellisuudessa olen ollut kanssanne naimisissa 26 vuotta.\" Pepi sanoo etsineensä nuorta miestä kuunsillalta ja löytäneensä nyt kivipatsaan, joka on julma työnantaja.  Nuoruuden kohtaamisen jälkeen molemmat ovat menneet tahollaan naimisiin ja menestyneet urallaan, mikä kerrotaan kahtena takautumana. Valmistautuessaan itsenäisyyspäivän kutsuille Pepi kirjoittaa Rambergille kirjeen, jossa hän kertoo avioitumisestaan vanhahkon tohtori Varalan kanssa, kahden lapsen syntymästä, miehen kuolemasta auto-onnettomuudessa, omasta päätöksestään jatkaa opintojaan tohtoriksi asti. Sydämessään hän on kuitenkin kaiken aikaa kuulunut Rambergille, jonka kuvaa hän säilyttää medaljongissa poikansa kuvan rinnalla - yhdennäköisyys on silmiinpistävä. Ramberg puolestaan muistelee omaa avioliittoaan, jonka pääasiallinen vaikutin oli raha. Tunteiden puuttuessa Ramberg uppoutuu yhä enemmän työhönsä ja rahan ansaitsemiseen, kun taas hermoherkkä vaimo pakenee miehen välinpitämättömyyttä alkoholiin.  Uudessa puhelinkeskustelussa Pepi ja Ramberg kuvittelevat, millaista heidän elämänsä olisi ollut yhdessä, heidän todellinen elämänsä. Ramberg haluaa tavata Pepin, jonka nimeä hän ei vieläkään tiedä, mutta nainen väistelee peläten todellisuuden rikkovan puhtaan kuvitelman. Tuomari Lindmanin avulla Ramberg ryhtyy selvittämään naisen henkilöllisyyttä ja keskustelee lääkärin kanssa vaimonsa hoidosta ja mahdollisesta avioerosta. Vaimo pyytää päästä matkustamaan Tukholmaan, ja Ramberg myöntyy.  Rambergin herrapäivällisillä jäljitetään kuunsillan tyttöä. Professori Anttila muistaa Linda Varalan alias Pepin, jonka Ramberg tunnistaa keskustelukumppanikseen. Ramberg odottaa Pepiä tämän työpaikan ulkopuolella, tervehtii ja lähettää kukkia, mutta ei mene pitemmälle. Pepi on poissa tolaltaan, kiihtyy vielä enemmän, kun Ramberg soittaa ja kertoo tulevansa tapaamaan häntä illalla. Odotettu kohtaus uhkaa epäonnistua, kun Pepin poika Juhani esiintyy uhmaavasti Rambergia kohtaan, mutta kaikki sulaa sovintoon: Pepi ja Ramberg lähtevät yhdessä uudelle taipaleelle.   - Suomen kansallisfilmografia 5:n (1989) mukaan.",      "Tyttö sinä olet tähti  \"Tyttö sinä olet tähti\"  kertoo varakkaan perheen tyttären Nellin ja lähiön kasvatin, hiphop-DJ Sunen musiikillisen yhteistyön pohjalta nousevan rakkaustarinan. Varakkaan perheen tytär Nelli laulaa kuorossa ja unelmoi poplaulajan urasta, mutta perhe ja poikaystävä Mikko painostavat häntä pyrkimään lääkikseen. Hän esittäytyy levy-yhtiössä, mutta saa kuulla tuottaja Anssilta, että ilman demolevyä asiat eivät etene. Anssi esittelee hänelle Sunen, Kärkiryhmä-hiphop-yhtyeen lauluntekijän, jolla on oma kotistudio hylätyssä teollisuushallissa. Sune torjuu ylimielisesti Nellin pyynnön auttaa demon teossa.   Levy-yhtiössä D.T. miksaa Kärkiryhmän kappaleen uuteen kuosiin, ”radiosoittoon sopivaksi”, mutta tähän Sune ei suostu. Hän purkaa kumppaniensa Kondiksen ja Isukin kanssa levytyssopimuksen saman tien. Kapakassa lyödyn vedon seurauksena naisten seurassa kömpelö Sune yrittää iskeä Nellin demon teon varjolla ja saa sovittua tapaamisen; Nelli puolestaan sanoo Mikon olevan hänen isoveljensä. Tapaaminen studiolla menee puihin, mutta Nelli etsiytyy kuitenkin kuuntelemaan Kärkiryhmän keikkaa. Isukki kaappaa tytön backstagelle, missä Nelli leikkaa asiantuntevasti terän hänen reteiltä puheiltaan. Sune tarjoaa Nellille uutta tilaisuutta demontekoon – nyt tosimielellä. He ystävystyvät. Sune esittelee Nellille ratavarren ”gallerian”, mihin he tekevät yhteisen graffitin.  Nellin ystävä Mari huomauttaa tämän tyylin ja kiinnostuksen olevan vaihtumassa Mikosta Suneen. Nelli hyvittelee yhteisen asunnon hankkimista suunnittelevaa Mikkoa ja ilmoittaa Sunelle, että yhteistyö saa jäädä yhteen lauluun. Valmiin kappaleen he kuuntelevat Nellin kotona, jolloin asia paljastuu hänen yllättyneille vanhemmilleen.   Sune säveltää Nellille uuden kappaleen, herkän rakkauslaulun  Pidä musta kii . Mikon saapuminen studiolle keskeyttää heidän intiimiksi kääntyneen hetkensä. Suhteen luonne valkenee Mikolle, kun Sune puhuttelee häntä Nellin veljenä. Kotona Nelli selittää asiaa Mikolle parhain päin ja he päättävät hankkia yhteisen asunnon.   Mari paljastaa Sunelle Nellin ja Mikon olevan pari. Sune tuntee tulleensa vedätetyksi ja juo päänsä täyteen. Nellille hän kertoo kaiken lähteneen hänen puoleltaan vedonlyönnistä; Nelli tarjoaa ivaten itseään demolevyn vastineeksi. Sune torjuu tarjouksen, mutta tekee levyn valmiiksi. Sune on sotkea välinsä myös Isukin ja Kondiksen kanssa. Nelliltä menee lääkikseen pyrkiminen penkin alle, mutta levy-yhtiö tarjoaa demon kuultuaan hänelle levytyssopimuksen. Sune hämmästyttää Anssin ja D.T.:n lupaamalla sävellyksensä ilmaiseksi Nellille. Levy ilmestyy ja saa radiosoiton.   Nelli on muuttamassa Mikon kanssa omaan asuntoon, kun hän huomaa tämän pimittäneen Sunen yhteydenoton. Hän lähtee kesken muuton Mikko ja Mari kannoillaan Sunen keikkapaikalle. Nelli ja Sune tekevät sovinnon, mutta vasta Isukin komennosta Sune tajuaa lähteä tytön perään. Hän löytää Nellin hoivaamassa Marin kanssa jonossa etuillutta ja turpiinsa saanutta Mikkoa. Sune tunnustaa Nellille tunteensa ja kaikki muu tuntuu katoavan heidän ympäriltään. Kärkiryhmän  Elämä kantaa  -kappaleen aikana myös Nelli nousee lavalle. Yleisö aplodeeraa, kun Nelli ja Sune suutelevat."
]
    txt=[re.sub("[0-9]","N",t) for t in txt]

    target,aggregated=explain(txt,model,tokenizer)
    
    print_aggregated(target,aggregated)

