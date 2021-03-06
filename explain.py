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


    txt=["Tytt?? kuunsillalta Matti Kassilan ohjaama ja k??sikirjoittama draama  Tytt?? kuunsillalta (1956) pohjautuu Juhani Tervap????n (Hella Wuolijoki) kuunnelmaan. Kun tohtori ja kahden aikuisen lapsen leski??iti Pepi Varala (Ansa Ikonen) n??kee konsertissa nuoruudenrakkautensa Erik Rambergin (Joel Rinne) kyll??styneen oloisena, p????tt???? h??n her??tt???? t??m??n vuorineuvokseksi kohonneen miehen opiskeluaikaisen el??m??nhalun. Pepi soittaa Erikille salaten henkil??llisyytens?? esitellen itsens?? tyt??ksi kuunsillalta. H??n tunnustaa my??hemmin l??mpim??t tunteensa, joihin rakkaudettomaan avioliittoonsa ja vaimonsa Flooran (Kerstin Nylander) alkoholismiin kyll??stynyt mies oitis vastaa. Tohtori Pepi Varala n??kee vuorineuvos Erik Rambergin vaimonsa ja rouva Lindmanin seurassa itsen??isyysp??iv??n konsertissa. Vuorineuvos poistuu konsertista v??liajalla kotiin kyll??stytty????n yst??v??tt??rien juopuneeseen ja ????nekk????seen juoruiluun. Konsertin j??lkeen Pepi soittaa vuorineuvokselle paljastamatta henkil??llisyytt????n ja palauttaa miehen mieleen osakunnan kes??juhlat 26 vuotta aikaisemmin, jolloin Ramberg haastoi er????n tyt??n k??velem????n kanssaan kuunsillalle. Rambergilla on vaikeuksia muistaa tapahtumaa, Pepille kokemus on ollut unohtumaton: \"Todellisuudessa olen ollut kanssanne naimisissa 26 vuotta.\" Pepi sanoo etsineens?? nuorta miest?? kuunsillalta ja l??yt??neens?? nyt kivipatsaan, joka on julma ty??nantaja.  Nuoruuden kohtaamisen j??lkeen molemmat ovat menneet tahollaan naimisiin ja menestyneet urallaan, mik?? kerrotaan kahtena takautumana. Valmistautuessaan itsen??isyysp??iv??n kutsuille Pepi kirjoittaa Rambergille kirjeen, jossa h??n kertoo avioitumisestaan vanhahkon tohtori Varalan kanssa, kahden lapsen syntym??st??, miehen kuolemasta auto-onnettomuudessa, omasta p????t??ksest????n jatkaa opintojaan tohtoriksi asti. Syd??mess????n h??n on kuitenkin kaiken aikaa kuulunut Rambergille, jonka kuvaa h??n s??ilytt???? medaljongissa poikansa kuvan rinnalla - yhdenn??k??isyys on silmiinpist??v??. Ramberg puolestaan muistelee omaa avioliittoaan, jonka p????asiallinen vaikutin oli raha. Tunteiden puuttuessa Ramberg uppoutuu yh?? enemm??n ty??h??ns?? ja rahan ansaitsemiseen, kun taas hermoherkk?? vaimo pakenee miehen v??linpit??m??tt??myytt?? alkoholiin.  Uudessa puhelinkeskustelussa Pepi ja Ramberg kuvittelevat, millaista heid??n el??m??ns?? olisi ollut yhdess??, heid??n todellinen el??m??ns??. Ramberg haluaa tavata Pepin, jonka nime?? h??n ei viel??k????n tied??, mutta nainen v??istelee pel??ten todellisuuden rikkovan puhtaan kuvitelman. Tuomari Lindmanin avulla Ramberg ryhtyy selvitt??m????n naisen henkil??llisyytt?? ja keskustelee l????k??rin kanssa vaimonsa hoidosta ja mahdollisesta avioerosta. Vaimo pyyt???? p????st?? matkustamaan Tukholmaan, ja Ramberg my??ntyy.  Rambergin herrap??iv??llisill?? j??ljitet????n kuunsillan tytt????. Professori Anttila muistaa Linda Varalan alias Pepin, jonka Ramberg tunnistaa keskustelukumppanikseen. Ramberg odottaa Pepi?? t??m??n ty??paikan ulkopuolella, tervehtii ja l??hett???? kukkia, mutta ei mene pitemm??lle. Pepi on poissa tolaltaan, kiihtyy viel?? enemm??n, kun Ramberg soittaa ja kertoo tulevansa tapaamaan h??nt?? illalla. Odotettu kohtaus uhkaa ep??onnistua, kun Pepin poika Juhani esiintyy uhmaavasti Rambergia kohtaan, mutta kaikki sulaa sovintoon: Pepi ja Ramberg l??htev??t yhdess?? uudelle taipaleelle.   - Suomen kansallisfilmografia 5:n (1989) mukaan.",      "Tytt?? sin?? olet t??hti  \"Tytt?? sin?? olet t??hti\"  kertoo varakkaan perheen tytt??ren Nellin ja l??hi??n kasvatin, hiphop-DJ Sunen musiikillisen yhteisty??n pohjalta nousevan rakkaustarinan. Varakkaan perheen tyt??r Nelli laulaa kuorossa ja unelmoi poplaulajan urasta, mutta perhe ja poikayst??v?? Mikko painostavat h??nt?? pyrkim????n l????kikseen. H??n esitt??ytyy levy-yhti??ss??, mutta saa kuulla tuottaja Anssilta, ett?? ilman demolevy?? asiat eiv??t etene. Anssi esittelee h??nelle Sunen, K??rkiryhm??-hiphop-yhtyeen lauluntekij??n, jolla on oma kotistudio hyl??tyss?? teollisuushallissa. Sune torjuu ylimielisesti Nellin pyynn??n auttaa demon teossa.   Levy-yhti??ss?? D.T. miksaa K??rkiryhm??n kappaleen uuteen kuosiin, ???radiosoittoon sopivaksi???, mutta t??h??n Sune ei suostu. H??n purkaa kumppaniensa Kondiksen ja Isukin kanssa levytyssopimuksen saman tien. Kapakassa ly??dyn vedon seurauksena naisten seurassa k??mpel?? Sune yritt???? iske?? Nellin demon teon varjolla ja saa sovittua tapaamisen; Nelli puolestaan sanoo Mikon olevan h??nen isoveljens??. Tapaaminen studiolla menee puihin, mutta Nelli etsiytyy kuitenkin kuuntelemaan K??rkiryhm??n keikkaa. Isukki kaappaa tyt??n backstagelle, miss?? Nelli leikkaa asiantuntevasti ter??n h??nen reteilt?? puheiltaan. Sune tarjoaa Nellille uutta tilaisuutta demontekoon ??? nyt tosimielell??. He yst??vystyv??t. Sune esittelee Nellille ratavarren ???gallerian???, mihin he tekev??t yhteisen graffitin.  Nellin yst??v?? Mari huomauttaa t??m??n tyylin ja kiinnostuksen olevan vaihtumassa Mikosta Suneen. Nelli hyvittelee yhteisen asunnon hankkimista suunnittelevaa Mikkoa ja ilmoittaa Sunelle, ett?? yhteisty?? saa j????d?? yhteen lauluun. Valmiin kappaleen he kuuntelevat Nellin kotona, jolloin asia paljastuu h??nen yll??ttyneille vanhemmilleen.   Sune s??velt???? Nellille uuden kappaleen, herk??n rakkauslaulun  Pid?? musta kii . Mikon saapuminen studiolle keskeytt???? heid??n intiimiksi k????ntyneen hetkens??. Suhteen luonne valkenee Mikolle, kun Sune puhuttelee h??nt?? Nellin veljen??. Kotona Nelli selitt???? asiaa Mikolle parhain p??in ja he p????tt??v??t hankkia yhteisen asunnon.   Mari paljastaa Sunelle Nellin ja Mikon olevan pari. Sune tuntee tulleensa ved??tetyksi ja juo p????ns?? t??yteen. Nellille h??n kertoo kaiken l??hteneen h??nen puoleltaan vedonly??nnist??; Nelli tarjoaa ivaten itse????n demolevyn vastineeksi. Sune torjuu tarjouksen, mutta tekee levyn valmiiksi. Sune on sotkea v??lins?? my??s Isukin ja Kondiksen kanssa. Nellilt?? menee l????kikseen pyrkiminen penkin alle, mutta levy-yhti?? tarjoaa demon kuultuaan h??nelle levytyssopimuksen. Sune h??mm??stytt???? Anssin ja D.T.:n lupaamalla s??vellyksens?? ilmaiseksi Nellille. Levy ilmestyy ja saa radiosoiton.   Nelli on muuttamassa Mikon kanssa omaan asuntoon, kun h??n huomaa t??m??n pimitt??neen Sunen yhteydenoton. H??n l??htee kesken muuton Mikko ja Mari kannoillaan Sunen keikkapaikalle. Nelli ja Sune tekev??t sovinnon, mutta vasta Isukin komennosta Sune tajuaa l??hte?? tyt??n per????n. H??n l??yt???? Nellin hoivaamassa Marin kanssa jonossa etuillutta ja turpiinsa saanutta Mikkoa. Sune tunnustaa Nellille tunteensa ja kaikki muu tuntuu katoavan heid??n ymp??rilt????n. K??rkiryhm??n  El??m?? kantaa  -kappaleen aikana my??s Nelli nousee lavalle. Yleis?? aplodeeraa, kun Nelli ja Sune suutelevat."
]
    txt=[re.sub("[0-9]","N",t) for t in txt]

    target,aggregated=explain(txt,model,tokenizer)
    
    print_aggregated(target,aggregated)

