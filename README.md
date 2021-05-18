# explain-bert

Quick code for basic BERT explanation. Sorry about all the hard-coded paths etc.

1) Data is in .jsonl like this:

    {"id": "778", "label": "lab1", "sentence": "Se alkoi sateessa  Se alkoi sateessa  (NNNN) on romanttisia sävyjä sisältävä komediallinen rikoselokuva, jonka Thure Bahne ohjasi, kun se kesken kuvausten otettiin pois ohjaaja Eddie Stenbergilta. Vuokko Takala (Eila Peitsalo) pakenee ukkosmyrskyä kalastusmajaan ja tapaa salaperäisen miehen (Tauno Palo), jota hän epäilee salakuljettaja Rompan Eetuksi. Miehen ilmiantamisen sijaan Vuokko päättää seurata häntä Helsinkiin salakuljettajien piilopaikkaan sillä ehdolla, että tämä lupaa ilmiantaa itsensä ja toverinsa poliisille. Tornionjokilaaksossa metsästetään salakuljettajia. Eräässä väijytyksessä saadaan lasti ja apureita kiinni, mutta päätekijä, Rompan Eetu pääsee pakoon. Tapahtumat siirtyvät saaressa sijaitsevalle kalastusmajalle, joka näyttää olevan salakuljettajien varasto ja lymypaikka. Myrsky pakottaa nuoren tytön, Vuokon, etsimään suojaa saaresta. Mökissä hän tapaa salaperäisen miehen, joka peittelee henkilöllisyyttään. Mies ja tyttö tutustuvat, kiusoittelevat toisiaan, leikkivät miestä ja vaimoa.   Radion uutislähetyksestä Vuokko saa aih"}
    {"id": "779", "label": "lab2", "sentence": "some other text"}

2) Then you can run `make_dataset.py` to turn it into HF Dataset

3) Then `train_explain.py` to train the model

4) Then `explain.py` to do stuff with it
