# Riconoscimento visivo di un cane/gatto per classi sbilanciate

<img src="https://live.staticflickr.com/3640/3639419066_d8de3661a2_b.jpg"/>

La seguente repo contiene il programma in Python per progettare una CNN per il riconoscimento di cane/gatto attraverso l'inseriemnto di una foto (o serie di foto).

Per creare il dataset è possibile accedere a [questo sito](https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip)e scaricare il dataset di cani e gatti messo a disposizione. Si avrà una cartella con all'interno 12500 immagini di gatti e 12500 immagini di cani; da questo dataset andremo a creare un treaning set(per addestrare il modello) e un test set(per le prove di riconscimento).

## Directory Tree
 * `Model_12_12` --> Directory che contiene il modello fittato e pesato della nostra CNN, eseguito su un treaning set perfettamente bilanciato (12K gatti e 12K cani)
 * `Model_12_9` --> Directory che contiene il modello fittato e pesato della nostra CNN, eseguito su un treaning set poco sbilanciato (12K gatti e 9.5K cani)
 * `Model_12_5` --> Directory che contiene il modello fittato e pesato della nostra CNN, eseguito su un treaning set molto sbilanciato (12K gatti e 5K cani)
 * `Model_12_1` --> Directory che contiene il modello fittato e pesato della nostra CNN, eseguito su un treaning set estremamente sbilanciato (12K gatti e 1.2K cani)
 * `ModelPy` --> Directory che contiene i vari modelli CNN non ancora fittati (un fitting richiede in media 4-6 ore di computazione)
   * [Balance_12_12.py](https://github.com/CNN-Dog-Cat/ClassificationDog-Cat/blob/master/ModelsPy/Balance_12_12.py) è un file in python per l'esecuzione e il fitting del modello bilanciato
   * [Unbalance_12_9.py](https://github.com/CNN-Dog-Cat/ClassificationDog-Cat/blob/master/ModelsPy/Unbalance_12_9.py) è un file in python per l'esecuzione e il fitting del modello poco sbilanciato
   * [Unbalance_12_5.py](https://github.com/CNN-Dog-Cat/ClassificationDog-Cat/blob/master/ModelsPy/Unbalance_12_5.py) è un file in python per l'esecuzione e il fitting del modello poco sbilanciato
   * [Unbalance_12_1.py](https://github.com/CNN-Dog-Cat/ClassificationDog-Cat/blob/master/ModelsPy/Unbalance_12_1.py) è un file in python per l'esecuzione e il fitting del modello poco sbilanciato
 * `Flatten_Output` --> directory che contiene l'output del livello di Flatten della CNN, utile per inserirlo in altri modelli o classificatori per testarne la potenza e il confronto con questa CNN.
 * `Weight` --> directory che contiene i pesi del primo livello della CNN
 * [Dog_Cat_Exe.py](https://github.com/CNN-Dog-Cat/ClassificationDog-Cat/blob/master/Dog_Cat_Exe.py) programma in python per l'esecuzione dei modelli già fittati e funzionanti per testarne la classificazione mediante test set e altre immagini casuali.
 * `Tabella Valori` --> directory che contiene i valori dei vari modelli

## Passi da seguire
  * Scaricare il data set a [questo link](https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip)
  * Scaricare la repo seguente
  * Creare un training set inserendo in una cartella 12000 immagini di gatti e 12000 immagini di cani, inserendo cani e gatti in due directory separate
  * Creare un test set inserendo in una cartella 650/700 immagini di cani e gatti tra le immagini rimanenti, anche in modo casuale(le daremo impasto al modello una volta che sarà implementato)
  * L'esecuzione, al primo lancio, porterà via parecchio tempo (dalle 3 alle 6 ore), quindi mettetevi comodi e pensate a fare altro, altrimenti scaricate i modelli già pronti e divertitevi nell'utilizzo.
    * Potete modificare la CNN, se volete, cambiando il numero di layer e neuroni a vostro piacimento, questo farà variare il modello, anche di molto
    * Potetre modificare il numero di epoche, ma per avere un buon modello addestrato dovrete garantirli almeno 20 epoche
  * Una volta che il modello sarà addestrato, il programma avrà salvato il modello in un file chiamato 'model.h5', questo sarà fondamentale per riprendere il modello, in successivi programmi, e riutulizzarlo dandoli in pasto delle foto scelte da voi (ogni model.h5 sarà salvato nella corrispettiva cartella di Model_12_? in base a queale modello abbiate lanciato)
  * Eseguire il programma [Dog_Cat_Exe.py](https://github.com/CNN-Dog-Cat/ClassificationDog-Cat/blob/master/Dog_Cat_Exe.py) per classificare una foto scelta da voi o un seti di foto(700 di gatti e 700 di cani)
    * Il risultato sarà 0 o 1, 0 = Gatto e 1 = Cane, ma è già presente un IF per stampare l'etichetta corrispondente
  * Una volta eseguito lo script verranno generati diversi file e risultati:
    * Il file di output per il flatten, che potrete riutilizzare per altri modelli
    * Il file dei pesi del primo livello
    * I valori che sono stati utilizzati per testare il modello
    * I numeri di errori di classificazione e i relativo errore di classificazione in percentuale(%)
    
    
## Tabella Valori
### CNN 4 Layers
<img src="https://github.com/CNN-Dog-Cat/ClassificationDog-Cat/blob/master/Tabella%20Valori/ValoriCNN4.jpg"/>

### CNN 6 Layers
<img src="https://github.com/CNN-Dog-Cat/ClassificationDog-Cat/blob/master/Tabella%20Valori/ValoriCNN6.jpg"/>
    
