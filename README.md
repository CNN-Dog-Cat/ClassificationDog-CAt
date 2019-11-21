# Riconoscimento visivo di un cane/gatto

La seguente repo contiene il programma in Python per progettare una CNN per il riconoscimento di cane/gatto attraverso l'inseriemnto di una foto.

Per prima cosa basta eseguire il file [Dog&Cat.py](https://github.com/CNN-Dog-Cat/ClassificationDog-Cat/blob/master/Dog_Cat.py) per poter creare ed istruire il nostro modello con un certo treaning set.
Per creare il modello di training è possibile accedere a [questo sito](https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip)e scaricare il dataset di cani e gatti messo a disposizione. Si avrà una cartella con all'interno 12500 immagini di gatti e 12500 immagini di cani; da questo dataset andremo a creare un treaning set(per addestrare il modello) e un test set(per le prove di riconscimento).

## Passi da seguire
  * Scaricare il data set a [questo link](https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip)
  * Scaricare la repo seguente
  * Creare un training set inserendo in una cartella 12000 immagini di gatti e 12000 immagini di cani, inserendo cani e gatti in due directory separate
  * Creare un test set inserendo in una cartella 500 immagini di cani e gatti tra le immagini rimanenti, anche in modo casuale(le daremo impasto al modello una volta che sarà implementato)
  * Modificare la 'path_train' con il percorso in cui si trova la directory di treaning del file [Dog_Cat.py](https://github.com/CNN-Dog-Cat/ClassificationDog-Cat/blob/master/Dog_Cat.py)
  * Modificare il 'path_test' con il percorso in cui si trova la directory e l'elemento selezionato con il relativo formato del file [Dog_Cat.py](https://github.com/CNN-Dog-Cat/ClassificationDog-Cat/blob/master/Dog_Cat.py)
  * Modificare il percorso dove salvare il modello una volta che sarà completo e eddestrato del file [Dog_Cat.py](https://github.com/CNN-Dog-Cat/ClassificationDog-Cat/blob/master/Dog_Cat.py)
  * Lanciare il programma [Dog_Cat.py](https://github.com/CNN-Dog-Cat/ClassificationDog-Cat/blob/master/Dog_Cat.py)
  * L'esecuzione, al primo lancio, porterà via parecchio tempo (dalle 3 alle 6 ore), quindi mettetevi comodi e pensate a fare altro.
    * Potete modificare la CNN, se volete, cambiando il numero di layer e neuroni a vostro piacimento, questo farà variare il modello, anche di molto
    * Potetre modificare il numero di epoche, ma per avere un buon modello addestrato dovrete garantirli almeno 20 epoche
    
  * Una volta che il modello sarà addestrato, il programma avrà salvato il modello in un file chiamato 'model.h5', questo sarà fondamentale per riprendere il modello, in successivi programmi, e riutulizzarlo dandoli in pasto delle foto scelte da voi
  * Eseguire il programma [Test_Dog_Cat.py](https://github.com/CNN-Dog-Cat/ClassificationDog-Cat/blob/master/Test_Dog_Cat_.py) per classificare una foto scelta da voi
    * Cambiate il 'path_test' per poter cambiare l'immagine di test e vedere la classificazione del modello all'opera
    * Il risultato sarà 0 o 1, 0 = Gatto e 1 = Cane, ma è già presente un IF per stampare l'etichetta corrispondente
    
