# YN-CHRS

Le dossier data contient les données fournies.

Le dossier src contient tout le travail efféctué.
Au sein de celui-ci, on retrouve :
    - les dossiers model_weight contenant des modèles pré-entrainés, dont les poids ont été sauvegardés afin d'éviter de
    devoir effectuer l'entrainement à chaque lancement.

    - le fichier read_pics.py contenant les fonctions fournies ainsi que les fonctions permettant de crééer et
    d'utiliser les modèles.

    - le fichier CNN1D.py contenant l'implémentation du réseau neuronal convolutif (CNN) final utilisé.

    - le fichier mean-clustering.py contenant les fonctions utilisées pour la première approche de résolution du problème,
    celle de la résolution par clustering. Ces fonctions ne sont plus utilisées.

    - les fichiers model_stat contenant pour chaque modèle, tous les caractères qu'il lui est possible de détecter
    lorsqu'il prend en entrée la trame d'un caractère spécifique, ainsi que leur proportion de détection.

    - le fichier neural-network.py, contenant l'implémentation d'un RNN classique, que nous n'utilisons finalement plus.

    - les fichiers output contenant des exemple de résultat en sortie d'un modèle lors de la prédiction sur le fichier
    pics_LOGINMDP.bin, avant la phase de post-traitement (cette dernière entrainant une perte d'informations).

    - les fichiers post-treatment.py et pre-treatment.py contenant les fonctions utilisées pour le pré et post
    traitement.

    - le fichier statictrames contenant les valeurs des trames.
