# Projet Deep Learning 
## Introduction 

Dans le cadre de l'enseignement Deep Learning, nous avons travaillé sur un article de recherche traitant d'apprentissage automatique. Afin de changer des images, nous avons choisi de travailler sur des données audios et ainsi apprendre à les manipuler.

Voici donc l'article choisi : https://arxiv.org/pdf/1608.04363.pdf

L'article présente une méthode afin de classifier des sons urbains et ainsi apprendre à les reconnaître. 

## Analyse du papier

Nous avons fait un résumé et une critique de ce papier dans ce [fichier .md](https://gitlab.insa-rouen.fr/mdavid/prj_deep/-/blob/master/recherches.md).

On y retrouve principalement une analyse de la data augmentation faite ainsi que des possibles améliorations à faire.

## Expériences liées au projet
Afin d'expérimenter le domaine du machine Learning, nous avons mener quelques expériences reposant sur les dits de notre article étudié, mais aussi sur la curiosité de notre groupe. 

### Expérience 1 : Obtenir les même résultats que l'article


Notre première expérience est tout d'abord d'obtenir les mêmes résultats qu'énoncés dans l'article, en utilisant l'architecture de réseau décrite et en ne procédant à aucune augmentation des données.

**Résultat** : Nous obtenons les mêmes résultats de l'article.

### Expérience 2 : Tester l'influence de la data-augmentation décrite dans l'article

D'après l'article, plusieurs types d'augmentation peuvent être réalisés sur les données audios : augmentation/réduction des tons, ajout de bruit gaussien, changement de tempo. L'article décrit aussi que certaines de augmentations ne doivent être réalisées seulement sur certaines classes, ce que nous avons essayé de faire. Néanmoins, nous n'avons pas accès aux paramètres des fonctions d'augmentations de l'article, nous avons donc testé à taton.

**Résultat** : Nous obtenons les mêmes résultats de l'article.


### Expérience 3 : Tester l'influence des 10 folds

Nous avons voulu voir dans quelle mesure le fait de ne pas réaliser une expérience sur 10 folds pouvaient nuire à la précisions de l'accuracy. Nous avons donc regroupé l'ensemble des sons dans un seul dossier et séparé les sets de training, validation et test en 80% / 10% / 10%. 

### Expérience 4 : Utilisation d'autres modèles pour amélioration

Après s'être rendu compte que l'augmentation des données ne permettait pas forcément une amélioration significative de l'accuracy, nous avons cherché à utiliser d'autre architecture pour classifier nos sons urbains.

Nous avons notamment essayé de représenter les audios par d'autres spectre (mfcc) et de réaliser les expériences avec et sans les fold.

L'architecture de MobileNet a aussi été envisagé, sans résultats pertinents.

## Résultats

Les résultats obtenus sont présentés dans le diaporama Présentation.pdf.
