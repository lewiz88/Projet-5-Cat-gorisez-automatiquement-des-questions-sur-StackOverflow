# Présentation:

> Ce projet a été réalisé dans le cadre de la formation d'**Ingénieur en Machine Learning chez OpenClassroom**, notre mission sera donc de catégoriser automatiquement des questions sur le célèbre site StackOverflow en leurs assignant des tags afin de pouvoir retrouver facilement les questions par la suite, pour cela nous allons mettre en place un système de suggestion de tags pour le site, celui-ci prendra la forme d’un algorithme de Machine Learning qui assigne automatiquement plusieurs tags pertinents à une question
[Stack Overflow](https://stackoverflow.com/) propose un outil d’export de données [stackexchange explorer](https://data.stackexchange.com/stackoverflow/queries), qui recense un grand nombre de données authentiqueS de la plateforme d’entraide, nous nous servirons de cette plateforme pour extraire les différentes questions par le biais de requêtes SQL tout ceci dans le but d’obtenir les données nécessaires à l’entraînement de nos algorithmes. 

# Contenu du dépôt:

+ Code de l'API( main et modules),
+ Modelèle d'apprentissage supervisé utilisé(model.pkl),
+ Module permettant la transformation des données en matrice de nombre binaires(vecto.pkl),
+ NoteBook d'exploration,
+ NoteBook de test.

# Stack technique:

+ Python
+ FastAPI
+ Jupyter NoteBook
+ Uvicorn
+ Heroku



> [Documentation interactive de l'API](http://127.0.0.1:8000/docs)
> [Point d'entrée de l'API](http://127.0.0.1:8000/)

### Auteur: Ralph ZOGO