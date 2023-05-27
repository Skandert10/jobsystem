#!/usr/bin/env python
# coding: utf-8

# # Description des données

# ### Importation des librairies

# In[438]:


import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt


# ### Importation des données

# In[439]:


df  = pd.read_excel('bd_employé_aiventu.xlsx')


# ### Affichage de la base de données

# In[440]:


df.head(5)


# ### Affichage de la dimension du DataFrame

# In[441]:


df.shape


# ### Affichage des informations sur notre DataFrame

# In[442]:


df.info()


# In[443]:


df[["Gender","Année_expériences"]].describe().T


# In[444]:


df[["Langue","Département","Entreprise","Compétences","Formation",
    "Institution","Expériences","Poste"]].describe().T


# ### Affichage des informations de la colonne cible

# In[445]:


print(df.Poste.describe().T)


# ### Affichage des type de données

# In[446]:


types = df.dtypes


# In[447]:


df_types = pd.DataFrame({'Nom de colonne': df.columns, 'Type de colonne': types.values}).T


# In[448]:


df_types


# In[449]:


df.dtypes


# ### Affichage des types de colonnes et leur nombre

# In[450]:


df.dtypes.value_counts()


# ### Calcule de la sommes des valeurs nulles par colonne et les afficher dans une nouvelle dataframe en fonction du total et du pourcentage

# In[451]:


total = df.isnull().sum().sort_values(ascending = False)
percent = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending = False)
pd.concat([total, percent], axis=1, keys=['Total', 'Percent']).transpose()


# ## Affichage des lignes dupliquées

# In[452]:


len (df[df.duplicated()])


# #### On remarque qu'on a des valeurs nulles dans la colonnes expériences, donc on va explorer des techniques d'imputation pour remédier a ce problème

# ### Affichage du pourcentage des valeurs de la colonne cible  

# In[453]:


df["Poste"].value_counts(normalize=True) 


# ### Diviser les features en deux catégories "numérique" et "catégoriques"
# 

# In[454]:


categorical_features = df[["Entreprise","Département","Langue","Compétences","Formation","Institution","Expériences","Poste"]]


# #### Affichage des valeurs possibles dans chaque colonne de type catégorique

# In[455]:


for col in categorical_features:
    print(f'{col :.<40} {df[col].unique()}')


# #### Affichage la distribution de chaque colonne de type catégorique dans un des graphiques

# #### Affichage des colonnes "Entreprise","Département" et "Langue" dans un diagramme de type pie

# In[456]:


for col in categorical_features[["Entreprise","Département","Langue"]]:
        sizes = df[col].value_counts().values
        labels = df[col].value_counts().index
        colors = ['#0879C4','#F79829','#03A8F2','#7CB5D3', '#F8C68A', '#028EEB', '#D0EAF5', '#2A72F7']
        explode = [0.03] * len(sizes)
        fig, ax = plt.subplots()
        fig.set_size_inches(10, 8)
        plt.title(f"Répartition des {col} par Employé", fontsize=16, fontweight="bold")
        patches, texts, autotexts = ax.pie(sizes, colors=colors, autopct='%1.1f%%', startangle=90, shadow=True, explode=explode)
        ax.axis('equal')
        legend_labels = [f"{label}: {size}" for label, size in zip(labels, sizes)]
        ax.legend(patches, legend_labels, loc='center', bbox_to_anchor=(0.5, -0.1), fontsize=10)
        for text in texts:
            text.set_color('grey')
            text.set_fontsize(12)
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontsize(12)
        plt.tight_layout()
        plt.show()


# In[457]:


df['Département'].value_counts()


# ### Affichage des compétences dans un graphique de type word cloud

# #### Stockage des compétences dans une liste

# In[458]:


competences = df['Compétences'].str.cat(sep=',')


# #### Affichage

# In[459]:


from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)
stopwords.update(['le', 'la', 'les', 'des', 'du', 'un', 'une', 'et', 'ou', 'mais', 'donc', 'or', 'ni', 'car', 'pour', 'par', 'avec', 'depuis', 'chez', 'vers', 'sans', 'sous', 'sur', 'en', 'dans', 'devant', 'derrière', 'après', 'lors', 'parmi', 'entre', 'jusque', 'jusqu', 'tout', 'toute', 'tous', 'toutes', 'ce', 'cet', 'cette', 'ces', 'quel', 'quelle', 'quels', 'quelles', 'notre', 'nos', 'votre', 'vos', 'leur', 'leurs', 'mon', 'ma', 'mes', 'ton', 'ta', 'tes', 'son', 'sa', 'ses', 'lui', 'elle', 'eux', 'elles', 'moi', 'toi', 'soi', 'se', 'même', 'autre', 'autres', 'quelque', 'quelques', 'chacun', 'chacune', 'plusieurs', 'certains', 'certaines', 'aucun', 'aucune', 'personne', 'rien', 'trop', 'beaucoup', 'peu', 'moins', 'très', 'assez', 'suffisamment', 'tel', 'telle', 'tels', 'telles', 'quelquefois', 'parfois', 'souvent', 'rarement', 'jamais', 'toujours', 'maintenant', 'avant', 'après', 'dès', 'pendant', 'tout', 'toute', 'tous', 'toutes', 'chaque', 'autour', 'environ', 'à', 'au', 'aux', 'de', 'des', 'd', 'avec', 'sans', 'pour', 'par', 'sur', 'sous', 'chez', 'devant', 'derrière', 'dans', 'en', 'vers', 'entre', 'jusque', 'jusqu', 'depuis', 'selon', 'hors', 'malgré', 'pendant', 'pour', 'contre', 'à', 'au', 'aux', 'à côté de', 'à travers', 'autour de', 'grâce à', 'en face de', 'à cause de', 'au-dessus de', 'en-dessous de', 'à l intérieur de', 'à l extérieur de', 'à partir de', 'à la place de', 'au lieu de', 'au-delà de', 'parmi', 'avant', 'après', 'devant', 'derrière', 'chez', 'dans', 'en', 'hors', 'lors de', 'malgré', 'pendant', 'sous', 'sur', 'depuis', 'vers', 'contre', 'par', 'pour', 'sans', 'sous', 'chez', 'devant', 'derrière', 'dans', 'en'])
wordcloud = WordCloud(width=1000, height=600,
                      background_color='white',
                      stopwords=stopwords,
                      min_font_size=10).generate(competences)

plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.title('Les compétences les plus fréquentes des employés', color = 'blue', fontweight = 300)
plt.tight_layout(pad=0)

plt.show()


# #### Affichage la distribution des niveaux de formations des employés dans un graphique de type pie

# #### Définition d'un dicitonnaire contenant les différents types de diplomes

# In[460]:


niveaux = {"Licence": 0, "Master": 0,"Ingénierie": 0, "Autres": 0}


# #### Définition de la fonction pour récupérer le nombre de niveaux de formations des employés

# In[461]:


def niveau_etudes(df):
    niveaux = {"Licence": 0, "Master": 0, "Ingénierie": 0, "Autres": 0}
    for formation in df["Formation"]:
        formation_str = str(formation).lower()
        formations = []
        for niv_formation in ["licence", "master", "ingénierie"]:
            if niv_formation in formation_str:
                formations.append(niv_formation)
                niveaux[niv_formation.capitalize()] += formation_str.count(niv_formation)
        if not formations:
            niveaux["Autres"] += 1
        elif len(formations) > 1:
            for niv_formation in formations:
                niveaux[niv_formation.capitalize()] += 1
    return niveaux


# #### Affichage 

# In[462]:


colors = plt.cm.tab20.colors[:len(niveaux)]
niveaux = niveau_etudes(df)
labels = list(niveaux.keys())
sizes = list(niveaux.values())
explode = [0.02] * len(sizes)
fig1, ax1 = plt.subplots()
plt.title("Répartition des niveaux de formation par employé", fontsize=16, fontweight="bold")
plt.legend(labels, loc="best", fontsize=12)
ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, shadow=True, explode=explode)
plt.show()


# #### Enregistrer la sommes de chaque valeur possible dans la colonne cible dans une variable nommée temp

# In[463]:


temp = df["Poste"].value_counts()


# ### Affichage de la distribution de la valeur cible 'Poste'

# In[464]:


import seaborn as sns
import matplotlib.pyplot as plt

df1 = pd.DataFrame({'Poste': temp.index,'values': temp.values})

plt.figure(figsize=(10,6))
plt.title('Répartition des postes', fontsize=15, fontweight='bold')

sns.set_color_codes("pastel")
ax = sns.barplot(x='values', y='Poste', data=df1)

new_labels = [label.get_text()[0:20] + '...' for label in ax.get_yticklabels()]
ax.set_yticklabels(new_labels)

plt.xlabel('Values')
plt.ylabel('Poste')
plt.show()


# #### Affichage des valeurs unique de la colonne cible

# In[465]:


df['Poste'].unique()


# ### Affichage de la distribution des valeurs de la colonne cible

# In[466]:


df['Poste'].value_counts()


# ### Affichage de la distribution des institutions des employés et leurs domaines

# #### Affichage des institutions les fréquentes dans un graphique de type word cloud

# In[467]:


universites = df['Institution'].str.cat(sep=',')


# In[468]:


from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)
# stopwords.update(['le', 'la', 'les', 'des', 'du', 'un', 'une', 'l', 'ou', 'mais', 'donc', 'or', 'ni', 'car', 'pour', 'par', 'avec', 'depuis', 'chez', 'vers', 'sans', 'sous', 'sur', 'en', 'dans', 'devant', 'derrière', 'après', 'lors', 'parmi', 'entre', 'jusque', 'jusqu', 'tout', 'toute', 'tous', 'toutes', 'ce', 'cet', 'cette', 'ces', 'quel', 'quelle', 'quels', 'quelles', 'notre', 'nos', 'votre', 'vos', 'leur', 'leurs', 'mon', 'ma', 'mes', 'ton', 'ta', 'tes', 'son', 'sa', 'ses', 'lui', 'elle', 'eux', 'elles', 'moi', 'toi', 'soi', 'se', 'même', 'autre', 'autres', 'quelque', 'quelques', 'chacun', 'chacune', 'plusieurs', 'certains', 'certaines', 'aucun', 'aucune', 'personne', 'rien', 'trop', 'beaucoup', 'peu', 'moins', 'très', 'assez', 'suffisamment', 'tel', 'telle', 'tels', 'telles', 'quelquefois', 'parfois', 'souvent', 'rarement', 'jamais', 'toujours', 'maintenant', 'avant', 'après', 'dès', 'pendant', 'tout', 'toute', 'tous', 'toutes', 'chaque', 'autour', 'environ', 'à', 'au', 'aux', 'de', 'des', 'd', 'avec', 'sans', 'pour', 'par', 'sur', 'sous', 'chez', 'devant', 'derrière', 'dans', 'en', 'vers', 'entre', 'jusque', 'jusqu', 'depuis', 'selon', 'hors', 'malgré', 'pendant', 'pour', 'contre', 'à', 'au', 'aux', 'à côté de', 'à travers', 'autour de', 'grâce à', 'en face de', 'à cause de', 'au-dessus de', 'en-dessous de', 'à l intérieur de', 'à l extérieur de', 'à partir de', 'à la place de', 'au lieu de', 'au-delà de', 'parmi', 'avant', 'après', 'devant', 'derrière', 'chez', 'dans', 'en', 'hors', 'lors de', 'malgré', 'pendant', 'sous', 'sur', 'depuis', 'vers', 'contre', 'par', 'pour', 'sans', 'sous', 'chez', 'devant', 'derrière', 'dans', 'en'])
wordcloud = WordCloud(width=1000, height=600,
                      background_color='white',
                      stopwords=stopwords,
                      min_font_size=10).generate(universites)

plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)

plt.show()


# #### Affichage de la distribution des domaines des institutions

# #### Définition des listes contenant les deux domaines spécifiques des universités

# In[469]:


ingenierie = [ "Ecole Supérieure Privée d’Ingénierie et de Technologie – ESPRIT","Iteam university", "EPI école pluridisciplinaire internationale", "ULT", "Université SESAME", "ISAMM", "TEK-UP University", 'Ingénierie', 'Engineering', 'Ingénieurs', 'Polytech',"intl Polytech intl","TIME Université","École nationale d\'électronique et des télécommunications de Sfax ENET\'com","École nationale d\'ingénieurs ENIB","École nationale d\'ingénieurs ENICarthage","École nationale d\'ingénieurs ENIG","École nationale d\'ingénieurs ENIM", "École nationale d\'ingénieurs ENIS", "École nationale d\'ingénieurs ENISo", 'École nationale d\'ingénieurs ENIT', 'École nationale des sciences de l\'informatique ENSI','École nationale des sciences et technologies avancées à Borj Cédria ENSTA-B','École polytechnique de Tunisie EPT','École supérieure de la statistique et de l\'analyse de l\'information ESSAI','École supérieure des communications de Tunis SUP\'COM','Institut national des sciences appliquées et de technologie INSAT','Institut supérieur d\'informatique ISI','informatique et des technologies de la communication de Hammam Sousse ISITCOM','informatique et de multimédia ISIMS', 'arts multimédia ISAMM',
               'sciences appliquées et de technologie ISSAT', 'IPEIT', "Institut supérieur des études technologiques ISET", "ISIT'Com", "Université Centrale", "ESP", "SUP'DE COM", "Faculté des Sciences", "ISET'Com",'Ecole Supérieure de Technologie et d\'Informatique'
                   "ISET", "Université de Moncton","Langues Appliquées et d'Informatique de Nabeul ISLAIN", 'Technologies de l\'Information et de la Communication (ISTIC)', 'Technologies Avancées en Informatique et Réseaux', 'Instituts Supérieurs des Etudes Technologiques']

gestion_commerce = ['MSB Mediterranean School of Business', 'Etudes Commerciales de Carthage IHEC Carthage','School of Business ESB', 'Paris-Dauphine','Tunis-Dauphine', 'PSL', 'Hautes Etudes IHET',
               'commerce (ESCT)','de gestion ISG ', 'de gestion ','ISG', 'IHEC' ,'Economiques et Commerciales ESSECT','ESSECT'  'Tunis Business School TBS', 'MSB','Management', 'Gestion', 'Commerce', 'Business', 'Administration','Economie','Economique et de Gestion FSEG','Economiques et de Gestion FSEG','Commerce et de Comptabilité ISCC','Administration des Entreprises ISAE','Economiques et de Gestion FSEG', 'de Gestion Industrielle ISGIS','Hautes Etudes Commerciales IHES','Commerce ESC ','Administration des Affaires', 'Juridiques Economiques et de Gestion FSEG','commerce ESC','Commerce Electronique ESEN',
               'Comptabilité et d\'Administration des Entreprises ISCAE','Gestion de Kairouan ISIGK','Economiques et de Gestion FSEG ','ISG','Tunis Dauphine PSL',
               'Economiques et de Gestion','Management', 'ESC', 'ISGT']

#     technologie = ["Institut supérieur des études technologiques ISET", 
#                    "ISIT'Com", 
#                    "Université Centrale", 
#                    "ESP", "SUP'DE COM", 
#                    "Faculté des Sciences",
#                     "ISET'Com", 'Institut Supérieur Privé Tunis Dauphine - Tunis',
#                    "ISET", "Université de Moncton",
#                    "Institut Supérieur des Langues Appliquées et d'Informatique de Nabeul ISLAIN", 
#                   'Institut Supérieur des Technologies de l\'Information et de la Communication (ISTIC)', 
#                   'Institut Supérieur des Technologies Avancées en Informatique et Réseaux']


# #### Définition de la fonction pour déterminer les domaines des institutions en se basant sur les listes définies auparavnt avec le calcul de similarité entre la colonne et les listes

# In[470]:


from fuzzywuzzy import fuzz
import nltk
from nltk.corpus import stopwords
def compter_universites_par_domaine(df):
    domaines = {}
    stop_words = set(stopwords.words('french'))
    for i, row in df.iterrows():
        universites = row['Institution'].split(',')
        domaines_trouves = set()
        for uni in universites:
            uni_words = [w for w in uni.lower().split() if w not in stop_words and len(w) > 1]
            uni_filtre = ' '.join(uni_words).lower()
            score_ingenierie = max([fuzz.partial_ratio(uni_filtre, x.lower()) for x in ingenierie])
            score_gestion_commerce = max([fuzz.partial_ratio(uni_filtre, x.lower()) for x in gestion_commerce])
            if score_ingenierie >= 83:
                domaines_trouves.add("Ingénierie et technologie")
            if score_gestion_commerce >= 83:
                domaines_trouves.add("Gestion et commerce")
        if len(domaines_trouves) == 0:
            if "Autres" in domaines:
                domaines["Autres"] += 1
            else:
                domaines["Autres"] = 1
        else:
            for domaine in domaines_trouves:
                if domaine in domaines:
                    domaines[domaine] += 1
                else:
                    domaines[domaine] = 1
    return domaines


# #### Appliquer la fonction compter_universites_par_domaine sur notre dataframe

# In[471]:


domaine = compter_universites_par_domaine(df)
print(domaine)


# #### Affichage de la répartition des domaines des institutions

# In[472]:


fig, ax = plt.subplots(figsize=(10,6))
ax.bar(domaine.keys(), domaine.values(), color=["#F79829", "#03A8EB", "#0978BB", "#89CCEC"])

for i, v in enumerate(domaine.values()):
    ax.text(i - 0.1, v + 0.5, str(v), color='black', fontweight='bold')

plt.title("Répartition des universités par domaine")
plt.xlabel("Domaines")
plt.ylabel("Nombre d'universités")

plt.show()


# #### Ajout d'une nouvelle colonne qui correspond au domaine de chaque université

# In[473]:


df["Domaine"] = ""
for index, row in df.iterrows():
    domaines = compter_universites_par_domaine(df.loc[[index]])
    if len(domaines) == 1:
        df.at[index, "Domaine"] = list(domaines.keys())[0]
    else:
        domaines_concat = ""
        for domaine in domaines:
            if domaines_concat != "":
                domaines_concat += " / "
            domaines_concat += domaine
        df.at[index, "Domaine"] = domaines_concat


# #### Affichage de dataframe après la modification

# In[474]:


df


# #### Affichage des valeurs possibles pour la colonne domaine

# In[475]:


df['Domaine'].value_counts()


# #### Affichage des histogrammes pour décriver le statut de la colonne cible en fonction des features catégoriques

# In[476]:


# ConsultantTechniqueMicrosoftDynamics365= df[df["Poste"]=='Consultant Technique Microsoft Dynamics 365']
# ConsultantBI= df[df["Poste"]=='Consultant BI']
# DéveloppeurInformatique = df[df["Poste"]=='Développeur Informatique']
# ChargéRecrutement= df[df["Poste"]=='Chargé Recrutement']
# DigitalMarketer= df[df["Poste"]=='Digital Marketer']
# ConsultantFonctionnelMicrosoftDynamics365= df[df["Poste"]=='Consultant Fonctionnel MicrosoftDynamics 365']
# BusinessDeveloper= df[df["Poste"]=='Business Developer']
# AdministrateurMicrosoft365Azure= df[df["Poste"]=='Administrateur Microsoft 365 / Azure']


# In[477]:


# categorical_features = df[['Gender', 'Domaine', 'Compétences', 'Département', 'Entreprise', 'Poste']]
# f, axes = plt.subplots(2, 2, figsize=(40,17), facecolor='white')
# f.suptitle('La fréquence des valeurs catégoriques par valeur cible (Y)', fontsize=30, color = '#427AA1', fontweight = 'bold')

# ax1 = sns.countplot(x="Gender", hue="Poste", data=categorical_features, palette="Blues_r", ax=axes[0,0])
# ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0, ha="center", fontweight='bold', fontsize=12)
# ax1.set_title('Poste par Genre', fontsize=18)
# ax1.legend(title="Titre", loc='upper left')

# ax4 = sns.countplot(x="Entreprise", hue="Poste", data=categorical_features, palette="Blues_r", ax=axes[0,1])
# ax4.set_xticklabels(ax4.get_xticklabels(), rotation=90, ha="center", fontweight='bold', fontsize=12)
# ax4.set_title('Poste par Entreprise', fontsize=14)
# ax4.legend(title="Titre", loc='upper right')

# ax2 = sns.countplot(x="Domaine", hue="Poste", data=categorical_features, palette="Blues_r",ax=axes[1,0])
# ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90, ha="center", fontweight='bold', fontsize=12)
# ax2.set_title('Poste par Domaine', fontsize=14)
# ax2.legend(title="Titre", loc='upper right')

# ax3 = sns.countplot(x="Département", hue="Poste", data=categorical_features, palette="Reds", ax=axes[1,1])
# ax3.set_xticklabels(ax3.get_xticklabels(), rotation=90, ha="center", fontweight='bold', fontsize=12)
# ax3.set_title('Poste par Département', fontsize=14)
# ax3.legend(title="Titre", loc='upper right')


# In[478]:


df_with_no_encoding=df.copy()


# #### Mettre les valeurs de postes en miniscule

# In[479]:


df['Poste'].unique()


# In[480]:


df['Poste'].value_counts()


# #### Affichage des colonnes de notre dataframe

# In[481]:


df.columns


# # Feature Engineering

# ### Imputing

# #### Impute les valeurs manquantes par les valeurs les plus fréquentes dans chaque colonne

# In[482]:


from sklearn.impute import SimpleImputer
def data_imputer(data):
    df_imputed = df.copy(deep = True)
    imputer = SimpleImputer(strategy = 'most_frequent')
    df_imputed.iloc[:,:] = imputer.fit_transform(df_imputed)
    return df_imputed


# In[483]:


df = data_imputer(df)


# In[484]:


df.isnull().sum()


# ### Transformation de la colonne cible 

# In[485]:


df['Poste'].unique()


# In[486]:


df['Poste'] = df['Poste'].replace('Développeur .NET', 'Développeur Informatique')
df['Poste'] = df['Poste'].replace('Technicien en Mobiscript', 'Développeur Informatique')
df['Poste'] = df['Poste'].replace('Commercial Junior', 'Business Developer')


# In[487]:


df['Poste'].unique()


# In[488]:


df['Poste'] = df['Poste'].apply(lambda x: x.lower())


# #### Affichage des valeurs de la colonne cible après la transformation

# In[489]:


df['Poste'].unique()


# In[490]:


df['Poste'].value_counts()


# #### Dans la partie feature engineering, on a décider d'attribuer un système de socring pour les colonnes Langues, Expériences et Formation afin d'en tirer plus de profit et d'informations de ces colonnes.

# ### Transformation de la colonne Compétences (Attribution de catégorie pour chaque liste de compétences)

# #### Récupérer les compétences de chaque poste dans un dictionnaire

# In[491]:


competences_par_poste = {}
for poste in df['Poste'].unique():
    competences_list = set()
    for index, row in df[df['Poste'] == poste].iterrows():
        comp = row['Compétences'].split(',')
        competences_list.update([comp.strip() for comp in comp])
    competences_par_poste[poste] = list(competences_list)


# #### Affichage du dictionnaire

# In[492]:


competences_par_poste


# In[493]:


df['Poste']


# #### Fonction d'attribution des catégories de compétences

# In[494]:


from fuzzywuzzy import fuzz

def attribuer_categories(df):
    categories_attribuees = {}

    for i, row in df.iterrows():
        competences = row['Compétences'].split(',')
        poste = ""

        for competence in competences:
            for p, competences_poste in competences_par_poste.items():
                for comp in competences_poste:
                    similarity = fuzz.token_set_ratio(competence.strip().lower(), comp.lower())
                    if similarity >= 90:
                        poste = p
                        break
                if poste:
                    break
            if poste:
                break

        if poste:
            categorie = "catégorie_" + poste
            if categorie in categories_attribuees:
                categories_attribuees[categorie] += 1
            else:
                categories_attribuees[categorie] = 1

    return categories_attribuees


# #### Appliquer la fonction sur la dataframe et afficher le dictionnaire de catégorie de compétences obtenus

# In[495]:


categories_comp = attribuer_categories(df)
print(categories_comp)


# #### Appliquer la fonction sur la dataframe 

# In[496]:


df["Catégorie_compétences"] = ""
for index, row in df.iterrows():
    categories_attribuees = attribuer_categories(df.loc[[index]])
    if len(categories_attribuees) == 1:
        df.at[index, "Catégorie_compétences"] = list(categories_attribuees.keys())[0]
    else:
        cat_concat = ""
        for cat in categories_attribuees:
            if cat_concat != "":
                cat_concat += " / "
            cat_concat += cat
        df.at[index, "Catégorie_compétences"] = cat_concat


# #### Affichage de dataframe après transformation de la colonne Compétences

# In[497]:


df[['Catégorie_compétences', 'Poste']].head(80)


# #### Afficher les valeurs uniques de la nouvelle colonne

# In[498]:


df['Catégorie_compétences'].unique()


# #### Afficher le nombre de chaque catégorie

# #### On a remarqué que certains postes peuvent etre unies dans un seul poste, par exemple Développeur informatique et développeur .NET et technicien en mobiscript peuvent etre unies dans un seule poste : Développeur informatique. Meme chose pour les deux postes commercial junior et business developer peuvent etre regroupés dans un seule poste de business developer

# In[499]:


df['Langue'].value_counts()


# ### Scoring Langues

# Dans cette fonction, on va attribuer des scores pour les langues en fonction de leur poste puisque certaines langues sont plus appréciées que les autres

# In[500]:


def calculer_score_langue(df):
    scores_langue_favoris = {"francais": 1.5, "anglais": 3.5, "arabe": 0.5}
    scores_langue_standard = {"francais": 1.5, "anglais": 2.5, "arabe": 0.5}
    postes_favoris = ['consultant fonctionnel microsoft dynamics 365', 
                      'business developer', 'digital marketer']
    df["Langues_score"] = 0
    for i, row in df.iterrows():
        poste = row["Poste"].lower()
        langues = row["Langue"]
        langues_liste = langues.split()
        score_total = 0
        if poste in postes_favoris:
            scores_langue = scores_langue_favoris
        else:
            scores_langue = scores_langue_standard
        for langue in langues_liste:
            if langue.lower() in scores_langue :
                score_total += scores_langue[langue.lower()]
            
            if langue.lower() not in scores_langue and langue.lower() != '-' :
                score_total += 1
        df.at[i, "Langues_score"] = score_total
    return df


# #### Appliquer la fonction de scoring des langues sur la dataframe

# In[501]:


df = calculer_score_langue(df)


# #### Affichage des valeurs possibles de la nouvelle colonnes et leurs nombres

# In[502]:


df['Langues_score'].value_counts()


# ### Scoring des formations

# #### Définition d'un barème de scoring des niveaux de formations è travers un dictionnaire

# In[503]:


niveaux = [("ingénierie", 7),("master", 5),("licence", 3),
           ("bachelor", 4),  ("bts", 1.5), ("doctorat", 9)]


# #### Définition de la fonction de soring des niveaux de formations

# In[504]:


def calculate_education_score(row, niveaux):
    education_score = 0
    education = row["Formation"].lower()
    for keyword, score in niveaux:
        if keyword in education:
            education_score += score
    if education_score == 0:
        education_score = 1 
    return education_score


# #### Appliquer la fonction de scoring des formations sur la dataframe

# In[505]:


df["Education_score"] = df.apply(calculate_education_score, axis=1, args=(niveaux,))


# #### Affichage des valeurs possibles de la nouvelle colonne et leurs nombres

# In[506]:


df["Education_score"].value_counts()


# ### Scoring des expériences

# #### Définition de la fonction de scoring des expériences en fonction de leurs nombre d'expériences

# In[507]:


def calcul_experience_score(df):
    df["experience_score"] = 0   
    for i in range(len(df)):
        experience_str = str(df.iloc[i]["Expériences"])
        num_experiences = len(experience_str.split("\n"))
        if num_experiences >= 5:
            df.at[i, "experience_score"] = 8
        elif num_experiences >= 3:
            df.at[i, "experience_score"] = 5
        elif num_experiences == 2:
            df.at[i, "experience_score"] = 3
        elif num_experiences == 1:
            df.at[i, "experience_score"] = 1.5
    
    return df


# #### Appliquer la fonction de scoring des expériences sur la dataframe

# In[508]:


df = calcul_experience_score(df)


# #### Affichage des valeurs possibles de la nouvelle colonne et leurs nombres

# In[509]:


df["experience_score"].value_counts()


# #### Appliquer la fonction de scoring des expériences sur la dataframe

# In[510]:


df


# ### Afficher les histogrammes des features numériques en fonction de leurs densités

# In[511]:


numerical_features = ["Année_expériences", "Langues_score","Education_score","experience_score"]

fig , ax = plt.subplots(figsize=(15, 25),
                       nrows = len(numerical_features))
for i in range(0,len(ax)):
    ax[i].hist(df[numerical_features[i]] ,bins = 10 ,)
    ax[i].set(xlabel = numerical_features[i])


# #### Afficher la matrice de corrélation générale

# In[512]:


corr = df[['Gender', 'Année_expériences', 'Langues_score', 'Education_score', 'experience_score', 'Domaine', 'Catégorie_compétences']].corr()

plt.figure(figsize=(15,10))
sns.heatmap(corr, annot=True, vmin=-1.0, cmap='Blues_r',linecolor="white")
plt.title("Correlation Heatmap" ,fontsize = 20 ,fontweight ="bold");


# In[513]:


df['Poste'].unique()


# #### Vérifiez les valeurs manquantes, on remarque qu'on a une dataset qui ne contient pas des valeurs manquantes

# In[514]:


df.isnull()


# #### On remarque qu'on a pas maintenant des valeurs manquantes

# In[515]:


# import seaborn as sns
# df.corr()
# plt.figure(figsize=(20,15))
# sns.heatmap(corr, annot=True, vmin=-1.0, cmap='Blues_r',linecolor="white")
# plt.title("Correlation Heatmap" ,fontsize = 20 ,fontweight ="bold");


# ### Encodage des colonnes nominales

# Caractéristiques nominales --> Encodage à chaud (one-hot encoding)
# 
# Cependant, notre dataset contient de nombreuses caractéristiques nominales, l'encodage à chaud peut produire trop de colonnes, ce qui entraîne éventuellement la malédiction de la dimensionnalité et une perte d'informations pertinentes lors de l'étape de sélection des caractéristiques. Pour cette raison, nous utilisons le schéma suivant :
# Caractéristiques nominales --> Encodage de labels

# #### définition de la fonction LabelEncoder()

# In[516]:


df['Poste'].unique()


# In[517]:


def label_encoder(data, cols):
    data_le = data.copy(deep=True)
    label_encoders = {}
    for col in cols:
        label_encoders[col] = LabelEncoder()
        data_le[col] = label_encoders[col].fit_transform(data_le[col])
    return data_le, label_encoders


# In[518]:


col = ['Poste']
df, label_encoders = label_encoder(df, col)


# In[519]:


def obtenir_nom_classes(label_encoders, col):
    encoder = label_encoders[col]
    nom_classes = encoder.classes_
    codes_classes = encoder.transform(nom_classes)
    mapping_classes = dict(zip(codes_classes, nom_classes))
    return mapping_classes

mapping_classes = obtenir_nom_classes(label_encoders, 'Poste')
print(mapping_classes)


# In[520]:


df_nom_classes = pd.DataFrame(list(mapping_classes.items()), columns=['Nom de la classe', 'Classe encodée'])
df_nom_classes


# In[521]:


class_names = label_encoders['Poste'].classes_
print(class_names)


# #### définition de la fonction OneHotEncoder()

# In[522]:


from sklearn.preprocessing import OneHotEncoder

def onehot_encode(df, cat_cols, encoders=None):
    if encoders is None:
        encoders = {}
        
    for col, prefix in cat_cols.items():
        if col not in encoders:
            encoders[col] = OneHotEncoder(sparse=False, handle_unknown='ignore')
            encoders[col].fit(df[[col]]) 
        enc = encoders[col]
        encoded = pd.DataFrame(enc.transform(df[[col]]), columns=enc.categories_[0])
        encoded.columns = [f"{prefix}_{val}" for val in encoded.columns]
        df = pd.concat([df.drop(col, axis=1), encoded], axis=1)

    return df, encoders


# #### Appliquer one hot encoder sur notre colonne nominale 'Domaine'

# In[523]:


df_enc_ohe,encoders = onehot_encode(df, { 'Domaine':'DOM', 
                                          'Catégorie_compétences':'COMP'})


# #### Affichage de notre encodeur entrainé

# On a appliquer LabelEncoder sur la colonne compétences puisqu'elle contient plusieurs modalités, donc en 
# utilisant label encoder à la place de one hot encoder, on va éviter dans un problème de malédiction de la dimensionnalité
# de notre dataframe

# On a appliquer one hot encoder sur la colonne domaine puisqu'elle contient moins de modalités que la colonne compétences

# #### Affichage des colonnes de la dataframe encodées

# In[524]:


df_enc = df_enc_ohe


# In[525]:


df_enc.columns


# In[526]:


# df_enc = pd.concat([df_enc.drop(columns=["Compétences"]), df_competences], axis=1)


# #### Suppression des colonnes inutiles, tels que ID, Nom, Prénom, etc...

# In[527]:


cols_to_drop = ['ID', 'Entreprise', 'Code', 'Département' ,'Compétences','Nom',
                'Prenom', 'Num_Tel', 'E-mail', 'Langue', 'Adresse', 'Formation',
                'Institution', 'Expériences']
df_without_scalers = df_enc.drop(cols_to_drop, axis=1)


# In[528]:


# df_without_scalers[['Poste', 'Compétences_0',
#        'Compétences_1', 'Compétences_2', 'Compétences_3', 'Compétences_4',
#        'Compétences_5', 'Compétences_6', 'Compétences_7', 'Compétences_8',
#        'Compétences_9', 'Compétences_10', 'Compétences_11', 'Domaine_0',
#        'Domaine_1', 'Domaine_2', 'Domaine_3']]


# #### Réajuster la colonne poste en derniere colonne étant notre colonne cible (Y)

# #### Affichage de la dataframe avant la standardisation des colonnes numériques

# In[529]:


poste = df_without_scalers.pop('Poste')
df_without_scalers['Poste'] = poste
df_without_scalers


# In[530]:


len (df_without_scalers[df_without_scalers.duplicated()] )


# In[531]:


len(df_without_scalers)


# In[532]:


# Suppression des observations dupliquées
df_without_scalers.drop_duplicates(inplace = True )
len(df_without_scalers)


# In[533]:


df_without_scalers.isnull().sum()


# ### Standardisation des colonnes numériques  
# 

# #### On va appliquer les deux standarizer sur les colonnes numériques pour évaluer ensuite et choisir le meilleur entre les deux en fonction des performances des modèles 

# # MinMax Scaler

# In[534]:


df_with_min_max_scaler = df_without_scalers.copy()


# In[535]:


len (df_without_scalers[df_without_scalers.duplicated()] )


# #### Application de MinMax Scaler

# In[536]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
numerical_features = [ "Année_expériences" ,  "Langues_score","Education_score","experience_score"]
transformed_numerical_features  = scaler.fit_transform(df_without_scalers[numerical_features])
df_with_min_max_scaler = pd.DataFrame(transformed_numerical_features,columns=numerical_features)


# #### Affichage des colonnes numérqieus standarisées

# In[537]:


df_with_min_max_scaler


# #### Détection des valeurs abérrantes

# In[538]:


plt.style.use("seaborn")


# In[539]:


fig,ax = plt.subplots(figsize = (30 ,20))
ax.boxplot(df_with_min_max_scaler, 
           labels = df_with_min_max_scaler.columns)
fig.suptitle("Détection des valeurs aberrrantes" , fontsize = 25 ,  fontweight ="bold");


# #### Suppression des colonnes numériques non standarisées

# In[540]:


df_enc = df_without_scalers.drop(['Année_expériences','Langues_score','Education_score','experience_score'], axis = 1)


# #### Concaténer la dataframe des colonnes numériques standarisées avec la dataframe encodées

# In[541]:


df_with_min_max_scaler = pd.concat([df_with_min_max_scaler, df_enc], axis=1)


# In[542]:


df_with_min_max_scaler


# In[543]:


# cols_to_drop = ['Id', 'Entreprise', 'Code', 'Département', 'Nom', 'Prenom', 'Date_Recrutement', 'Num_Tel', 'E-mail', 'Langue', 'Adresse', 'Formation', 'Institution', 'Expériences']
# df_with_min_max_scaler = df_with_min_max_scaler.drop(cols_to_drop, axis=1)


# #### Réajuster la dataframe et affichage de la dataframe standarisées avec MinMax Scaler

# In[544]:


poste = df_with_min_max_scaler.pop('Poste')
df_with_min_max_scaler['Poste'] = poste
df_with_min_max_scaler


# In[545]:


len(df_with_min_max_scaler[df_with_min_max_scaler.duplicated()])


# In[546]:


len(df_with_min_max_scaler)


# In[547]:


# Suppression des observations dupliquées
df_with_min_max_scaler.drop_duplicates(inplace = True )
len(df_with_min_max_scaler)


# In[548]:


df_with_min_max_scaler = df_with_min_max_scaler.dropna()


# In[549]:


fig,ax = plt.subplots(figsize = (20 ,10))
labels = ['Année_expériences', 'Langues_score', 'Education_score', 'experience_score']
ax.boxplot(df_with_min_max_scaler.loc[:, labels])
ax.set_xticklabels(labels)
fig.suptitle("Détection des valeurs aberrrantes" , fontsize = 25 ,  fontweight ="bold");


# # Application de StandardScaler()

# In[550]:


df_with_standard_scaler = df_without_scalers.copy()


# #### Application de Standard Scaler

# In[551]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
numerical_features = [ "Année_expériences" ,  "Langues_score","Education_score","experience_score"]
transformed_numerical_features  = scaler.fit_transform(df_without_scalers[numerical_features])
df_with_standard_scaler = pd.DataFrame(transformed_numerical_features,columns=numerical_features)


# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# numerical_features = [ "Année_expériences" ,  "Langues_score","Education_score","experience_score"]
# transformed_numerical_features  = scaler.fit_transform(df_without_scalers[numerical_features])
# df_with_min_max_scaler = pd.DataFrame(transformed_numerical_features,columns=numerical_features)


# #### Affichage des colonnes numériques standarisées

# In[552]:


df_with_standard_scaler


# #### Détection des valeurs abérrantes

# #### Suppression des colonnes non standarisées

# In[553]:


df_enc = df_without_scalers.drop(['Année_expériences','Langues_score','Education_score','experience_score'], axis = 1)


# #### Concaténer la dataframe des colonnes numériques standarisées avec la dataframe encodées

# In[554]:


df_with_standard_scaler = pd.concat([df_with_standard_scaler, df_enc], axis=1)


# #### Réajuster la dataframe et affichage de la dataframe standarisées avec Standard Scaler

# In[555]:


poste = df_with_standard_scaler.pop('Poste')
df_with_standard_scaler['Poste'] = poste
df_with_standard_scaler


# In[556]:


len(df_with_standard_scaler[df_with_standard_scaler.duplicated()])


# In[557]:


len(df_with_standard_scaler)
df_with_standard_scaler.drop_duplicates(inplace = True )
len(df_with_standard_scaler)


# In[558]:


df_with_standard_scaler.isnull().sum()


# In[559]:


df_with_standard_scaler = df_with_standard_scaler.dropna()


# In[560]:


fig,ax = plt.subplots(figsize = (20 ,10))
labels = ['Année_expériences', 'Langues_score', 'Education_score', 'experience_score']
ax.boxplot(df_with_standard_scaler.loc[:, labels])
ax.set_xticklabels(labels)
fig.suptitle("Détection des valeurs aberrrantes" , fontsize = 25 ,  fontweight ="bold");


# In[561]:


#  def onehot_encode(df, column_dict):
#     df = df.copy()
#     for col, prefix in column_dict.items():
#         encoder = LabelEncoder(sparse=False)
#         col_array = df[col].values.reshape(-1, 1)
#         encoded_cols = encoder.fit_transform(col_array)
#         n_cols = encoded_cols.shape[1]
#         col_names = [prefix + '_' + str(i) for i in range(n_cols)]
#         df = pd.concat([df, pd.DataFrame(encoded_cols, columns=col_names)], axis=1)
#         df = df.drop(col, axis=1)
#     return df


# In[562]:


# df_without_scalers = onehot_encode(df_without_scalers , {"Compétences" : "COMP",
#                         "Domaine" : "DOM",
#                         })


# In[563]:


# encoded_cols = df_without_scalers.filter(like="_")  # sélectionner toutes les colonnes encodées
# other_cols = df_without_scalers.drop(columns=encoded_cols.columns)  # sélectionner les autres colonnes
# df_without_scalers = pd.concat([encoded_cols, other_cols], axis=1)  # concaténer les colonnes dans le bon ordre


# #### Affichage de la dataframe non standarisées

# In[564]:


df_without_scalers


# # Modélisation

# #### Importation des librairies 

# In[565]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split 
from sklearn.metrics import f1_score , confusion_matrix,classification_report
from sklearn.model_selection import learning_curve
import numpy as np


# ### Fonction d'évaluation des modèles

# #### On a definit une fonction qui prend en paramètre le dataframe et le modèle qui a pour objectif d'entrainer le modèle et l'evaluer et afficher le classification report et la matrice de confusion ainsi qu'une courbe pour déterminer si le modèle est en Overfitting ou pas pour chaque modèle et observer l'évolution du training score et validation score de chaque modèle
# 

# In[566]:


def evaluation(model, dataframe):
    Y = dataframe.iloc[:, -1]
    X = dataframe.drop("Poste", axis=1)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)

    print(confusion_matrix(Y_test, y_pred))
    print(classification_report(Y_test, y_pred))

    N, train_score, val_score = learning_curve(model, X_train, Y_train,
                                               cv=4, scoring='f1_macro',
                                               train_sizes=np.linspace(0.1, 1, 10)
                                               )

    plt.figure()
    plt.plot(N, train_score.mean(axis=1), label='train score')
    plt.plot(N, val_score.mean(axis=1), label='validation score')
    plt.legend()
    plt.title(type(model).__name__) 
    plt.show()


# In[567]:


dict_of_models = {
    'Decision Tree Classifier' : DecisionTreeClassifier(),
    'SVM Calssifier': SVC(random_state=0),
    'KNN Classifier': KNeighborsClassifier(),
    'XGBoost Classifier' : XGBClassifier(), 
    'RandomForest Classifier' : RandomForestClassifier(), 
    'Naive Bayes' : GaussianNB(),
                                    }


# # Résultats sans scalers

# In[568]:


import warnings
warnings.filterwarnings('ignore')
Y_with_no_scalers=df_without_scalers.iloc[:,-1]
X_with_no_scalers=df_without_scalers.drop("Poste",axis=1)
X_train_with_no_scalers,X_test_with_no_scalers,Y_train_with_no_scalers,Y_test_with_no_scalers=train_test_split(X_with_no_scalers,Y_with_no_scalers,test_size=0.2)
for name,model in dict_of_models.items():
    print(name)
    evaluation(model,df_without_scalers)


# # Résultats avec StandardScaler()

# In[569]:


import warnings
warnings.filterwarnings('ignore')
Y_with_standard_scaler=df_with_standard_scaler.iloc[:,-1]
X_with_standard_scaler=df_with_standard_scaler.drop("Poste",axis=1)
X_train_with_standard_scaler,X_test_with_standard_scaler,Y_train_with_standard_scaler,Y_test_with_standard_scaler=train_test_split(X_with_standard_scaler,Y_with_standard_scaler,test_size=0.2)
for name,model in dict_of_models.items():
    print(name)
    evaluation(model,df_with_standard_scaler)


# # Résultats avec MinMaxScaler()

# In[570]:


df_with_min_max_scaler = df_with_min_max_scaler.dropna()


# In[571]:


Y=df_with_min_max_scaler.iloc[:,-1]
X=df_with_min_max_scaler.iloc[:, :-1]
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
for name,model in dict_of_models.items():
    print(name)
    evaluation(model,df_with_min_max_scaler)


# In[572]:


tab_comp =pd.DataFrame({'Decision Tree Classifier': [0.75, 0.76 ],
                        'KNN Classifier': [0.58, 0.68],'NB Classifier': [0.85, 0.80], 
                        'Random Forest Classifier': [0.85, 0.84],
                        'XGBoost Classifier': [0.81, 0.83],  'SVM Classifier': [0.7, 0.81] },
                      index = ['Données normalisées', ' Données non normalisées'])
tab_comp


# In[573]:


df['Poste'].value_counts()


# #### On a calculé le nombre d'observation pour nos classes, et on a remarqué que nos classes sont désiquilibrées donc on a décidé a faire recours a des techniques de sampling pour éviter le problème d'une dataset déséquilibrée

# In[574]:


# df_over = df_class_0
# for i in range(1, 19):
#     df_class_i = dataframe[dataframe["Poste"] == i]
#     df_class_i_over = df_class_i.sample(count_class_0, replace=True)
#     df_over = pd.concat([df_over, df_class_i_over], axis=0)


# # Application d'OverSampler

# #### Puisque nos classes sont désiquilibrées, on a décider d'utiliser en premier lieu la technique d'oversampling pour faire l'équilibre entre nos classes

# In[575]:


from imblearn.over_sampling import RandomOverSampler
def ros(X_train, y_train):
    ros = RandomOverSampler()
    X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
    return X_resampled, y_resampled
# df_class_1_over = df_class_1.sample(count_class_0, replace=True)
# df_class_2_over = df_class_2.sample(count_class_0, replace=True)
# df_class_3_over = df_class_3.sample(count_class_0, replace=True)
# df_class_4_over = df_class_4.sample(count_class_0, replace=True)
# df_class_5_over = df_class_5.sample(count_class_0, replace=True)
# df_class_6_over = df_class_6.sample(count_class_0, replace=True)
# df_class_7_over = df_class_7.sample(count_class_0, replace=True)
# df_class_8_over = df_class_8.sample(count_class_0, replace=True)
# df_class_9_over = df_class_9.sample(count_class_0, replace=True)
# df_class_10_over = df_class_10.sample(count_class_0, replace=True)
# df_class_11_over = df_class_11.sample(count_class_0, replace=True)
# df_class_12_over = df_class_12.sample(count_class_0, replace=True)
# df_class_13_over = df_class_13.sample(count_class_0, replace=True)
# df_class_14_over = df_class_14.sample(count_class_0, replace=True)
# df_class_15_over = df_class_15.sample(count_class_0, replace=True)
# df_class_16_over = df_class_16.sample(count_class_0, replace=True)
# df_class_17_over = df_class_17.sample(count_class_0, replace=True)
# df_class_18_over = df_class_18.sample(count_class_0, replace=True)
# df_class_19_over = df_class_19.sample(count_class_0, replace=True)

# df_test_over = pd.concat([df_class_1_over, df_class_2_over, df_class_3_over, df_class_4_over, df_class_5_over, df_class_6_over, df_class_7_over, df_class_8_over, df_class_9_over, df_class_10_over, df_class_11_over, df_class_12_over, df_class_13_over, df_class_14_over, df_class_15_over, df_class_16_over, df_class_17_over, df_class_18_over, df_class_19_over], axis=0)
# X_train_ros = df_test_over.drop('Poste', axis = 1)
# y_train_ros = df_test_over['Poste']
# df_over = df_test_over.sample(frac=1, random_state=42)


# In[576]:


X_train = df_without_scalers.iloc[:, :-1]
y_train = df_without_scalers.iloc[:, -1]
X_over, y_over = ros(X_train, y_train)
df_over = pd.concat([X_over, y_over], axis=1)


# In[577]:


df_over['Poste'].value_counts()


# #### La dimesnion du dataframe après l'aplication de l'oversampler

# In[578]:


df_over.shape


# #### Affichage de la dataframe après l'oversampling

# In[579]:


df_over.head()


# #### Evaluation des performance du modèle après l'application de l'oversampler

# In[580]:


for name,model in dict_of_models.items():
    print(name)
    evaluation(model,df_over)


# # SMOTE

# In[581]:


from imblearn.over_sampling import SMOTE
smote= SMOTE()
X = df_with_standard_scaler.drop("Poste", axis=1)
y = df_with_standard_scaler["Poste"]
X_sm, y_sm = smote.fit_resample(X, y)


# In[582]:


df_sm=pd.concat([X_sm,y_sm],axis=1)


# In[583]:


df_sm.shape


# In[584]:


df_sm['Poste'].value_counts()


# In[585]:


for name,model in dict_of_models.items():
    print(name)
    evaluation(model,df_sm)


# In[586]:


# tab_comp =pd.DataFrame({'Decision Tree Classifier': [0.7, 0.69,0.72 ], 'AdaBoost Classifier': [0.62, 0.63, 0.64],
#                         'KNN Classifier': [0.49, 0.63,0.73],'NB Classifier': [0.67, 0.72,0.72], 
#                         'Logistic Regression': [0.8, 0.76,0.73],'Random Forest Classifier': [0.75, 0.76,0.7],
#                         'XGBoost Classifier': [0.77, 0.72, 0.74],  'GB Classifier': [0.79, 0.72,0.72] },
#                       index = ['Smote', 'OverSampler'])
# tab_comp


# ### Optimisation des modèles

# #### Ci-Dessous on va utiliser la fonction :
# 
# *   RandomizedSearchCV à la place dans le but de déterminer les meilleures paramètres d'un modèle 
# 
# *   Classification Report pour déterminer les performances des modèles 
# 

# #### Affichage des matrices de confusion

# In[587]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

import seaborn as sns
def plot_conf_mat(conf_mat, model_name):
    fig , ax = plt.subplots(figsize=(5,5))
    ax = sns.heatmap(conf_mat,annot = True,fmt="d" )
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(f"Confusion Matrix - {model_name}");


# #### Evaluation des prédiction

# ##### On a utilisé plusieurs métriques pour évaluer nos modèles tels que accuracy_score, precision score, recall_score et le score f1

# In[588]:


def evaluate_preds(y_true, y_preds):
    accuracy = accuracy_score(y_true, y_preds)
    precision = precision_score(y_true, y_preds,average = 'macro')
    recall = recall_score(y_true, y_preds,average = 'macro' )
    f1 = f1_score(y_true, y_preds,average = 'macro')
    metric_dict = {"accuracy": round(accuracy,2), 
                   "precision": round(precision,2), 
                   "recall": round(recall), 
                   "f1":round(f1,2)}
    print(f"Acc: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1 score: {f1:.2f}")
    return metric_dict


# In[589]:


# def plot_decision_boundary(classifier, X, y):
#     # Créer une grille de points pour la visualisation de la frontière de décision
#     x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#     x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#     xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.1),
#                            np.arange(x2_min, x2_max, 0.1))

#     # Prédire la classe pour chaque point de la grille
#     Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
#     Z = Z.reshape(xx1.shape)

#     # Afficher la frontière de décision et les points de données
#     plt.contourf(xx1, xx2, Z, alpha=0.3)
#     plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k')

#     plt.xlabel('Feature 1')
#     plt.ylabel('Feature 2')
#     plt.title('Decision Boundary')

#     plt.show()


# ## Tuning KNN

# In[590]:


from sklearn.model_selection import RandomizedSearchCV
np.random.seed(42)

param_kneighbors = {
    'n_neighbors':[2,3,4,5],
    'leaf_size': range(1,50,5),
    'weights': ['uniform', 'distance'],
    'metric': ['minkowski', 'chebyshev', 'manhattan', 'euclidean']
}
Y=df_sm.iloc[:,-1]
X=df_sm.drop("Poste",axis=1)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)

knn = KNeighborsClassifier()

knn_cv = RandomizedSearchCV(knn, param_kneighbors, cv=5, scoring="f1_macro")

knn_cv.fit(X_train, Y_train)
ovr_classifier_knn = OneVsRestClassifier(knn_cv.best_estimator_)
ovr_classifier_knn.fit(X_train, Y_train)

print(f"Les paramètres optimisés de KNN : {knn_cv.best_params_}")


# In[591]:


# for k in range(1,20):
#     knn = KNeighborsClassifier(n_neighbors = k)
#     knn.fit(X_train, Y_train)
#     train_accuracy = knn.score(X_train, Y_train)
#     validation_accuracy = knn.score(X_test, Y_test)
#     print('The Training Accuracy for min_samples_split {:.1f} is: {}'.format(k, train_accuracy))
#     print('The Validation Accuracy for min_samples_split {:.1f} is: {}'.format(k, validation_accuracy))


# In[592]:


from sklearn.model_selection import cross_val_score
np.random.seed(42)

y_preds = ovr_classifier_knn.predict(X_test)
model_single_score = recall_score(Y_test, y_preds, average='macro')
model_cross_val_score = np.mean(cross_val_score(ovr_classifier_knn, X, Y, cv=5, scoring="f1_macro"))

print(f"Model recall score: {model_single_score}")
print(f"Average 5-Fold CV RECALL Score: {round(np.mean(model_cross_val_score), 4)}")
print()
print(classification_report(Y_test, y_preds))

eval_metrics = evaluate_preds(Y_test, y_preds)

sns.set_theme(font_scale=1.5)
N,train_score,val_score = learning_curve(ovr_classifier_knn,X_train,Y_train,cv=4,                          
                            scoring='f1_macro',train_sizes=np.linspace(0.1,1,10))                                         
plt.figure()
plt.plot(N, train_score.mean(axis=1),label='train score')
plt.plot(N, val_score.mean(axis=1),label='validation score')
plt.legend() 
sns.set_theme(font_scale = 1.5)
conf_mat = confusion_matrix(Y_test, y_preds)
plot_conf_mat(conf_mat, "KNeigbors Classifier")


# # Tuning Decision Tree Classifier

# #### Recherche aléatoire des paramètres optimaux pour le classificateur d'arbre de décision (DecisionTreeClassifier)

# In[593]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.model_selection import cross_val_score

param_dist = {"max_depth": range(8,20),
              "min_samples_leaf": range(1,10),
             "criterion": ["gini","entropy"],
               "splitter": ["best"]}

Y=df_sm.iloc[:,-1]
X=df_sm.drop("Poste",axis=1)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)

tree = DecisionTreeClassifier()
tree_cv = RandomizedSearchCV(tree, param_distributions=param_dist, cv=5,scoring="f1_macro")
tree_cv.fit(X_train, Y_train)

ovr_classifier_tree = OneVsRestClassifier(tree_cv.best_estimator_)

ovr_classifier_tree.fit(X_train, Y_train)

y_pred = ovr_classifier_tree.predict(X_test)

print(f"Tuned Decision Tree Parameters: {tree_cv.best_params_}")


# In[594]:


# for min_l in range(1,10):
#     tree = DecisionTreeClassifier(min_samples_leaf=min_l, random_state=42)
#     tree.fit(X_train, Y_train)
#     train_accuracy = tree.score(X_train, Y_train)
#     validation_accuracy = tree.score(X_test, Y_test)
#     print('The Training Accuracy for min_samples_leaf {:.2f} is: {}'.format(min_l, train_accuracy))
#     print('The Validation Accuracy for min_samples_leaf {:.2f} is: {}'.format(min_l, validation_accuracy))


# In[595]:


# for max_d in range(1, 21):
#     tree = DecisionTreeClassifier(max_depth=max_d, random_state=42)
#     tree.fit(X_train, Y_train)
#     train_accuracy = tree.score(X_train, Y_train)
#     validation_accuracy = tree.score(X_test, Y_test)
#     print('The Training Accuracy for max_depth {} is: {}'.format(max_d, train_accuracy))
#     print('The Validation Accuracy for max_depth {} is: {}'.format(max_d, validation_accuracy))


# In[596]:


np.random.seed(42)

y_preds = ovr_classifier_tree.predict(X_test)
model_single_score =recall_score(Y_test,y_preds, average = 'weighted')
# 5-fold cross-validation

model_cross_val_score = np.mean(cross_val_score(ovr_classifier_tree , X , Y , cv = 5, scoring = "f1_weighted"))

print(f"Model recall score: {model_single_score}")
print(f"Average 5-Fold CV RECALL Score: {round(np.mean(model_cross_val_score),4)}")
print()
print (classification_report(Y_test,y_preds))

eval_metrics = evaluate_preds(Y_test, y_preds)
 
N,train_score,val_score = learning_curve(ovr_classifier_tree,X_train,Y_train,cv=4,scoring='f1_macro',train_sizes=np.linspace(0.1,1,10))
                                            
plt.figure()
plt.plot(N, train_score.mean(axis=1),label='train score')
plt.plot(N, val_score.mean(axis=1),label='validation score')
plt.legend() 
sns.set_theme(font_scale = 1.5)

conf_mat = confusion_matrix(Y_test,y_preds)

plot_conf_mat(conf_mat,"DecisionTree Classifier")


# # SVC

# In[597]:


from sklearn.svm import SVC

hyper_params= {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}
Y=df_sm.iloc[:,-1]
X=df_sm.drop("Poste",axis=1)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)

svm = SVC()

svm_cv = RandomizedSearchCV(svm, param_distributions=hyper_params, cv=5, random_state=42, scoring="f1_weighted", n_iter=30)

svm_cv.fit(X_train, Y_train)

ovr_classifier_svm = OneVsRestClassifier(svm_cv.best_estimator_)

ovr_classifier_svm.fit(X_train, Y_train)

y_pred = ovr_classifier_tree.predict(X_test)

print(f"Tuned SVM Parameters: {svm_cv.best_params_}")


# In[598]:


np.random.seed(42)

y_preds = ovr_classifier_svm.predict(X_test)
model_single_score = recall_score(Y_test, y_preds, average='macro')
model_cross_val_score = np.mean(cross_val_score(ovr_classifier_svm, X, Y, cv=5, scoring="f1_macro"))

print(f"Model recall score: {model_single_score}")
print(f"Average 5-Fold CV RECALL Score: {round(np.mean(model_cross_val_score), 4)}")
print()
print(classification_report(Y_test, y_preds))

eval_metrics = evaluate_preds(Y_test, y_preds)

sns.set_theme(font_scale=1.5)
conf_mat = confusion_matrix(Y_test, y_preds)

plot_conf_mat(conf_mat, "SV Classifier")


# # Ada boost

# In[599]:


# np.random.seed(42)

# hyper_params_Adaboost={
#                        'n_estimators':[10, 50, 100, 500],
#                        'learning_rate':[0.0001, 0.001, 0.01, 0.1, 1.0],
#                        'algorithm':["SAMME","SAMME.R"]
#              }

# Y=df_sm.iloc[:,-1]
# X=df_sm.drop("Poste",axis=1)
# X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)

# adb = AdaBoostClassifier()

# adb_cv = RandomizedSearchCV(adb,hyper_params_Adaboost,scoring='f1_weighted',cv=5, random_state = 42)

# adb_cv.fit(X_train, Y_train)

# ovr_classifier_adb = OneVsRestClassifier(adb_cv.best_estimator_)

# ovr_classifier_adb.fit(X_train, Y_train)

# print(f"Tuned Adaboost Parameters: {adb_cv.best_params_}")


# In[600]:


# np.random.seed(42)

# y_preds = ovr_classifier_adb.predict(X_test)
# model_single_score = recall_score(Y_test, y_preds, average='weighted')
# model_cross_val_score = np.mean(cross_val_score(ovr_classifier_adb, X, Y, cv=5, scoring="f1_weighted"))

# print(f"Model recall score: {model_single_score}")
# print(f"Average 5-Fold CV RECALL Score: {round(np.mean(model_cross_val_score), 4)}")
# print()
# print(classification_report(Y_test, y_preds))

# eval_metrics = evaluate_preds(Y_test, y_preds)

# sns.set_theme(font_scale=1.5)
# conf_mat = confusion_matrix(Y_test, y_preds)

# plot_conf_mat(conf_mat, "Adaboost Classifier")


# ## Tuning Random Forest

# In[601]:


hyper_params_RandomForest={
                       'n_estimators': [50,100,200,300,400],
                       'max_features': ['auto', 'sqrt', 'log2'],
                       'max_depth' : [4,5,6,7,8],
                       'min_samples_split': [2, 5, 10],
                       'criterion' :['gini', 'entropy'],
                       'bootstrap': [True, False],
}

Y=df_sm.iloc[:,-1]
X=df_sm.drop("Poste",axis=1)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)

rf = RandomForestClassifier()

rf_cv = RandomizedSearchCV(rf,hyper_params_RandomForest,cv=5, scoring = 'f1_weighted')

rf_cv.fit(X_train, Y_train)

ovr_classifier_rf = OneVsRestClassifier(rf_cv.best_estimator_)

ovr_classifier_rf.fit(X_train, Y_train)

print(f"Tuned Random Forest Parameters: {rf_cv.best_params_}")


# In[602]:


np.random.seed(42)

y_preds = ovr_classifier_rf.predict(X_test)
model_single_score = recall_score(Y_test, y_preds, average='weighted')
model_cross_val_score = np.mean(cross_val_score(ovr_classifier_rf, X, Y, cv=5))

print(f"Model recall score: {model_single_score}")
print(f"Average 5-Fold CV RECALL Score: {round(np.mean(model_cross_val_score), 4)}")
print()
print(classification_report(Y_test, y_preds))

eval_metrics = evaluate_preds(Y_test, y_preds)

sns.set_theme(font_scale=1.5)
conf_mat = confusion_matrix(Y_test, y_preds)

plot_conf_mat(conf_mat, "Random Forest Classifier")


# ## Tuning XGBoost

# In[603]:


df_sm['Poste'].unique()


# In[604]:


hyper_params_XGBoost = { 'learning_rate': [0.01,0.05,0.1],
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
        "n_estimators": [100, 200, 300, 400, 500],
        "objective": "multi:softmax",
        }

Y = df_sm.iloc[:, -1]
X = df_sm.drop("Poste", axis=1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

num_classes = Y.nunique()  # Obtenir le nombre de classes car xgboost exige de lui passer en paramètres le nombre de classes!!

xgb = XGBClassifier(num_class=num_classes)  # Spécifier le nombre de classes

xgb_cv = RandomizedSearchCV(xgb, hyper_params_XGBoost, cv=5, scoring='f1_weighted')
xgb_cv.fit(X_train, Y_train)

ovr_classifier_xgb = OneVsRestClassifier(xgb_cv.best_estimator_)
ovr_classifier_xgb.fit(X_train, Y_train)

print(f"Tuned XGBoost Parameters: {xgb_cv.best_params_}")


# In[605]:


np.random.seed(42)

y_preds = ovr_classifier_xgb.predict(X_test)
model_single_score = recall_score(Y_test, y_preds, average='weighted')
model_cross_val_score = np.mean(cross_val_score(ovr_classifier_xgb, X, Y, cv=5))

print(f"Model recall score: {model_single_score}")
print(f"Average 5-Fold CV RECALL Score: {round(np.mean(model_cross_val_score), 4)}")
print()
print(classification_report(Y_test, y_preds))

eval_metrics = evaluate_preds(Y_test, y_preds)

sns.set_theme(font_scale=1.5)
conf_mat = confusion_matrix(Y_test, y_preds)

plot_conf_mat(conf_mat, "XG Boost Classifier")


# ## Tuning Naive Bayes Classifier

# In[606]:


params_NB = {'var_smoothing': np.logspace(0,-9, num=100)}


Y = df_sm.iloc[:, -1]
X = df_sm.drop("Poste", axis=1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


nb = GaussianNB() 

nb_cv = RandomizedSearchCV(nb, params_NB, cv=5, scoring='f1_weighted')
nb_cv.fit(X_train, Y_train)

ovr_classifier_nb = OneVsRestClassifier(nb_cv.best_estimator_)
ovr_classifier_nb.fit(X_train, Y_train)

print(f"Tuned NB Parameters: {nb_cv.best_params_}")


# In[607]:


np.random.seed(42)

y_preds = ovr_classifier_nb.predict(X_test)
model_single_score = recall_score(Y_test, y_preds, average='weighted')
model_cross_val_score = np.mean(cross_val_score(ovr_classifier_nb, X, Y, cv=5))

print(f"Model recall score: {model_single_score}")
print(f"Average 5-Fold CV RECALL Score: {round(np.mean(model_cross_val_score), 4)}")
print()
print(classification_report(Y_test, y_preds))

eval_metrics = evaluate_preds(Y_test, y_preds)

sns.set_theme(font_scale=1.5)
conf_mat = confusion_matrix(Y_test, y_preds)

plot_conf_mat(conf_mat, "Naive Bayes Classifier")


# ## Tuning Logistic Regression

# In[608]:


# from sklearn.linear_model import LogisticRegression

# C_values = [0.1, 1, 10]  # Valeurs de 'C' à tester

# for C in np.arange(0.1,1.1,0.1) :
#     model = LogisticRegression(multi_class='multinomial', solver='lbfgs', C=C)
#     model.fit(X_train, Y_train)
#     train_accuracy = model.score(X_train, Y_train)
#     validation_accuracy = model.score(X_test, Y_test)
    
#     print(f"Performance du modèle pour C = {C}:")
#     print('Le score d\'entraînement pour C {:.2f} est : {}'.format(C, train_accuracy))
#     print('Le score de validation pour C {:.2f} est : {}'.format(C, validation_accuracy))
#     print("==============================")


# In[222]:


# param_lr = {'C': np.arange(0.1, 1, 10)}


# Y = df_sm.iloc[:, -1]
# X = df_sm.drop("Poste", axis=1)
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


# lr = LogisticRegression(multi_class="multinomial") 

# lr_cv = RandomizedSearchCV(lr, param_lr, cv=5, scoring='f1_weighted')
# lr_cv.fit(X_train, Y_train)

# ovr_classifier_lr = OneVsRestClassifier(lr_cv.best_estimator_)
# ovr_classifier_lr.fit(X_train, Y_train)

# print(f"Tuned NB Parameters: {lr_cv.best_params_}")


# In[223]:


# np.random.seed(42)

# y_preds = ovr_classifier_lr.predict(X_test)
# model_single_score = recall_score(Y_test, y_preds, average='weighted')
# model_cross_val_score = np.mean(cross_val_score(ovr_classifier_lr, X, Y, cv=5))

# print(f"Model recall score: {model_single_score}")
# print(f"Average 5-Fold CV RECALL Score: {round(np.mean(model_cross_val_score), 4)}")
# print()
# print(classification_report(Y_test, y_preds))

# eval_metrics = evaluate_preds(Y_test, y_preds)

# sns.set_theme(font_scale=1.5)
# conf_mat = confusion_matrix(Y_test, y_preds)

# plot_conf_mat(conf_mat, "Logistic Regression Classifier")


# In[ ]:


x= pd.DataFrame({'RandomForest': [0.85, 0.76, 0.81], 'AdaBoost': [0.72,0.80,0.76], 'SVM': [0.59,0.79,0.68], 'KNN': [0.80,0.84,0.82], 'Logistic Regression': [0.88,0.71,0.79], 'Decision Tree Classifier': [0.77,0.79,0.78], 'Gaussian NB': [0.64,0.88,0.74], 'Linear Discriminant Analysis': [0.89,0.71,0.79],'MLP':[0.88,0.79,0.83] },
                      index = ['Precision','Recall','F1-Score'])
x


# In[439]:


from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle

pred_rf = ovr_classifier_rf.predict_proba(X_test)
pred_knn = ovr_classifier_knn.predict_proba(X_test)
pred_adb = ovr_classifier_adb.predict_proba(X_test)
pred_xgb = ovr_classifier_xgb.predict_proba(X_test)

# Déterminer les courbes ROC pour chaque classe
fpr = dict()
tpr = dict()
roc_auc = dict()
n_classes = y_sm_test_bin.shape[1]

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_sm_test_bin[::, i], ovr_classifier_rf[::, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Tracer les courbes ROC pour chaque classe
plt.figure(figsize=(8,6))
colors = cycle(['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan'])
lw = 2

for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curves for multiclass classification')
plt.legend(loc="lower right")
plt.show()


# In[436]:


# from sklearn.metrics import roc_curve, roc_auc_score, auc
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.pipeline import make_pipeline
# from sklearn.feature_selection import SelectKBest,f_classif,chi2
# from sklearn.preprocessing import PolynomialFeatures,StandardScaler, label_binarize
# from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import GaussianNB
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.neural_network import MLPClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import train_test_split 
# from sklearn.metrics import f1_score , confusion_matrix,classification_report
# from sklearn.model_selection import learning_curve 
# from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
# np.random.seed(42)
# y_sm_test_bin = label_binarize(Y_test, classes=np.unique(Y))


# y_pred_proba_rf = ovr_classifier_rf.predict_proba(X_test)
# auc1 = roc_auc_score(Y_test, y_pred_proba_rf, multi_class='ovr', average = 'macro')
# print(f"Aire sous la courbe ROC 1 : {auc1:.4f}")

# y_pred_proba_knn = ovr_classifier_knn.predict_proba(X_test)
# auc2 = roc_auc_score(Y_test, y_pred_proba_knn, multi_class='ovr', average = 'macro')
# print(f"Aire sous la courbe ROC 2 : {auc2:.4f}")
# #-----------------------------------------------------
# y_pred_proba_adb= ovr_classifier_adb.predict_proba(X_test)
# auc3 = roc_auc_score(Y_test, y_pred_proba_adb, multi_class='ovr',average = 'macro')
# print(f"Aire sous la courbe ROC 3 : {auc3:.4f}")
# #-----------------------------------------------------
# y_pred_proba_tree = ovr_classifier_tree.predict_proba(X_test)
# auc4 = roc_auc_score(Y_test, y_pred_proba_tree, multi_class='ovr', average = 'macro')
# print(f"Aire sous la courbe ROC 4 : {auc4:.4f}")
# #-----------------------------------------------------


# In[437]:


# from sklearn.preprocessing import label_binarize
# #binarize the y_values

# y_test_binarized=label_binarize(Y_test,classes=np.unique(Y))

# # roc curve for classes
# fpr = {}
# tpr = {}
# thresh ={}
# roc_auc = dict()

# n_class = y_test_binarized.shape[1]
# for i in range(n_class):    
#     fpr[i], tpr[i], thresh[i] = roc_curve(y_test_binarized[:,i], y_pred_proba_rf[:,i])
#     roc_auc[i] = auc(fpr[i], tpr[i])
    
#     # plotting    
#     plt.plot(fpr[i], tpr[i], linestyle='--', 
#              label='%s vs Rest (AUC=%0.2f)'%(classes[i],roc_auc[i]))

# plt.plot([0,1],[0,1],'b--')
# plt.xlim([0,1])
# plt.ylim([0,1.05])
# plt.title('Multiclass ROC curve')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive rate')
# plt.legend(loc='lower right')
# plt.show()


# In[ ]:


# y_pred_proba_knn = ovr_clf_knn.predict_proba(X_sm_test)[::,1]
# y_pred_proba_rf = ovr_clf_rf.predict_proba(X_sm_test)[::,1]
# y_pred_proba_tree = ovr_clf_tree.predict_proba(X_sm_test)[::,1]
# y_pred_proba_adb = ovr_clf_adb.predict_proba(X_sm_test)[::,1]


# In[ ]:


# fprrf, tprrf, threshrf = roc_curve(y_sm_test,y_pred_proba_rf, pos_label = "commercial junior" )
# fprknn, tprknn, threshknn = roc_curve(y_sm_test,y_pred_proba_knn, pos_label = "commercial junior" )
# fpradb, tpradb, threshadb = roc_curve(y_sm_test,y_pred_proba_adb, pos_label = "commercial junior" )
# fprtree, tprtree, threshtree = roc_curve(y_sm_test,y_pred_proba_tree, pos_label = "commercial junior" )


# In[ ]:


# rand_probs = [0 for i in range(len(y_sm_test))]
# p_fpr, p_tpr, _ = roc_curve(y_sm_test, rand_probs, pos_label='Consultant BI')


# plt.style.use('seaborn')
# plt.plot(fprrf, tprrf, linestyle='--', color='orange', label='Random Forest')
# plt.plot(fprknn, tprknn, linestyle='--', color='blue', label='KNN')
# plt.plot(fprtree, tprtree, linestyle='--', color='red', label='Decision Tree')
# plt.plot(fpradb, tpradb, linestyle='--', color='yellow', label='AdaBoost')
# plt.plot([0, 1], [0, 1], linestyle='--', color='black')
# plt.title('ROC CURVE')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.legend(loc='best')
# plt.show()
# plt.savefig('ImageName', format='png', dpi=200, transparent=True)

# # plt.figure(figsize=(10,7))
# # plt.plot([0, 1], [0, 1], 'k--')
# # plt.plot(fpr1,tpr1,label="Random Forest Classifier, auc="+str(round(auc1,2)))
# # plt.plot(fpr2,tpr2,label="KNN Classifier, auc="+str(round(auc2,2)))
# # plt.plot(fpr3,tpr3,label="Decision Tree Classifier, auc="+str(round(auc3,2)))
# # plt.plot(fpr4,tpr4,label="AdaBoost Classifier, auc="+str(round(auc4,2)))
# # plt.legend(loc=4, title='Models', facecolor='white')
# # plt.xlabel('False Positive Rate')
# # plt.ylabel('True Positive Rate')
# # plt.title('ROC', size=15)
# # plt.box(False)
# # plt.savefig('ImageName', format='png', dpi=200, transparent=True)


# In[ ]:


# from sklearn.preprocessing import label_binarize
# import numpy as np
# from sklearn.metrics import roc_curve,auc
# #binarize the y_values
# y_test_binarized=label_binarize(y_sm_test,classes=np.unique(y_sm_test))
# # roc curve for classes
# fpr = {}
# tpr = {}
# thresh ={}
# roc_auc = dict()
# n_class = len(np.unique(y_sm_test))
# for i in range(n_class):    
#     fpr[i], tpr[i], thresh[i] = roc_curve(y_test_binarized[:,i], y_pred_proba_rf[:,i])
#     roc_auc[i] = auc(fpr[i], tpr[i])
#     fpr1= fpr[i]
#     tpr1 = tpr[i]   
# for i in range(n_class):    
#     fpr[i], tpr[i], thresh[i] = roc_curve(y_test_binarized[:,i], y_pred_proba_knn[:,i])
#     roc_auc[i] = auc(fpr[i], tpr[i])
#     fpr2= fpr[i]
#     tpr2 = tpr[i]
# for i in range(n_class):    
#     fpr[i], tpr[i], thresh[i] = roc_curve(y_test_binarized[:,i], y_pred_proba_tree[:,i])
#     roc_auc[i] = auc(fpr[i], tpr[i])
#     fpr3= fpr[i]
#     tpr3 = tpr[i]
# for i in range(n_class):    
#     fpr[i], tpr[i], thresh[i] = roc_curve(y_test_binarized[:,i], y_pred_proba_adb[:,i])
#     roc_auc[i] = auc(fpr[i], tpr[i])
#     fpr4= fpr[i]
#     tpr4 = tpr[i]


# In[652]:


candidate_df2 = pd.DataFrame({"Gender": 1,
                             "Compétences": ["Power BI, SQL, Java, Data Mining, ETL, Machine Learning, PHP, CSS, HTML"],
                             "Institution": ["ISG Tunis"],
                             "Année_expériences": [1],
                             "Langues": ["Arabe - Francais - Anglais"],
                             "Formation": ["Licence en Business Intelligence"],
                             "Expériences": ["stage d'été chez délice"]})


# In[653]:


numerical_features = ["Année_expériences", "Langues_score", "Education_score", "experience_score"]


# In[656]:


from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    ('experience_score', calcul_experience_score),
    ('language_score', calculate_language_score),
    ('education_score', calculate_education_score),
    ('count_universities', compter_universites_par_domaine),
    ('assign_categories', attribuer_categories),
    ('one_hot_encoding', encoders),
    ('drop_columns', pd.DataFrame.drop, ['Institution', 'Expériences', 'Langues', 'Formation']),
    ('scaling', scaler),
    ('drop_numerical_columns', pd.DataFrame.drop, numerical_features),
    ('classifier', ovr_classifier_nb)  # Assurez-vous d'importer et d'initialiser votre modèle 'shb_classifier'
])


# In[609]:


candidate_df = pd.DataFrame({"Gender": 1,
                             "Compétences": ["Power BI, SQL, Java, Data Mining, ETL, Machine Learning, PHP, CSS, HTML"],
                             "Institution": ["ISG Tunis"],
                             "Année_expériences": [1],
                             "Langues": ["Arabe - Francais - Anglais"],
                             "Formation": ["Licence en Business Intelligence"],
                             "Expériences": ["stage d'été chez délice"]})


# In[610]:


candidate_df


# In[611]:


candidate_df  = calcul_experience_score(candidate_df)


# In[612]:


def calculate_language_score(row):
    score = 0
    if 'Arabe' in row['Langues']:
        score += 1
    if 'Français' in row['Langues']:
        score += 1.5
    if 'Anglais' in row['Langues']:
        score += 2
    if score == 0:
        score = 0.5 # score pour les autres langues
    return score

# 3. Appliquer la fonction calculate_language_score à chaque ligne
candidate_df['Langues_score'] = candidate_df.apply(calculate_language_score, axis=1)


# In[613]:


candidate_df['Education_score'] = candidate_df.apply(calculate_education_score, axis=1, args=(niveaux,))


# In[614]:


domaines_trouves = compter_universites_par_domaine(candidate_df)


# In[615]:


domaines_trouves


# In[616]:


candidate_df["Domaine"] = ""
for index, row in candidate_df.iterrows():
    domaines = compter_universites_par_domaine(candidate_df.loc[[index]])
    if len(domaines) == 1:
        candidate_df.at[index, "Domaine"] = list(domaines.keys())[0]
    else:
        domaines_concat = ""
        for domaine in domaines:
            if domaines_concat != "":
                domaines_concat += " / "
            domaines_concat += domaine
        candidate_df.at[index, "Domaine"] = domaines_concat


# In[617]:


cat_comp = attribuer_categories(candidate_df)


# In[618]:


cat_comp


# In[619]:


candidate_df["Catégorie_compétences"] = ""
for index, row in candidate_df.iterrows():
    categories_attribuees = attribuer_categories(candidate_df.loc[[index]])
    if len(categories_attribuees) == 1:
        candidate_df.at[index, "Catégorie_compétences"] = list(categories_attribuees.keys())[0]
    else:
        cat_concat = ""
        for cat in categories_attribuees:
            if cat_concat != "":
                cat_concat += " / "
            cat_concat += cat
        candidate_df.at[index, "Catégorie_compétences"] = cat_concat


# In[620]:


candidate_df


# In[621]:


# def label_encoder(data, cols):
#     data_le = data.copy(deep = True)
#     le = LabelEncoder()
#     for col in cols:    
#         data_le[col] = le.fit_transform(data_le[col])
#     return data_le


# In[622]:


candidate_df, _ = onehot_encode(candidate_df, {'Domaine' : 'DOM', 'Catégorie_compétences' : 'COMP'}, encoders)


# In[623]:


candidate_df


# In[624]:


# # Transformer la matrice sparse en matrice dense
# candidate_features = vectorizer.transform(candidate_df['Compétences']).toarray()

# # Créer la dataframe avec les noms des colonnes du vectorizer
# candidate_features_df = pd.DataFrame(candidate_features, columns=vectorizer.get_feature_names())


# In[625]:


# # Concaténer la dataframe candidate_df avec candidate_features_df
# candidate_df = pd.concat([candidate_df, candidate_features_df], axis=1)

# # Afficher la nouvelle dataframe


# In[626]:


# # Créer une liste de toutes les colonnes de la dataframe d'entrainement
# all_cols = df_sm.columns

# # Vérifier si chaque colonne existe dans la dataframe du candidat et ajouter les colonnes manquantes avec des zéros
# for col in all_cols:
#     if col not in candidate_df.columns:
#         candidate_df[col] = 0


# In[627]:


candidate_df = candidate_df.drop(['Compétences'], axis=1)


# In[628]:


candidate_df.shape


# In[629]:


# # Liste des colonnes de la dataframe d'entrainement
# train_columns = X_sm_train.columns.tolist()

# # Liste des colonnes de la dataframe du candidat
# candidate_columns = candidate_df.columns.tolist()

# # Liste des colonnes manquantes dans la dataframe du candidat
# missing_columns = list(set(train_columns) - set(candidate_columns))

# # Ajouter les colonnes manquantes dans la dataframe du candidat, initialisées à 0
# for column in missing_columns:
#     candidate_df[column] = 0

# # Réorganiser les colonnes dans la dataframe du candidat pour qu'elles soient dans le même ordre que dans la dataframe d'entraînement
# candidate_df = candidate_df.reindex(columns=train_columns)


# In[630]:


candidate_df = candidate_df.drop(['Institution', 'Expériences', 'Langues', 'Formation'], axis=1)
candidate_df.columns


# In[631]:


# y_pred = KNN.predict(candidate_df)

# # Afficher les prédictions
# print(y_pred)


# In[632]:


# def complete_dataframe(candidate_df, df_train):
#     # Trouver les colonnes relatives aux domaines manquantes dans la dataframe du candidat
#     domains_columns = [col for col in df_train.columns if col.startswith('DOM') and col not in candidate_df.columns]
    
#     # Ajouter les colonnes manquantes à la dataframe du candidat avec des valeurs initiales de 0 pour toutes les lignes
#     for col in domains_columns:
#         candidate_df[col] = 0
    
#     return candidate_df


# In[633]:


# candidate_df.iloc[:, 3:]
numerical_features = [ "Année_expériences" ,  "Langues_score","Education_score","experience_score"]
transformed_numerical_features_cand  = scaler.transform(candidate_df[numerical_features])
df_with_min_max_scaler_cand = pd.DataFrame(transformed_numerical_features_cand,columns=numerical_features)
candidate_df = candidate_df.drop(['Année_expériences','Langues_score','Education_score','experience_score'], axis = 1)
candidate_df = pd.concat([df_with_min_max_scaler_cand, candidate_df], axis=1)


# In[634]:


candidate_df


# In[636]:


y_pred = ovr_classifier_xgb.predict(candidate_df)

# Afficher les prédictions
print(y_pred)


# In[637]:


# import pickle
# filename = 'jobPredModel.sav'
# pickle.dump(ovr_classifier_knn,open(filename, 'wb'))


# In[662]:


import pickle

jobPredModel = ovr_classifier_xgb  # Remplacez par votre modèle

# Enregistrement du modèle, des fonctions et des encodeurs
loaded_objects = {
    'model': jobPredModel,
    'scaler': scaler,
    'compter_universites_par_domaine': compter_universites_par_domaine,
    'calculate_language_score': calculate_language_score,
    'calcul_experience_score': calcul_experience_score,
    'attribuer_categories': attribuer_categories,
    'encoders': encoders
}
pickle.dump(loaded_objects, open('jobPredModel.sav', 'rb'))


# In[658]:


y_pred = jobPredModel.predict(candidate_df)

# Afficher les prédictions
print(y_pred)


# In[224]:


# cols_to_keep = candidate_df.drop(['Année_expériences','experience_score','Langues_score', 'Education_score'])
# candidate_df = pd.concat([candidate_df_sc, candidate_df[cols_to_keep]], axis=1)


# In[ ]:


# from sklearn.preprocessing import label_binarize
# #binarize the y_values

# y_test_binarized=label_binarize(y_sm_test,classes=np.unique(y_sm_test))

# # roc curve for classes
# fpr = {}
# tpr = {}
# thresh ={}
# roc_auc = dict()

# n_class = classes.shape[0]

# for i in range(n_class):    
#     fpr[i], tpr[i], thresh[i] = roc_curve(y_test_binarized[:,i], y_pred_proba_knn[:,i])
#     roc_auc[i] = auc(fpr[i], tpr[i])
    
#     # plotting    
#     plt.plot(fpr[i], tpr[i], linestyle='--', 
#              label='%s vs Rest (AUC=%0.2f)'%(classes[i],roc_auc[i]))

# plt.plot([0,1],[0,1],'b--')
# plt.xlim([0,1])
# plt.ylim([0,1.05])
# plt.title('Multiclass ROC curve')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive rate')
# plt.show()











# # from sklearn.preprocessing import label_binarize
# # import numpy as np
# # from sklearn.metrics import roc_curve,auc
# # #binarize the y_values
# # y_test_binarized=label_binarize(y_sm_test,classes=np.unique(y_sm_test))
# # # roc curve for classes
# # fpr = {}
# # tpr = {}
# # thresh ={}
# # roc_auc = dict()
# # n_class = len(np.unique(y_sm_test))
# # for i in range(n_class):    
# #     fpr[i], tpr[i], thresh[i] = roc_curve(y_test_binarized[:,i], y_pred_proba_rf[:,i])
# #     roc_auc[i] = auc(fpr[i], tpr[i])
# #     fpr1= fpr[i]
# #     tpr1 = tpr[i]   
# # for i in range(n_class):    
# #     fpr[i], tpr[i], thresh[i] = roc_curve(y_test_binarized[:,i], y_pred_proba_knn[:,i])
# #     roc_auc[i] = auc(fpr[i], tpr[i])
# #     fpr2= fpr[i]
# #     tpr2 = tpr[i]
# # for i in range(n_class):    
# #     fpr[i], tpr[i], thresh[i] = roc_curve(y_test_binarized[:,i], y_pred_proba_tree[:,i])
# #     roc_auc[i] = auc(fpr[i], tpr[i])
# #     fpr3= fpr[i]
# #     tpr3 = tpr[i]
# # for i in range(n_class):    
# #     fpr[i], tpr[i], thresh[i] = roc_curve(y_test_binarized[:,i], y_pred_proba_adb[:,i])
# #     roc_auc[i] = auc(fpr[i], tpr[i])
# #     fpr4= fpr[i]
# #     tpr4 = tpr[i]



# # plt.figure(figsize=(10,7))
# # plt.plot([0, 1], [0, 1], 'k--')
# # plt.plot(fpr1,tpr1,label="Random Forest Classifier, auc="+str(round(auc1,2)))
# # plt.plot(fpr2,tpr2,label="KNN Classifier, auc="+str(round(auc2,2)))
# # plt.plot(fpr3,tpr3,label="Decision Tree Classifier, auc="+str(round(auc3,2)))
# # plt.plot(fpr4,tpr4,label="AdaBoost Classifier, auc="+str(round(auc4,2)))
# # plt.legend(loc=4, title='Models', facecolor='white')
# # plt.xlabel('False Positive Rate')
# # plt.ylabel('True Positive Rate')
# # plt.title('ROC', size=15)
# # plt.box(False)
# # plt.savefig('ImageName', format='png', dpi=200, transparent=True)


# In[ ]:


len(np.unique(y_sm_test))


# In[ ]:


np.unique(y_sm_test)


# In[ ]:


y_test_binarized.shape


# In[ ]:


# from sklearn.metrics import roc_curve, roc_auc_score
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.pipeline import make_pipeline
# from sklearn.feature_selection import SelectKBest,f_classif,chi2
# from sklearn.preprocessing import PolynomialFeatures,StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import GaussianNB
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.neural_network import MLPClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import train_test_split 
# from sklearn.metrics import f1_score , confusion_matrix,classification_report
# from sklearn.model_selection import learning_curve 
# from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

# np.random.seed(42)

# X_sm_train , X_sm_test , y_sm_train , y_sm_test = train_test_split (X_sm , y_sm , test_size = 0.2 , random_state = 42, stratify = y_sm)

# # créer un classificateur One-vs-Rest avec le classificateur de votre choix
# ovr_clf_rf = OneVsRestClassifier(RandomForestClassifier(n_estimators=150,min_samples_split=14,max_features='auto',max_depth=9,criterion='entropy',bootstrap=False))
# # entraîner le classificateur sur vos données d'entraînement
# ovr_clf_rf.fit(X_sm_train, y_sm_train)
# # prédire les probabilités pour vos données de test
# y_pred_proba_rf = ovr_clf_rf.predict_proba(X_sm_test)
# # calculer l'aire sous la courbe ROC en utilisant la stratégie One-vs-Rest
# auc1 = roc_auc_score(y_sm_test, y_pred_proba_rf, multi_class='ovr')
# print(f"Aire sous la courbe ROC 1 : {auc1:.4f}")
# #-------------------------------------------
# ovr_clf_knn = OneVsRestClassifier(KNeighborsClassifier(weights = 'uniform', p = 1, n_neighbors = 10, metric = 'minkowski', leaf_size= 20))
# ovr_clf_knn.fit(X_sm_train, y_sm_train)
# y_pred_proba_knn = ovr_clf_knn.predict_proba(X_sm_test)
# auc2 = roc_auc_score(y_sm_test, y_pred_proba_knn, multi_class='ovr')
# print(f"Aire sous la courbe ROC 1 : {auc2:.4f}")
# #-----------------------------------------------------

# ovr_clf_tree = OneVsRestClassifier(DecisionTreeClassifier(splitter='best', min_samples_leaf = 3, max_features = 9, max_depth = 8, criterion = 'gini'))
# ovr_clf_tree.fit(X_sm_train, y_sm_train)
# y_pred_proba_tree = ovr_clf_tree.predict_proba(X_sm_test)
# auc3 = roc_auc_score(y_sm_test, y_pred_proba_tree, multi_class='ovr')
# print(f"Aire sous la courbe ROC 1 : {auc3:.4f}")
# #-----------------------------------------------------
# ovr_clf_adb = OneVsRestClassifier(AdaBoostClassifier(n_estimators=200,learning_rate=0.01,algorithm='SAMME'))
# ovr_clf_adb.fit(X_sm_train, y_sm_train)
# y_pred_proba_adb = ovr_clf_adb.predict_proba(X_sm_test)
# auc4 = roc_auc_score(y_sm_test, y_pred_proba_adb, multi_class='ovr')
# print(f"Aire sous la courbe ROC 1 : {auc4:.4f}")


# # ovr_clf_knn = OneVsRestClassifier(KNeighborsClassifier(weights = 'uniform', p = 1, n_neighbors = 1, metric = 'minkowski', leaf_size= 20))
# # ovr_clf_knn.fit(X_sm_train, y_sm_train)
# # y_pred_proba_knn = ovr_clf_knn.predict_proba(X_sm_test)
# # auc = roc_auc_score(y_sm_test, y_pred_proba_knn, multi_class='ovr')
# # print(f"Aire sous la courbe ROC 3 : {auc:.4f}")
# #-------------------------------------------------------
# # ovr_clf_knn = OneVsRestClassifier(KNeighborsClassifier(weights = 'uniform', p = 1, n_neighbors = 1, metric = 'minkowski', leaf_size= 20))
# # ovr_clf_knn.fit(X_sm_train, y_sm_train)
# # y_pred_proba_knn = ovr_clf_knn.predict_proba(X_sm_test)
# # auc = roc_auc_score(y_sm_test, y_pred_proba_knn, multi_class='ovr')
# # print(f"Aire sous la courbe ROC 3 : {auc:.4f}")

# # svm_cv= SVC(gamma=1e-05,C=1,probability=True)
# # svm_cv.fit(X_sm_train,y_sm_train)
# # y_pred_proba_svc = svm_cv.predict_proba(X_sm_test)[::,1]
# # fpr4, tpr4, _ = roc_curve(y_sm_test,  y_pred_proba_svc)
# # auc4 = roc_auc_score(y_sm_test, y_pred_proba_svc,  multi_class = 'ovr')

# # tree_cv = DecisionTreeClassifier(splitter='best', min_samples_leaf = 1, max_features = None, max_depth = None, criterion = 'gini')
# # tree_cv.fit(X_sm_train, y_sm_train)
# # y_pred_proba_tree = tree_cv.predict_proba(X_sm_test)[::,1]
# # fpr5, tpr5, _ = roc_curve(y_sm_test,  y_pred_proba_tree)
# # auc5 = roc_auc_score(y_sm_test, y_pred_proba_tree, multi_class = 'ovr')

# # knn_cv =KNeighborsClassifier(weights = 'uniform', p = 1, n_neighbors = 1, metric = 'minkowski', leaf_size= 20)
# # knn_cv.fit(X_sm_train, y_sm_train)
# # y_pred_proba_knn = knn_cv.predict_proba(X_sm_test)[::,1]
# # fpr6, tpr6, _ = roc_curve(y_sm_test,  y_pred_proba_knn)
# # auc6 = roc_auc_score(y_sm_test, y_pred_proba_knn, multi_class = 'ovr')


# # LDA_cv =LinearDiscriminantAnalysis (solver = 'svd', tol = 0.0001)
# # LDA_cv.fit(X_sm_train, y_sm_train)
# # y_pred_proba_lda = LDA_cv.predict_proba(X_sm_test)[::,1]
# # fpr8, tpr8, _ = roc_curve(y_sm_test,  y_pred_proba_lda)
# # auc8 = roc_auc_score(y_sm_test, y_pred_proba_lda,  multi_class = 'ovr')

# plt.figure(figsize=(10,7))
# plt.plot([0, 1], [0, 1], 'k--')
# plt.plot(fpr2,tpr2,label="Random Forest, auc="+str(round(auc1,2)))
# plt.plot(fpr3,tpr3,label="Adaboost, auc="+str(round(auc4,2)))
# plt.plot(fpr5,tpr5,label="Decision Tree auc="+str(round(auc3,2)))
# plt.plot(fpr6,tpr6,label="KNN, auc="+str(round(auc2,2)))
# plt.legend(loc=4, title='Models', facecolor='white')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC', size=15)
# plt.box(False)
# plt.savefig('ImageName', format='png', dpi=200, transparent=True);


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




