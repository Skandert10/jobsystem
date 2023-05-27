from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import json
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler



app = FastAPI()
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

niveaux = [("ingénierie", 7),("master", 5),("licence", 3),
           ("bachelor", 4),  ("bts", 1.5), ("doctorat", 9)]

#### Définition de la fonction de soring des niveaux de formations

def calculate_education_score(row, niveaux):
    education_score = 0
    education = row["Formation"].lower()
    for keyword, score in niveaux:
        if keyword in education:
            education_score += score
    if education_score == 0:
        education_score = 1 
    return education_score
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
from fuzzywuzzy import fuzz
import nltk
from nltk.corpus import stopwords

ingenierie = [ "Ecole Supérieure Privée d’Ingénierie et de Technologie – ESPRIT","Iteam university", "EPI école pluridisciplinaire internationale", "ULT", "Université SESAME", "ISAMM", "TEK-UP University", 'Ingénierie', 'Engineering', 'Ingénieurs', 'Polytech',"intl Polytech intl","TIME Université","École nationale d\'électronique et des télécommunications de Sfax ENET\'com","École nationale d\'ingénieurs ENIB","École nationale d\'ingénieurs ENICarthage","École nationale d\'ingénieurs ENIG","École nationale d\'ingénieurs ENIM", "École nationale d\'ingénieurs ENIS", "École nationale d\'ingénieurs ENISo", 'École nationale d\'ingénieurs ENIT', 'École nationale des sciences de l\'informatique ENSI','École nationale des sciences et technologies avancées à Borj Cédria ENSTA-B','École polytechnique de Tunisie EPT','École supérieure de la statistique et de l\'analyse de l\'information ESSAI','École supérieure des communications de Tunis SUP\'COM','Institut national des sciences appliquées et de technologie INSAT','Institut supérieur d\'informatique ISI','informatique et des technologies de la communication de Hammam Sousse ISITCOM','informatique et de multimédia ISIMS', 'arts multimédia ISAMM',
               'sciences appliquées et de technologie ISSAT', 'IPEIT', "Institut supérieur des études technologiques ISET", "ISIT'Com", "Université Centrale", "ESP", "SUP'DE COM", "Faculté des Sciences", "ISET'Com",'Ecole Supérieure de Technologie et d\'Informatique'
                   "ISET", "Université de Moncton","Langues Appliquées et d'Informatique de Nabeul ISLAIN", 'Technologies de l\'Information et de la Communication (ISTIC)', 'Technologies Avancées en Informatique et Réseaux', 'Instituts Supérieurs des Etudes Technologiques']

gestion_commerce = ['MSB Mediterranean School of Business', 'Etudes Commerciales de Carthage IHEC Carthage','School of Business ESB', 'Paris-Dauphine','Tunis-Dauphine', 'PSL', 'Hautes Etudes IHET',
               'commerce (ESCT)','de gestion ISG ', 'de gestion ','ISG', 'IHEC' ,'Economiques et Commerciales ESSECT','ESSECT'  'Tunis Business School TBS', 'MSB','Management', 'Gestion', 'Commerce', 'Business', 'Administration','Economie','Economique et de Gestion FSEG','Economiques et de Gestion FSEG','Commerce et de Comptabilité ISCC','Administration des Entreprises ISAE','Economiques et de Gestion FSEG', 'de Gestion Industrielle ISGIS','Hautes Etudes Commerciales IHES','Commerce ESC ','Administration des Affaires', 'Juridiques Economiques et de Gestion FSEG','commerce ESC','Commerce Electronique ESEN',
               'Comptabilité et d\'Administration des Entreprises ISCAE','Gestion de Kairouan ISIGK','Economiques et de Gestion FSEG ','ISG','Tunis Dauphine PSL',
               'Economiques et de Gestion','Management', 'ESC', 'ISGT']

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
competences_par_poste = {'consultant technique microsoft dynamics 365': ['ASP.NET','Microsoft Dynamics 365 Business Central','C/AL.',
  'POO','Microsoft SQL Server','X++','SQL','Microsoft Power Platform','Microsoft Dynamics 365 Finance and Operations','ERP Microsoft Dynamics 365',
  'SQL Server Reporting Services (SSRS)','C#'], 'business developer': ['Négociations','Gestion de la relation client',  'Consulting',
  'Planification des ressources d’entreprise ERP.',  'Business plan',  'B2B',
  'Prospection téléphonique (CRM)',  'Leadership',  'CRM',  'Microsoft Dynamics 365',  'Vente de solutions',  'Business Development',  'Gestion de projet', 'Présentation',  'Stratégie marketing',
  'Stratégie commerciale','Marketing digital', 'Planification stratégique',  'Management',  "Gestion d'équipe",  'Service client',
  'B2C'], 'administrateur microsoft 365 / azure': ['DNS',  'Microsoft Exchange',  'System Center Configuration Manager (SCCM)',  'Certification ITIL',
  'Réseau', 'Infrastructure',  'Sécurité',  'Réseau TCP/IP',  'Informatique',  'VLAN',  'Office 365',  'Active Directory',
  'VPN',  'Système de noms de domaine (DNS)',  'Serveurs',  'Windows Server',  'Technologies de l’information',  'Migration Azure',  'Administration de systèmes Windows',  'DHCP',
  'Sécurité réseau'], 'consultant bi': ['',  'Dashboarding',  'SQL',  'Analyse de données',  'Scrum',  'SQL Server Integration Services (SSIS)',
  'Transformation et Chargement)',  'Data Mining',  'Data Visualization',  'Conception de bases de données',
  'Data Warehousing',  'SQL Server Analysis Services (SSAS)',  'Reporting',  'SQL Server Reporting Services (SSRS)',  'ETL (Extraction',
  'Microsoft Power BI'], 'digital marketer': ['Communication',  'Négociations',  'Réseaux sociaux',  'Design',  'Relationship Management',
  'stratégie marketing',  'B2B',  'Strategic Communications',  'Adobe Illustrator',  'Adobe Photoshop',  'marketing  numérique',  'planification stratégique',
  'CRM', 'Marketing Communications', 'Commercials',  'B2C marketing',  'B2B Marketing',  'Adobe Xd',  'Management',
  'communication',  'Employee Relations',  'publicité',  'B2C',  'Projet voltaire'], 'développeur informatique': ['Développement web',  'ASP.NET Core', 'Flutter',
  'Programmation orientée objet (POO)', 'Scrum', 'TypeScript'  '.NET framework',  'SQL Server Reporting Services (SSRS)',
  'MySQL', 'AngularJS',  'CSS',  'Kotlin',  'SQL Server',  'SQL',  'PHP',  'Java',  'Framework Spring',  'Angular',  'HTML5',
  'jQuery',  'Bases de données (MySQL)',  'Microsoft SQL Server',  'Programmation orientée objet (OOP)',
  'React Native',  'C#',  'NoSQL',  'C++',  'Services web RESTful',  'Mobiscript',  'JavaScript',  '.N C#',
  'Android Studio',  'Développement mobile',  'Entity Framework',  'Web Services'], 'consultant fonctionnel microsoft dynamics 365': ['Gestion de trésorerie',
  'Microsoft Dynamics 365 Business Central', 'gestion de la production',  'PDCA',  'Comptabilité',  'Gestion des budgets',
  'Gestion de la chaîne logistique',  'Microsoft Dynamics 365 Finance and Operations',  'Conformité réglementaire',  'Gestion des coûts',  'Planification financière',
  'gestion des commandes', 'Sipoc',  'ERP', 'Analyse financière',  'Optimisation des processus',  'Modélisation de processus métier',
  'Gestion de la relation client (CRM)',  'Ishikawa',  "Finance d'entreprise",  'Gestion de la fiscalité'], 'chargé recrutement': ['Évaluation de candidats',
  'Sourcing',  'Recrutement IT',  'Connaissance des outils de recrutement en ligne',  'Gestion de la marque employeur',  'Négociation.',
  'Communication interne',  "Entretiens d'embauche",  'Gestion des ressources humaines',  'Intégration des nouveaux employés',  'Intelligence émotionnelle',
  'Analyse de CVs']}
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

class model_input(BaseModel):
    Gender: int
    Compétences: str
    Institution: str
    Année_expériences: int
    Langues: str
    Formation: str
    Expériences: str

jobPredModel = pickle.load(open('jobPredModel.sav', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
encoders = pickle.load(open('encoders.pkl', 'rb'))

@app.post('/jobPred')
def jobPred(input_parameters: model_input):
    input_data = input_parameters.json()
    input_dict = json.loads(input_data)
    gender = input_dict['Gender']
    comp = input_dict['Compétences']
    inst = input_dict['Institution']
    annee_exp = input_dict['Année_expériences']
    langues = input_dict['Langues']
    formation = input_dict['Formation']
    exp = input_dict['Expériences']
    candidate_df = pd.DataFrame([input_dict])
    candidate_df  = calcul_experience_score(candidate_df)
    candidate_df['Langues_score'] = candidate_df.apply(calculate_language_score, axis=1)
    candidate_df['Education_score'] = candidate_df.apply(calculate_education_score, axis=1, args=(niveaux,))
    domaines_trouves = compter_universites_par_domaine(candidate_df)
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
    cat_comp = attribuer_categories(candidate_df)
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
    candidate_df, _ = onehot_encode(candidate_df, {'Domaine' : 'DOM', 'Catégorie_compétences' : 'COMP'}, encoders)
    candidate_df = candidate_df.drop(['Institution', 'Expériences', 'Langues', 'Formation', 'Compétences'], axis=1)
    numerical_features = [ "Année_expériences" ,  "Langues_score","Education_score","experience_score"]
    transformed_numerical_features_cand  = scaler.transform(candidate_df[numerical_features])
    df_with_min_max_scaler_cand = pd.DataFrame(transformed_numerical_features_cand,columns=numerical_features)
    candidate_df = candidate_df.drop(['Année_expériences','Langues_score','Education_score','experience_score'], axis = 1)
    candidate_df = pd.concat([df_with_min_max_scaler_cand, candidate_df], axis=1)
    transformed_data_list = candidate_df.values.tolist()
    prediction = jobPredModel.predict(transformed_data_list)
    if prediction[0] == 0:
        return 'Administrateur Azure 365'
    elif prediction[0] == 1:
        return 'Business Developer'
    elif prediction[0] == 2:
        return 'Chargé Recrutement'
    elif prediction[0] == 3:
        return 'Consultant BI'
    elif prediction[0] == 4:
        return 'Consultant Fonctionnel Dynamics 365'
    elif prediction[0] == 5:
        return 'Consultant technique Dynamics 365'
    elif prediction[0] == 6:
        return 'Digital Marketer'
    elif prediction[0] == 7:
        return 'Développeur Informatique'