# Classe singleton qui gère le json
# Étendu en classe d'outils pour traiter les données (énorme pot pourri, TODO nettoyer)

import json
from Levenshtein import distance as levenshtein_distance

def correlation_txt(texteA, texteB, seuil = 0.15):
        # Utilisation de la distance de Levenshtein pour déterminer si texteA = TexteB
        # Si distance < seuil * max(len(texteA), len(texteB)), on considère que les textes sont égaux
        if texteA is None or texteB is None:
            return False
        
        distance = levenshtein_distance(texteA, texteB)
        max_length = max(len(texteA), len(texteB))
        
        if max_length == 0:  # Eviter la division par zéro
            return True
        
        print(f"correlation_txt(): correlation de {texteA} avec {texteB}: Same = {distance} < {seuil * max_length}")
        
        return (distance < seuil * max_length) # Bool qui dit si les textes sont les memes

def write_to_json_file(data, filename):
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"Erreur ecriture Json: {e}")


class id_cooker:
    """ Singleton pour fabriquer des id uniques"""

    _instance = None
    _counter = 0

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(id_cooker, cls).__new__(cls)
        return cls._instance
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_new_id(self):
        self._counter += 1
        return self._counter


class StringVersion:
    # Classe représentant différentes versions d'une même chaine de caractères
    # Utilisée pour la correction OCR, pour stocker les différentes versions d'une chaîne de caractères et choisir la plus plausible
    def __init__(self):
        self.versions = []

    def add_version(self, string):
        self.versions.append(string)

    def get_all_versions(self):
        return self.versions
    
    def number_of_versions(self):
        return len(self.versions)

    def get_most_plausible(self):
        if len(self.versions) == 1:
            return self.versions[0]
        else:
            # Normalisation des versions
            normalized_versions = [version.strip().lower() for version in self.versions]
            # Calcul de la longueur moyenne des versions
            longueur_moyenne = round(sum(len(version.split()) for version in normalized_versions) / len(normalized_versions))
            mots = []
            for i_mot in range(longueur_moyenne):
                mots_at_i = []
                for version in normalized_versions:
                    mots_version = version.split()
                    if i_mot < len(mots_version):
                        mots_at_i.append(mots_version[i_mot])

                # Comptage des occurrences des mots
                word_bins = {}
                for mot in mots_at_i:
                    if mot in word_bins:
                        word_bins[mot] += 1
                    else:
                        word_bins[mot] = 1

                # Sélection du mot le plus fréquent
                if word_bins:
                    mots.append(max(word_bins, key=word_bins.get))

            # Reconstruction de la chaîne
            string_finale = " ".join(mots)
            return string_finale
        
    def to_dict(self):
        return {
            "versions": self.versions,
            "most_plausible": self.get_most_plausible()
        }


class SurveyedGuy:

    def __init__(self, id = None):
        self.id = id if id is not None else id_cooker.get_instance().get_new_id()
        self.seen_names = StringVersion()  # Utilisation de StringVersion pour stocker les noms
        # TODO: voir la complexité et s'il vaut mieux mettre un véritable nom à chaque fois qu'on ajoute un nom possible

    def get_name(self):
        return self.seen_names.get_most_plausible()
    
    def was_this_name_seen(self, name):
        # Vérifie si le nom a déjà été vu, strict
        return name in self.seen_names.get_all_versions()
    
    def add_name(self, name):
        # Ajoute un nom à la liste des noms vus
        if not self.was_this_name_seen(name):
            self.seen_names.add_version(name)

    def get_all_seen_names(self):
        return self.seen_names.get_all_versions()
    
    def to_dict(self):
        return {
            "id": self.id,
            "seen_names": self.seen_names.to_dict()
        }

class PeopleDatabase:
# Database liant les gens à un ID. On garde les noms comme Stringversions.

    _instance = None
    _people = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PeopleDatabase, cls).__new__(cls)
        return cls._instance
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def init(self, file):
        _people = []
    
    def add_person(self, person: SurveyedGuy):
        if self._people is None:
            self._people = []
        self._people.append(person)

    def add_person_from_name(self, name):
        # Ajoute une personne à partir d'un nom
        person = SurveyedGuy()
        person.add_name(name)
        self.add_person(person)
        return person.id

    def get_person_by_name(self, name):
        if self._people is None:
            return None
        for person in self._people:
            if person.was_this_name_seen(name):
                return person
        return None
    
    def get_id_from_name(self, name):
        # Retourne l'ID d'une personne à partir de son nom
        person = self.get_person_by_name(name)
        if person is not None:
            return person.id
        return None
    
    def get_person_by_id(self, id):
        if self._people is None:
            return None
        for person in self._people:
            if person.id == id:
                return person
        return None
    
    def get_all_people(self):
        if self._people is None:
            return []
        return self._people
    
    def to_dict(self):
        return {
            "people": [person.to_dict() for person in self._people] if self._people else []
        }
    def to_dict_smpl(self):
        return {
            "people": [person.get_name() for person in self._people] if self._people else []
        }
    
    def does_this_name_ring_a_bell(self, name):
        # Vérifie si un nom a déjà été vu dans la base de données
        if self._people is None:
            return False
        for person in self._people:
            if person.was_this_name_seen(name):
                return True
        return False
    
    def get_id_from_name(self, name):
        # Retourne l'ID d'une personne à partir de son nom
        if self._people is None:
            return None
        for person in self._people:
            if person.was_this_name_seen(name):
                return person.id
        return None
    
    def get_name_from_id(self, id):
        # Retourne le nom d'une personne à partir de son ID
        if self._people is None:
            return None
        for person in self._people:
            if person.id == id:
                return person.get_name()
        return None

    def export_to_json(self, filename):
        # Exporte la base de données en JSON
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Erreur lors de l'exportation en JSON: {e}")
        
    
