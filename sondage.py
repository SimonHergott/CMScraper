from composition_ecran import id_cooker
import data_helper
from data_helper import StringVersion, correlation_txt

# Objet sondage pour l'export vers un JSON: est associé à un objet composant, mais n'est pas défini par lui.
# On définit ici des objets modèle (model) pour rendre la structure de données un peu plus robuste.

    


class sondage_m:
    def __init__(self):
        self.id = id_cooker.get_instance().get_new_id()
        self.description = StringVersion()  # Utilisation de StringVersion pour stocker les descriptions
        self.auteur = StringVersion()  # Utilisation de StringVersion pour stocker l'auteur
        self.date_creation = None
        self.ouvert= None
        self.choix_unique = None
        self.options = [] # Tableau d'options

    def ajouter_option(self, option):
        # Retourne toujours l'instance stockée (existante ou nouvellement ajoutée)
        for i, option_ex in enumerate(self.options):
            if correlation_txt(option_ex.get_description(), option.get_description(), 0.5):
                self.options[i].ajouter_description(option.get_description())
                return self.options[i]
        self.options.append(option)
        return option

    def ajouter_description(self, description):
        self.description.add_version(description)

    def ajouter_auteur(self, auteur):
        self.auteur.add_version(auteur)
        
    def get_description(self):
        if not self.description.versions:
            return None
        return self.description.get_most_plausible()
    
    def get_auteur(self):
        return self.auteur.get_most_plausible()

    def to_dict(self):
        return {
            "id": self.id,
            "description": self.get_description(),
            "auteur": self.get_auteur(),
            "date_creation": self.date_creation,
            "ouvert": self.ouvert,
            "choix_unique": self.choix_unique,
            "options": [option.to_dict() for option in self.options]
        }
    
    def to_dict_smpl(self):
        return {
            "description": self.get_description(),
            "auteur": self.get_auteur(),
            "options": [option.to_dict() for option in self.options]
        }

class option_m:
    def __init__(self):
        self.id = None
        self.description = StringVersion()
        self.taux = None
        self.sondage_id = None
        self.respondents = [] # Liste d'ID lié à la DB des personnes
        self.scanned = False  # <- ajouté: indique si la liste des répondants a été parcourue

    def get_description(self):
        return self.description.get_most_plausible()
    
    def ajouter_description(self, description):
        self.description.add_version(description)

    def ajouter_respondent(self, respondent: int):
        if respondent not in self.respondents:
            self.respondents.append(respondent)

    def has_been_scanned(self) -> bool:
        return self.scanned

    def mark_scanned(self):
        self.scanned = True

    def to_dict(self):
        return {
            "id": self.id,
            "description": self.get_description(),
            "taux": self.taux,
            "sondage_id": self.sondage_id,
            "respondents": self.respondents,
            "scanned": self.scanned
        }
    
    def to_dict_smpl(self):
        return {
            "description": self.get_description(),
            "taux": self.taux,
            "respondents": self.respondents,
            "scanned": self.scanned
        }


