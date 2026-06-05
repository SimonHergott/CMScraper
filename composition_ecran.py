from data_helper import id_cooker

class composition_ecran:
    """ Classe permettant de gérer et d'organiser les éléments graphiques détectés sur l'écran.
    Chaque composition d'écran correspond à une frame et comporte plusieurs composants (voir définition des composants au dessous)
    Cette classe permet de vérifier l'intégrité de la composition d'écran détectée, aisi que de construire un score global de confiance.
    """

    def __init__(self, frame):
        self.frame = frame
        self.composants = []
        self.score = 0
        self.id = id_cooker.get_instance().get_new_id()

    def ajouter_composant(self, composant):
        self.composants.append(composant)

    def verifier_integrite(self):
        # Vérifie l'intégrité de la composition d'écran
        for composant in self.composants:
            composant.verifier_integrite()

    def get_all_composants(self):
        return self.composants
    
    def ordonner(self, threshold_incl = 0.9, verbose = False):
        # Pour tous les composants dans la composition, on regarde ceux qui sont imbriqués les uns dans les autres pour définir des fils et des pères.
        
        # Composants de sondage classique
        sondages = []
        options = []
        voir_reponses = []
        auteurs = []
        boutons_voir_tout = []

        # Composant d'option_reponse développée
        boutons_fermer_reponse = []
        personnes_sondees = []
        reponses_dev = []

        for composant in self.composants:
            if composant.is_sondage():
                sondages.append(composant)
                if verbose:
                    print(f"Sondage {composant.id} ajouté")
            if composant.is_option_reponse():
                options.append(composant)
                if verbose:
                    print(f"Option {composant.id} ajoutée")
            if composant.is_voir_reponses_option():
                voir_reponses.append(composant)
                if verbose:
                    print(f"Voir réponses {composant.id} ajouté")
            if composant.is_auteur_sondage():
                auteurs.append(composant)
                if verbose:
                    print(f"Auteur {composant.id} ajouté")
            if composant.is_bouton_voir_tout():
                boutons_voir_tout.append(composant)
                if verbose:
                    print(f"Bouton voir tout {composant.id} ajouté")
            if composant.is_bouton_fermer_reponse():
                boutons_fermer_reponse.append(composant)
                if verbose:
                    print(f"Bouton fermer réponse {composant.id} ajouté")
            if composant.is_personne_sondee():
                personnes_sondees.append(composant)
                if verbose:
                    print(f"Personne sondée {composant.id} ajoutée")
            if composant.is_reponse_dev():
                reponses_dev.append(composant)
                if verbose:
                    print(f"Réponse dev {composant.id} ajoutée")

        traitement_reponse_dev = False # Si on traite les réponses dev, impossible de traiter les sodages qui sont un peu flous derrière.
        for reponse_t in reponses_dev:
            traitement_reponse_dev = True
            for bouton_fermer_t in boutons_fermer_reponse:
                if bouton_fermer_t.est_contenu_dans(reponse_t.position) > threshold_incl:
                    reponse_t.ajouter_fils(bouton_fermer_t)
                    bouton_fermer_t.donner_parent(reponse_t)
                    if verbose:
                        print(f"Bouton fermer réponse {bouton_fermer_t.id} est un fils de réponse {reponse_t.id}")
            for personne_sondee_t in personnes_sondees:
                if personne_sondee_t.est_contenu_dans(reponse_t.position) > threshold_incl:
                    reponse_t.ajouter_fils(personne_sondee_t)
                    personne_sondee_t.donner_parent(reponse_t)
                    if verbose:
                        print(f"Personne sondée {personne_sondee_t.id} est un fils de réponse {reponse_t.id}")

        # On ne traite les sondages que si on n'a pas de reponse dev
        if not traitement_reponse_dev:
            for sondage_t in sondages:
                for option_t in options:
                    if option_t.est_contenu_dans(sondage_t.position) > threshold_incl:
                        sondage_t.ajouter_fils(option_t)
                        option_t.donner_parent(sondage_t)
                        if verbose:
                            print(f"Option {option_t.id} est un fils de sondage {sondage_t.id}")
                for auteur_t in auteurs:
                    if auteur_t.est_contenu_dans(sondage_t.position) > threshold_incl:
                        sondage_t.ajouter_fils(auteur_t)
                        auteur_t.donner_parent(sondage_t)
                        if verbose:
                            print(f"Auteur {auteur_t.id} est un fils de sondage {sondage_t.id}")
                for bouton_voir_tout_t in boutons_voir_tout:
                    if bouton_voir_tout_t.est_contenu_dans(sondage_t.position) > threshold_incl:
                        sondage_t.ajouter_fils(bouton_voir_tout_t)
                        bouton_voir_tout_t.donner_parent(sondage_t)
                        if verbose:
                            print(f"Bouton voir tout {bouton_voir_tout_t.id} est un fils de sondage {sondage_t.id}")
            
            for option_t in options:
                for voir_t in voir_reponses:
                    if voir_t.est_contenu_dans(option_t.position) > threshold_incl:
                        option_t.ajouter_fils(voir_t)
                        voir_t.donner_parent(option_t)
                        if verbose:
                            print(f"Voir réponses {voir_t.id} est un fils de option {option_t.id}")

    def sondage_mode(self):
        # Renvoie True si la composition d'écran est un sondage, False sinon
        for racine in self.get_racines_sondage():
            if racine.is_sondage():
                return True
        return False
    
    def reponse_dev_mode(self):
        # Renvoie True si la composition d'écran est une réponse dev, False sinon
        for racine in self.get_racines_reponse_dev():
            if racine.is_reponse_dev():
                return True
        return False

    def get_racines_sondage(self):
        # Renvoie les racines (mais que les sondages)
        racines = []
        for composant in self.composants:
            if composant.is_sondage() and not composant.a_un_parent():
                racines.append(composant)
        return racines
    
    def get_racines_reponse_dev(self):
        # Renvoie les racines (mais que les réponses dev)
        racines = []
        for composant in self.composants:
            if composant.is_reponse_dev() and not composant.a_un_parent():
                racines.append(composant)
        return racines

    def debug_imprimer_arbre_composants(self):
        # Debug exclusivement, imprime l'arborescence des composants
        # On part de chaque composant qui n'a pas de parent, et on déroule ensuite de fils en fils

        for cpst in self.get_all_composants():
            if not cpst.a_un_parent():
                print(f"Composant orphelin: {cpst.__class__.__name__} (id {cpst.id})")
                cpst.debug_print_composant(indent=2)

            
            



class composant:
    """ Classe générique sur la base de laquelle on construira les composants de la composition d'écran."""

    def __init__(self, position, confidence=0.0):
        """ Arguments:
        position: (xcenter, ycenter, width, height) - Component position in the frame."""
        self.id = id_cooker.get_instance().get_new_id()
        self.position = position
        self.confidence = confidence
        self.fils = []
        self.parent = None
        self.is_init = False # dit si un composant a été initialisé ou non

    
    def is_bouton_voir_tout(self):
        return False
    
    def is_bouton_fermer_reponse(self):
        return False
    
    def is_option_reponse(self):
        return False
    
    def is_personne_sondee(self):    
        return False
    
    def is_reponse_dev(self):
        return False
    
    def is_sondage(self):
        return False
    
    def is_voir_reponses_option(self):
        return False
    
    def is_auteur_sondage(self):
        return False
    
    def ajouter_fils(self, fils):
        self.fils.append(fils)

    def donner_parent(self, composant):
        # initialise le parent du composant en question
        self.parent = composant

    def est_contenu_dans(self, autre_position):
        # Renvoie La proportion du composant qui est contenue si le composant en question est contenu dans la bbox passéen en arguments
        # En gros c'est une IOU

        x1, y1, w1, h1 = self.position
        x2, y2, w2, h2 = autre_position
  
        x1min = x1 - w1 / 2
        y1min = y1 - h1 / 2
        x1max = x1 + w1 / 2
        y1max = y1 + h1 / 2

        x2min = x2 - w2 / 2
        y2min = y2 - h2 / 2
        x2max = x2 + w2 / 2
        y2max = y2 + h2 / 2

        #Coord intersections
        xmin = max(x1min, x2min)
        ymin = max(y1min, y2min)
        xmax = min(x1max, x2max)
        ymax = min(y1max, y2max)

        if xmax <= xmin or ymax <= ymin:
            return 0 # Pas d'intersection

        intersection_area = (xmax - xmin) * (ymax - ymin)
        rect1_area = w1 * h1
        rect2_area = w2 * h2
        union_area = rect1_area + rect2_area - intersection_area

        contenu = intersection_area / (w1*h1)
        return contenu

    
    def verifier_integrite(self):
        # Vérifie si le composant est intègre et a bien été initialisé
        # Si la fonction est appellée ici, on a un problème
        AssertionError("Impossible de vérifier l'intégrité d'un composant générique")
    
    def get_init_status(self):
        # Vérifie si le composant a été initialisé
        return self.is_init
    
    def set_init_status(self, status):
        # Met à jour le statut d'initialisation du composant
        self.is_init = status

    def a_un_parent(self):
        return not self.parent is None
    
    def debug_print_composant(self, indent = 0):
        # Imprime le composant et ses fils
        print(f"{' '*indent}{self.__class__.__name__} (id {self.id})")
        if len(self.fils) > 0:
            new_indent = indent + 2
            for child in self.fils:
                child.debug_print_composant(new_indent)

    def debug_print_composant_self(self):
        # Imprime juste le composant
        print(f"{self.__class__.__name__} (id {self.id})")

    



class bouton_voir_tout(composant):

    def __init__(self, position, confidence=0.0):
        super().__init__(position, confidence)

    def is_bouton_voir_tout(self):
        return True
    
    def verifier_integrite(self):
        return (len(self.fils) == 0) and self.parent.is_sondage()
    

class bouton_fermer_reponse(composant):

    def __init__(self, position, confidence=0.0):
        super().__init__(position, confidence)

    def is_bouton_fermer_reponse(self):
        return True
    
    def verifier_integrite(self):
        return (len(self.fils) == 0) and self.parent.is_reponse_dev()


class option_reponse(composant):
    
    def __init__(self, position, confidence=0.0):
        super().__init__(position, confidence)

    def is_option_reponse(self):
        return True
    
    def verifier_integrite(self):
        return (len(self.fils) == 1) and self.parent.is_sondage() and self.fils[0].is_is_voir_reponses_option()
        
class personne_sondee(composant):
    
    def __init__(self, position, confidence=0.0):
        super().__init__(position, confidence)

    def is_personne_sondee(self):
        return True
    
    def verifier_integrite(self):
        return (len(self.fils) == 0) and self.parent.is_reponse_dev()
        
class reponse_dev(composant):
    
    def __init__(self, position, confidence=0.0):
        super().__init__(position, confidence)

    def is_reponse_dev(self):
        return True
    
    def verifier_integrite(self):
        bfr = False
        for fils in self.fils:
            if fils.is_bouton_fermer_reponse():
                bfr = True
        return (len(self.fils) >= 1) and bfr
        
class sondage(composant):
    
    def __init__(self, position, confidence=0.0):
        super().__init__(position, confidence)

    def is_sondage(self):
        return True
    
    def donner_parent(self, composant):
        AssertionError("Un sondage ne peut contenir de parent.")

    def verifier_integrite(self):
        auteur = False
        option = False
        for fils in self.fils:
            if fils.is_auteur_sondage():
                auteur = True
            if fils.is_option_reponse():
                option = True
        return (len(self.fils) >= 2) and auteur and option
        
        
class voir_reponses_option(composant):
        
    def __init__(self, position, confidence=0.0):
        super().__init__(position, confidence)
    
    def is_voir_reponses_option(self):
        return True
    
    def verifier_integrite(self):
        return (len(self.fils) == 0) and self.parent.is_option_reponse()
    

class auteur_sondage(composant):
    
    def __init__(self, position, confidence=0.0):
        super().__init__(position, confidence)

    def is_auteur_sondage(self):
        return True
    
    def verifier_integrite(self):
        return (len(self.fils) == 0) and self.parent.is_sondage()
            



