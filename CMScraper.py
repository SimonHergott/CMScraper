import cv2
import numpy as np
import dotenv
from composition_ecran import composition_ecran, composant, auteur_sondage, bouton_fermer_reponse, bouton_voir_tout, option_reponse, personne_sondee, reponse_dev, sondage, voir_reponses_option
from ultralytics import YOLO
import os
from sondage import sondage_m, option_m
import json
import re
from Levenshtein import distance as levenshtein_distance
import time
import sys
from data_helper import correlation_txt, write_to_json_file, PeopleDatabase
from ocr import image_to_string_vlm
import mss
import pyautogui


def verbose(level):
    # Factory de décorateurs spécifiques à la verbose.
    # Niveau de verbose contenu entre 0 (minimal) et 3 (maximal), cf. .env pour régler le niveau de verbose.
    def decorator(func):
        def wrapper(*args, **kwargs):
            global current_verbosity_level
            if current_verbosity_level >= level:
                return func(*args, **kwargs)
            else:
                #print(f"Fonction '{func.__name__}' ignorée verbose faible")
                return None
        return wrapper
    return decorator

_sct = None # var globales de persistance de la session mss
_monitor = None

def read_screen(debug = False, i = -1):
    global _sct, _monitor
    
    if debug and i == -1:
        return cv2.imread("test_frame.png")
    elif debug and i >= 0:
        video_path = "test_short.mov"
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None  
        return frame
    
    else:
        if _sct is None:
            os.environ["DISPLAY"] = ":10.0"
            _sct = mss.MSS()
            _monitor = _sct.monitors[1]

        try:
            screen_shot = _sct.grab(_monitor)
        except mss.exception.ScreenShotError:
            _monitor = _sct.monitors[1]
            screen_shot = _sct.grab(_monitor)

        frame = np.array(screen_shot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        mid = frame.shape[1] // 2
        frame[:, mid:, :] = 0
        return frame
    
    
def class_id_to_name(id):
    # convertit l'id de classe en une str compréhensible.
    class_names = {
        0: "auteur_sondage",
        1: "bouton_fermer_reponse",
        2: "bouton_voir_tout",
        3: "option_reponse",
        4: "personne_sondee",
        5: "reponse_dev",
        6: "sondage",
        7: "voir_reponses_option",
    }

    return class_names[id]

    
def analyse_frames(frame):
    # Analyse la frame et renvoie un objet composition_ecran
    compo = composition_ecran(frame)
    bboxes = detecter_bboxes(frame)

    # Tri des bboxes: on applique des indices de confiance différents selon le type de composant
    conf = bboxes.conf
    cls = bboxes.cls
    xywh = bboxes.xywh

    # Indices de confiance par classe
    conf_thresholds = {
        0: float(os.getenv("CONF_AUTEUR_SONDAGE", os.getenv("INDICE_CONF_AUTEUR", "0.5"))),
        1: float(os.getenv("CONF_BOUTON_FERMER_REPONSE", os.getenv("INDICE_CONF", "0.7"))),
        2: float(os.getenv("CONF_BOUTON_VOIR_TOUT", os.getenv("INDICE_CONF", "0.7"))),
        3: float(os.getenv("CONF_OPTION_REPONSE", os.getenv("INDICE_CONF", "0.7"))),
        4: float(os.getenv("CONF_PERSONNE_SONDEE", os.getenv("INDICE_CONF", "0.7"))),
        5: float(os.getenv("CONF_REPONSE_DEV", os.getenv("INDICE_CONF", "0.7"))),
        6: float(os.getenv("CONF_SONDAGE", os.getenv("INDICE_CONF", "0.7"))),
        7: float(os.getenv("CONF_VOIR_REPONSES_OPTION", os.getenv("INDICE_CONF", "0.7"))),
    }

    # Filtrage par confiance
    high_conf_indices = []
    for i, class_id in enumerate(cls):
        seuil = conf_thresholds.get(int(class_id), float(os.getenv("INDICE_CONF", "0.7")))
        if conf[i] > seuil:
            high_conf_indices.append(i)

    filtered_boxes = xywh[high_conf_indices]
    filtered_conf = conf[high_conf_indices]
    filtered_cls = cls[high_conf_indices]

    zip_boxes = zip(filtered_boxes, filtered_conf, filtered_cls)
    # Construction d'un objet composition_ecran
    for box in zip_boxes:
        compo.ajouter_composant(make_component(box))
    
    compo.ordonner(threshold_incl=0.8, verbose=False)  # On range les composants les uns dans les autres

    return compo

def debug_exporter_composition_as_frame(composition, taillex, tailley):
    # Pour du debug, prend tout ce qui est visible sur la composition et renvoie une frame illustrative (taille en param)
    frame = np.zeros((tailley, taillex, 3), np.uint8)
    for composant in composition.get_all_composants():
        x, y, w, h = composant.position
        x1, y1, w1, h1 = int(x-w/2), int(y-h/2), int(x+w/2), int(y+h/2)
        cv2.rectangle(frame, (x1, y1), (w1, h1), (255, 255, 255), 1)
        cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)
        text = f"{composant.__class__.__name__} id:{composant.id} c:({int(x)},{int(y)}) conf:{composant.confidence:.2f}"
        cv2.putText(frame, text, (x1 + 3, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return frame

def debug_overlay_frame_composition(composition, frame, taillex, tailley, alpha_dim=0.4, alpha_overlay=1.0):
    output_frame = frame.copy()
    output_frame = cv2.convertScaleAbs(output_frame, alpha=alpha_dim)
    overlay_annotations = np.zeros_like(output_frame, dtype=np.uint8)

    for composant in composition.get_all_composants():
        x, y, w, h = composant.position
        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)
        cv2.rectangle(overlay_annotations, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.circle(overlay_annotations, (int(x), int(y)), 3, (0, 0, 255), -1)
        text = f"{composant.__class__.__name__} id:{composant.id} c:({int(x)},{int(y)}) conf:{composant.confidence:.2f}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_x = x1 + 3
        text_y = y1 - 5 if y1 - 5 > text_size[1] + 3 else y1 + text_size[1] + 3
        if text_y < text_size[1]:
            text_y = y2 + text_size[1] + 3
        cv2.putText(overlay_annotations, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    final_frame = cv2.addWeighted(output_frame, 1.0, overlay_annotations, alpha_overlay, 0)
    if final_frame.shape[1] != taillex or final_frame.shape[0] != tailley:
        final_frame = cv2.resize(final_frame, (taillex, tailley))
    return final_frame


def afficher_frame(frame):
    # Affiche la frame
    cv2.imshow("Frame", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def enregistrer_frame(frame, path):
    # Enregistre la frame
    cv2.imwrite(path, frame)
    print(f"Frame enregistrée à {path}")

@verbose(level=3)
def enregistrer_frame_dans_video(frame, path):
    # Met al frame à la suite de la vidéo indiquée. Si la vidéo n'existe pas (frame 0), on la crée
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = 30.0
    width = frame.shape[1]
    height = frame.shape[0]
    video = cv2.VideoWriter(path, fourcc, fps, (width, height))
    video.write(frame)
    video.release()
    print(f"Video màj à {path}")

def make_component(box_detail):
    # Box: (tensor([1112.8577,  704.6846,  207.5119,   53.7666]), tensor(0.9690), tensor(4.))
    # On convertit les tensors en tuples
    box = tuple(box_detail[0].tolist())
    confidence = box_detail[1]
    classe = int(box_detail[2])

    if classe == 0: # auteur_sondage
        component = auteur_sondage(box, confidence)
    elif classe == 1: # bouton_fermer_reponse
        component = bouton_fermer_reponse(box, confidence)
    elif classe == 2: # bouton_voir_tout
        component = bouton_voir_tout(box, confidence)
    elif classe == 3: # option_reponse
        component = option_reponse(box, confidence)
    elif classe == 4: # personne_sondee
        component = personne_sondee(box, confidence)
    elif classe == 5: # reponse_dev
        component = reponse_dev(box, confidence)
    elif classe == 6: # sondage
        component = sondage(box, confidence)
    elif classe == 7: # voir_reponses_option
        component = voir_reponses_option(box, confidence)

    return component

def detecter_bboxes(frame):
    model_path = os.getenv("MODEL")
    model = YOLO(model_path)
    results = model(frame, verbose=False)
    return results[0].boxes

def OCR(composant_graph, frame):
    # Extraction géométrique du composant
    x, y, w, h = composant_graph.position
    x1, y1 = max(0, int(x - w / 2)), max(0, int(y - h / 2))
    x2, y2 = min(frame.shape[1], int(x + w / 2)), min(frame.shape[0], int(y + h / 2))
    
    cropped_frame = frame[y1:y2, x1:x2]

    if composant_graph.is_sondage():
        return image_to_string_vlm(cropped_frame, context_type="poll")

    elif composant_graph.is_auteur_sondage():
        return image_to_string_vlm(cropped_frame, context_type="name")
    
    elif composant_graph.is_option_reponse():
        return image_to_string_vlm(cropped_frame, context_type="text")
    
    elif composant_graph.is_voir_reponses_option():
        res = image_to_string_vlm(cropped_frame, context_type="percentage")
        try:
            return int(''.join(filter(str.isdigit, res)))
        except:
            return None
        
    elif composant_graph.is_personne_sondee():
        return image_to_string_vlm(cropped_frame, context_type="name")

    return image_to_string_vlm(cropped_frame, context_type="text")
    

def correlation_sondage(sondageA, sondageB, seuil = 0.3):
    # Utilisation de la distance de Levenshtein pour déterminer si sondageA = sondageB
    # On compare les descriptions, auteurs et options de réponse
    if sondageA is None or sondageB is None:
        return False

    if not correlation_txt(sondageA.get_description(), sondageB.get_description(), seuil*2):
        return False
    
    if not correlation_txt(sondageA.get_auteur(), sondageB.get_auteur(), seuil):
        return False
    
    return True # On considère que deux sondages sont identiques s'ils portent le même auteur et la même description

    
def verifier_vision_sondage(composant_graph, frame, padding_minimal = 50):
    # Vérifie si le sondage est visible au complet
    # On vérifie que le composant est un sondage, et quu'il y a du padding au dessous du sondage (ou alors un nouveau sondage)
    if composant_graph.is_sondage():
        # On vérifie que le composant est un sondage
        # On vérifie qu'il y a du padding au dessous du sondage (ou alors un nouveau sondage)
        x, y, w, h = composant_graph.position
        x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
        
        # Vérifier si le sondage est visible au complet
        if not (y2 < frame.shape[0] - padding_minimal):  # Si le bas du sondage est à moins de 50 pixels du bas de l'écran
            # Vérifier s'il y a un autre sondage en dessous
            for composant in composant_graph.fils:
                if composant.is_sondage():
                    x2_s, y2_s, w2_s, h2_s = composant.position
                    x1_s, y1_s, x2_s, y2_s = int(x2_s - w2_s / 2), int(y2_s - h2_s / 2), int(x2_s + w2_s / 2), int(y2_s + h2_s / 2)
                    if y1 < y1_s:
                        return False

        
        # Verification que le sondage possède un auteur et un contenu
        auteur = [fils for fils in composant_graph.fils if fils.is_auteur_sondage()]
        if not auteur:
            print("Sondage sans auteur")
            return False
        
        contenu = [fils for fils in composant_graph.fils if fils.is_option_reponse()]
        if not contenu:
            print("Sondage sans contenu")
            return False
        
        # Si on arrive ici, le sondage est complet
        return True
    return False

def nettoyer_sondages(pack_de_sondages_orig):
    """ Nettoie une liste de sondage_m en enlevant:
        - Les option svues une seule fois
    Renvoie une copie!"""
    pack_de_sondages = pack_de_sondages_orig.copy()
    for sondage in pack_de_sondages:
        options_to_remove = []
        for option in sondage.options:
            if option.description.number_of_versions() == 1 and len(option.get_description()) <= 1:
                options_to_remove.append(option)
        for option in options_to_remove:
            sondage.options.remove(option)
    return pack_de_sondages



def scroll_down(scroll_type, goto_x=None, goto_y=None):
    """
    Scrolle vers le bas, en fonction du type de scroll.
    goto_x et goto_y sont optionnels, si fournis, on déplace la souris avant de scroller.
    """
    if goto_x is not None and goto_y is not None:
        move_mouse_to(goto_x, goto_y)

    try:
        if IS_MACOS or IS_XORG:
            if scroll_type == "small":
                #print("Small scroll")
                pyautogui.scroll(-1)
            elif scroll_type == "big":
                #print("Big scroll")
                pyautogui.scroll(-1)
                pyautogui.scroll(-1)
                pyautogui.scroll(-1)
            else:
                raise ValueError("Scroll inconnu")
            time.sleep(1.5)
        elif IS_WAYLAND:
            exit(1)
    except Exception as e:
        print(f"Erreur inattendue lors du scroll: {e}")


def simulate_click(x_screen, y_screen, button='left', clicks=1, interval=0.1):
    """
    Simule un clic de souris à la position donnée.
    """
    print(f"Cliquage à ({x_screen}, {y_screen}) avec le bouton {button}, {clicks} fois.")

    pyautogui.click(x=x_screen, y=y_screen, button=button, clicks=clicks, interval=interval)
    #time.sleep(10) # debug

def move_mouse_to(x_screen, y_screen, duration=0):
    """
    Déplace la souris à la position (x_screen, y_screen).
    Pas de duration pour wayland
    """
    #print(f"Déplacement de la souris à ({x_screen}, {y_screen})")
    pyautogui.moveTo(x_screen, y_screen, duration=duration)

def exporter_sondages(sondages_global, path="debug/sondages.json"):
    # Export des sondages en JSON
    sondages = nettoyer_sondages(sondages_global)
    data = {"sondages": [sondage.to_dict_smpl() for sondage in sondages]}
    write_to_json_file(data, path)
        
def lire_repondants_pour_option(option_component, option_model):
    """
    Ouvre la vue 'réponse développée' pour l'option donnée, lit tous les répondants,
    les associe à option_model, puis ferme la vue. Marque l'option comme scannée bitwise
    """

    if option_model.has_been_scanned():
        print("[OPTION] Skip : deja scannée")
        return
    
    # Clic bouton voir rep option
    voir_rep_op = [vro for vro in getattr(option_component, "fils", []) if vro.is_voir_reponses_option()]
    if not voir_rep_op:
        print("[OPTION] Skip: Pas de bouton 'voir_reponses_option'")
        return

    vro_component = voir_rep_op[0]
    x, y, w, h = vro_component.position
    print(f"[OPTION] Clic 'voir_reponses_option' à ({x}, {y})")
    simulate_click((x + WINDOW_TOP_LEFT_X)*OFFSET_FACTOR_X, (y + WINDOW_TOP_LEFT_Y)*OFFSET_FACTOR_Y)
    time.sleep(0.5)

    # Lcture list
    prev_ids = set()
    while True:
        frame = read_screen()
        if frame is None:
            break
        composition = analyse_frames(frame)

        if not composition.reponse_dev_mode():
            # Pb de vue qui met du temps à s'ouvrir
            time.sleep(0.3)
            frame = read_screen()
            if frame is None:
                break
            composition = analyse_frames(frame)
            if not composition.reponse_dev_mode():
                print("[OPTION] Mode reponse_dev non détecté après clic, abandon lecture répondants.")
                break

        reponses_dev_graphs = composition.get_racines_reponse_dev()
        if not reponses_dev_graphs:
            print("[OPTION] Aucune racine reponse_dev détectée.")
            break

        reponses_dev_graph = reponses_dev_graphs[0]

        current_ids = []
        for compo in reponses_dev_graph.fils:
            if compo.is_personne_sondee():
                personne = OCR(compo, frame)
                if not personne:
                    continue
                db = PeopleDatabase.get_instance()
                pid = db.get_id_from_name(personne)
                if pid is None:
                    pid = db.add_person_from_name(personne)
                option_model.ajouter_respondent(pid)
                current_ids.append(pid)

        if set(current_ids) == prev_ids:
            print("[OPTION] Fin de la liste des répondants pour cette option.")
            break

        prev_ids = set(current_ids)
        # Scroll dans le panneau rep dev
        rx, ry, rw, rh = reponses_dev_graph.position
        move_mouse_to(rx, ry)
        scroll_down("big")
        time.sleep(0.3)

    # Ferme + marque scannée
    frame = read_screen()
    if frame is not None:
        composition = analyse_frames(frame)
        bouton_fermer = [c for c in composition.get_all_composants() if c.is_bouton_fermer_reponse()]
        if bouton_fermer:
            bx, by, bw, bh = bouton_fermer[0].position
            simulate_click((bx + WINDOW_TOP_LEFT_X)*OFFSET_FACTOR_X, (by + WINDOW_TOP_LEFT_Y)*OFFSET_FACTOR_Y)
            time.sleep(0.2)

    option_model.mark_scanned()



def main_loop(debug_mode=False, debug_folder="debug"):
    frame_index = 0
    sondages_global = []  # Ensemble des sondages qui ont été vus

    while True and frame_index < 100:
        frame = read_screen()
        frame_index += 1
        if frame is None:
            break
        h, w = frame.shape[:2]
        composition = analyse_frames(frame)

        print(f"\nPasse {frame_index} -- Frame lue, analyse en cours...\n")

        if debug_mode:
            if not os.path.exists(f"{debug_folder}/analyses"):
                os.makedirs(f"{debug_folder}/analyses")
            if not os.path.exists(f"{debug_folder}/raw"):
                os.makedirs(f"{debug_folder}/raw")
            enregistrer_frame(debug_exporter_composition_as_frame(composition, w, h), f"{debug_folder}/analyses/compo{frame_index}.png")
            enregistrer_frame(debug_overlay_frame_composition(composition, frame, w, h), f"{debug_folder}/analyses/overlay{frame_index}.png")
            enregistrer_frame(frame, f"{debug_folder}/raw/frame{frame_index}.png")

        # Si on se retrouve en mode reponse_dev hors séquence, on ferme proprement et on continue
        if composition.reponse_dev_mode():
            bouton_fermer = [c for c in composition.get_all_composants() if c.is_bouton_fermer_reponse()]
            if bouton_fermer:
                bx, by, bw, bh = bouton_fermer[0].position
                simulate_click((bx + WINDOW_TOP_LEFT_X)*OFFSET_FACTOR_X, (by + WINDOW_TOP_LEFT_Y)*OFFSET_FACTOR_Y)
                time.sleep(0.2)
            continue

        if composition.sondage_mode():
            sondages_graph = composition.get_racines_sondage()
            sg_complet = None
            for sg in sondages_graph:
                # Affichage complet
                bouton_voir_tout = [c for c in sg.fils if c.is_bouton_voir_tout()]
                if bouton_voir_tout:
                    bx, by, bw, bh = bouton_voir_tout[0].position
                    simulate_click((bx + WINDOW_TOP_LEFT_X) * OFFSET_FACTOR_X, (by + WINDOW_TOP_LEFT_Y) * OFFSET_FACTOR_Y)
                    time.sleep(0.2)
                    break

                # verif complétude
                if verifier_vision_sondage(sg, frame):
                    sg_complet = sg
                    print(f"[SONDAGE] Sondage complet trouvé: {sg.id}")
                    break

            if sg_complet is None:
                print(f"[SONDAGE] Sondage incomplet, on scrolle un peu.")
                scroll_down("small")
                continue

            # Lecture dscr + auteur
            sm = sondage_m()
            descr = OCR(sg_complet, frame)
            if not descr:
                print("[SONDAGE] Sondage sans description, ignoré.")
                scroll_down("big")
                continue
            sm.ajouter_description(descr)

            auteurs = [f for f in sg_complet.fils if f.is_auteur_sondage()]
            if not auteurs:
                print("[SONDAGE] Sondage sans auteur, ignoré.")
                scroll_down("big")
                continue
            sm.ajouter_auteur(OCR(auteurs[0], frame))

            # Construction des options
            option_pairs = [] # paire composant - model, pour pouvoir lire ensuite la liste
            options_components = [f for f in sg_complet.fils if f.is_option_reponse()]
            for opt_comp in options_components:
                om = option_m()
                descr_option = OCR(opt_comp, frame)
                om.ajouter_description(descr_option)

                voir_rep_op = [vro for vro in opt_comp.fils if vro.is_voir_reponses_option()]
                if len(voir_rep_op) == 1:
                    om.taux = OCR(voir_rep_op[0], frame)

                option_deja_vue = sm.ajouter_option(om) # renvoie l'instance de l'option si elle a déjà été vue
                
                if option_deja_vue is not None:
                    option_pairs.append((opt_comp, option_deja_vue)) # Match du composant à l'option deja vue
                else:
                    option_pairs.append((opt_comp, om)) # Pas de match à option deja vue, nouvelle option
                    print(f"[SONDAGE] Nouvelle option créée: {descr_option}")

            # Fusion avec un sondage précédent si nécessaire
            sondage_m_prec = None
            for s_prec in sondages_global[-10:]:
                if correlation_sondage(s_prec, sm):
                    sondage_m_prec = s_prec
                    print("[SONDAGE] Sondage précédent trouvé, mise à jour.")
                    break

            if sondage_m_prec is not None:
                for desc in sm.description.get_all_versions():
                    sondage_m_prec.ajouter_description(desc)
                # Ajouter/mettre à jour options (ne sert en théorie à rien, TODO virer)
                for _, om in option_pairs:
                    sondage_m_prec.ajouter_option(om)
                sm_courant = sondage_m_prec
            else:
                print(f"[SONDAGE] Nouveau sondage: {sm.get_description()} — auteur: {sm.get_auteur()}")
                sondages_global.append(sm)
                sm_courant = sm

            # Remap de chaque paire vers l'instance autoritative stockée dans sm_courant
            option_pairs_mapped = []
            for opt_comp, om_tmp in option_pairs:
                om_auth = sm_courant.ajouter_option(om_tmp)
                option_pairs_mapped.append((opt_comp, om_auth))
            option_pairs = option_pairs_mapped

            # Lecture de toutes les options avant de scroller
            for opt_comp, om in option_pairs:
                print(f"[OPTION] Option {om.get_description()} - nombre de répondants connus: {len(om.respondents)}")
                if not om.has_been_scanned():
                    lire_repondants_pour_option(opt_comp, om)

            # Une fois toutes les options parcourues, on peut scroller
            scroll_down("small")
        else:
            # Aucun mode détecté -> petit scroll pour avancer
            scroll_down("small")

        exporter_sondages(sondages_global, "debug/sondages.json") # gourmandise
        PeopleDatabase.get_instance().export_to_json("debug/people_db.json")

    # Export
    exporter_sondages(sondages_global, "debug/sondages.json")
    PeopleDatabase.get_instance().export_to_json("debug/people_db.json")
                


if __name__ == "__main__":

    dotenv.load_dotenv()

    # Offsets de debug de la fenetre navigateur capturée
    WINDOW_TOP_LEFT_X = 0 # pos x=0
    WINDOW_TOP_LEFT_Y = 0 # pos y = 0

    # Correction à la main TODO: capter d'ou ca vient
    OFFSET_DEBUG_Y = 0 # -100 
    OFFSET_FACTOR_X = 1 #0.93 # Vraie position selon X = position détectée * OFFSET_FACTOR_X
    OFFSET_FACTOR_Y = 1 #0.94 # Idem

    # Compatibilité Wayland + Xorg
    IS_MACOS = sys.platform == 'darwin'
    IS_LINUX = sys.platform == 'linux'
    IS_XORG = False

    if IS_LINUX:
        xdg_session_type = os.environ.get('XDG_SESSION_TYPE')
        if xdg_session_type == 'x11':
            IS_XORG = True
            print("Environnement: Xorg")
        else:
            print("Environnement Linux inconnu, voir XDG_SESSION_TYPE")
            print("Default à Xorg")
            IS_XORG = True # Tente PyAutoGUI par défaut sur Linux si non spécifié
    elif IS_MACOS:
        print("Environnement: macOS")
    else:
        print(f"Système inconnu {sys.platform}")
        sys.exit(1)


    current_verbosity_level = int(os.getenv("VERBOSE", "0"))
    debug_mode = os.getenv("DEBUG", "false").lower() == "true"
    debug_folder = os.getenv("DEBUG_FOLDER", "debug")

    main_loop(debug_mode=debug_mode, debug_folder=debug_folder)

    print("Finito")







