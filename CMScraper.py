import cv2
import numpy as np
import dotenv
from composition_ecran import composition_ecran, composant, auteur_sondage, bouton_fermer_reponse, bouton_voir_tout, option_reponse, personne_sondee, reponse_dev, sondage, voir_reponses_option
from database import database_helper_names
from ultralytics import YOLO
import os
import pytesseract
from sondage import sondage_m, option_m
import json
import re
from Levenshtein import distance as levenshtein_distance
import time
import sys
from data_helper import correlation_txt, write_to_json_file, PeopleDatabase


def verbose(level):
    # Factory de décorateurs spécifiques à la verbose.
    # Niveau de verbose contenu entre 0 (minimal) et 3 (maximal), cf. .env pour régler le niveau de verbose.
    def decorator(func):
        def wrapper(*args, **kwargs):
            global current_verbosity_level
            if current_verbosity_level >= level:
                return func(*args, **kwargs)
            else:
                print(f"Fonction '{func.__name__}' ignorée verbose faible")
                return None
        return wrapper
    return decorator


def read_screen(debug = False, i = -1):
    # Lit l"écran et renvoie une frame
    # En débug, on utilise juste une frame qui traine
    if debug and i ==-1:
        return cv2.imread("test_frame.png")
    elif debug and i >= 0:
        # Lecture de la frame suivante (frame i) dans la vidéo de test
        video_path = "test_short.mov"
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None  
        return frame
    else:
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("Erreur lors de l'ouverture de la caméra")
            return None
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print("Erreur lors de la lecture de la frame")
            return None
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

    #tri des bboxes: on vire celles à <0.7 de confiance
    conf = bboxes.conf
    cls = bboxes.cls
    xywh = bboxes.xywh

    # Filter by confidence
    high_conf_indices = conf > float(os.getenv("INDICE_CONF"))
    filtered_boxes = xywh[high_conf_indices]
    filtered_conf = conf[high_conf_indices]
    filtered_cls = cls[high_conf_indices]

    zip_boxes = zip(filtered_boxes, filtered_conf, filtered_cls)
    # Construction d'un objet composition_ecran
    for box in zip_boxes:
        compo.ajouter_composant(make_component(box))
    
    compo.ordonner(threshold_incl=0.8, verbose=False) # On range les composants les uns dans les autres

    return compo

def debug_exporter_composition_as_frame(composition, taillex, tailley):
    # Pour du debug, prend tout ce qui est visible sur la composition et renvoie une frame illustrative (taille en param)
    frame = np.zeros((tailley, taillex, 3), np.uint8)
    for composant in composition.get_all_composants():
        x, y, w, h = composant.position
        x1, y1, w1, h1 = int(x-w/2), int(y-h/2), int(x+w/2), int(y+h/2)
        cv2.rectangle(frame, (x1, y1), (w1, h1), (255, 255, 255), 2)
        text_size = cv2.getTextSize(composant.__class__.__name__, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = x1 + 5
        text_y = y1 + text_size[1] + 5
        cv2.putText(frame, f"{composant.__class__.__name__}, id: {composant.id}, x:{x}, y:{y}", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
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
        cv2.rectangle(overlay_annotations, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{composant.__class__.__name__}, id: {composant.id}, x:{x}, y:{y}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = x1 + 5
        text_y = y1 - 10 if y1 - 10 > text_size[1] + 5 else y1 + text_size[1] + 5
        if text_y < text_size[1]:
            text_y = y2 + text_size[1] + 5
        cv2.putText(overlay_annotations, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    final_frame = cv2.addWeighted(output_frame, 1.0, overlay_annotations, alpha_overlay, 0)
    if final_frame.shape[1] != taillex or final_frame.shape[0] != tailley:
        final_frame = cv2.resize(final_frame, (taillex, tailley))
    return final_frame


def afficher_frame(frame):
    # Affiche la frame
    cv2.imshow("Frame", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

@verbose(level=2)
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
        component = auteur_sondage(box)
    elif classe == 1: # bouton_fermer_reponse
        component = bouton_fermer_reponse(box)
    elif classe == 2: # bouton_voir_tout
        component = bouton_voir_tout(box)
    elif classe == 3: # option_reponse
        component = option_reponse(box)
    elif classe == 4: # personne_sondee
        component = personne_sondee(box)
    elif classe == 5: # reponse_dev
        component = reponse_dev(box)
    elif classe == 6: # sondage
        component = sondage(box)
    elif classe == 7: # voir_reponses_option
        component = voir_reponses_option(box)

    return component

def detecter_bboxes(frame):
    model_path = os.getenv("MODEL")
    model = YOLO(model_path)
    results = model(frame)
    return results[0].boxes

def OCR(composant_graph, frame, confiance = 10):
    # On fait de l'OCR sur le composant précisément: selon le type, différentes stratégies

    c_frame = frame.copy() #évite les embrouilles par la suite

    #Réduction de la zone de la frame au composant
    if composant_graph.is_sondage():
        # Masquer l'auteur, et tout ce qui est plus bas que l'auteur
        x, y, w, h = composant_graph.position
        x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
        
        auteur = [element for element in composant_graph.fils if element.is_auteur_sondage() == True]

        if auteur:
            auteur = auteur[0]
            x_a, y_a, w_a, h_a = auteur.position
            x1_a, y1_a, x2_a, y2_a = int(x_a - w_a / 2), int(y_a - h_a / 2), int(x_a + w_a / 2)+200, int(y_a + h_a / 2) # TODO: trouver + propre que le +200 pour masquer la date et l'heure
            c_frame[y1_a:y2_a, x1_a:x2_a] = 0  # Masquer l'auteur

        reponses = [element for element in composant_graph.fils if element.is_option_reponse() == True]
        for reponse in reponses: # TODO: potentiellement redondant, amélioratioj possible en n'itérant que pour la réponse située le + haut
            x_r, y_r, w_r, h_r = reponse.position
            x1_r, y1_r, x2_r, y2_r = int(x_r - w_r / 2), int(y_r - h_r / 2), int(x_r + w_r / 2), int(x_r + h_r / 2)
            c_frame[y1_r:, :] = 0  # Masquer tout ce qui est à partir du début de la première réponse (et au dessous)

        cropped_frame = c_frame[y1:y2, x1:x2]

        # Quantization + upscaling
        scale_factor = 2
        upscaled_frame = cv2.resize(cropped_frame, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

        #enregistrer_frame(upscaled_frame, "debug/cropped_sondage.png") 

        data = pytesseract.image_to_data(upscaled_frame, output_type=pytesseract.Output.DICT)
        text = " ".join(data['text'][i] for i in range(len(data['text'])) if int(data['conf'][i]) > confiance)
        text = text.replace("\n", " ")
        return text

    elif composant_graph.is_auteur_sondage():
        x, y, w, h = composant_graph.position
        x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
        cropped_frame = frame[y1:y2, x1:x2]

        # Quantization + upscaling
        scale_factor = 2 
        upscaled_frame = cv2.resize(cropped_frame, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

        enregistrer_frame(upscaled_frame, "debug/cropped_auteur.png")

        text = pytesseract.image_to_string(upscaled_frame)
        # Lire seulement la première ligne
        first_line = text.split('\n')[0]
        return first_line
    
    elif composant_graph.is_option_reponse():
        x, y, w, h = composant_graph.position
        x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
        
        # On vire un peu de bordure en plu pour éviter des problèmes d'OCR
        x1_crop_margin = int(0.06 * w)
        # On masque des parties pour éviter d'avoir le % en réponse.
        cropped_frame_to_process = frame[y1:y2, x1 + x1_crop_margin:x2].copy()

        voir_reponses = [element for element in composant_graph.fils if element.is_voir_reponses_option() == True]
        if voir_reponses:
            voir_reponse = voir_reponses[0]
            x_vr, y_vr, w_vr, h_vr = voir_reponse.position
        
            x1_vr_rel = int(x_vr - w_vr / 2) - (x1 + x1_crop_margin)
            y1_vr_rel = int(y_vr - h_vr / 2) - y1
            x2_vr_rel = int(x_vr + w_vr / 2) - (x1 + x1_crop_margin)
            y2_vr_rel = int(y_vr + h_vr / 2) - y1

            x1_vr_rel = max(0, x1_vr_rel)
            y1_vr_rel = max(0, y1_vr_rel)
            x2_vr_rel = min(cropped_frame_to_process.shape[1], x2_vr_rel)
            y2_vr_rel = min(cropped_frame_to_process.shape[0], y2_vr_rel)
            
            if x1_vr_rel < x2_vr_rel and y1_vr_rel < y2_vr_rel:
                cropped_frame_to_process[y1_vr_rel:y2_vr_rel, x1_vr_rel:x2_vr_rel] = 255 # masquage blanc

        scale_factor = 2 
        upscaled_frame_for_ocr = cv2.resize(cropped_frame_to_process, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

        enregistrer_frame(upscaled_frame_for_ocr, "debug/cropped_option.png")

        data = pytesseract.image_to_data(upscaled_frame_for_ocr, output_type=pytesseract.Output.DICT)
        text = " ".join(data['text'][i] for i in range(len(data['text'])) if int(data['conf'][i]) > confiance)

        return text
    
    elif composant_graph.is_voir_reponses_option():
        x, y, w, h = composant_graph.position
        padding_x = int(w * 0.5)
        padding_y = int(h * 0.5) # 50% en + de chaque coté

        x1_padded = max(0, int(x - w / 2) - padding_x)
        y1_padded = max(0, int(y - h / 2) - padding_y)
        x2_padded = min(frame.shape[1], int(x + w / 2) + padding_x)
        y2_padded = min(frame.shape[0], int(y + h / 2) + padding_y)
        
        cropped_frame = frame[y1_padded:y2_padded, x1_padded:x2_padded]
      
        scale_factor = 2
        upscaled_frame = cv2.resize(cropped_frame, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC) 

        enregistrer_frame(upscaled_frame, "debug/taux.png")

        raw_text = pytesseract.image_to_string(upscaled_frame, config='--psm 7')
        match = re.search(r'(\d{1,2})%', raw_text) # Regex qui matche le taux
        if match:
            taux = int(match.group(1))
            return taux
        else:
            return None
        
    elif composant_graph.is_personne_sondee():
        x, y, w, h = composant_graph.position
        x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)

        # On mange qqs px sur la gauche pour éviter de capturer l'image de profil
        x1 +=50 # TODO: paramétrer et mettre dans config
        
        cropped_frame = frame[y1:y2, x1:x2]

        # Quantization + upscaling
        scale_factor = 2 
        upscaled_frame = cv2.resize(cropped_frame, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
        
        enregistrer_frame(upscaled_frame, "debug/cropped_personne.png")

        text = pytesseract.image_to_string(upscaled_frame)
        # Lire seulement la première ligne
        first_line = text.split('\n')[0]
        return first_line
    

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

def nettoyer_sondages(pack_de_sondages):
    """ Nettoie une liste de sondage_m en enlevant:
        - Les option svues une seule fois"""
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
                pyautogui.scroll(3)
            elif scroll_type == "big":
                pyautogui.scroll(6)
            else:
                raise ValueError("Scroll inconnu")
            time.sleep(1.5)
        elif IS_WAYLAND:
            if scroll_type == "small":
                _wayland_mouse_controller.scroll(-100) # Valeurs douteuses
            elif scroll_type == "big":
                _wayland_mouse_controller.scroll(-300)
            else:
                raise ValueError("Scroll inconnu")
    except Exception as e:
        print(f"Erreur inattendue lors du scroll: {e}")


def simulate_click(x_screen, y_screen, button='left', clicks=1, interval=0.1):
    """
    Simule un clic de souris à la position donnée.
    """
    print(f"Cliquage à ({x_screen}, {y_screen}) avec le bouton {button}, {clicks} fois.")
    if IS_MACOS or IS_XORG:
        pyautogui.click(x=x_screen, y=y_screen, button=button, clicks=clicks, interval=interval)
        #time.sleep(10) # debug
    elif IS_WAYLAND:
        move_mouse_to(x_screen, y_screen)
        for _ in range(clicks):
            if button == 'left':
                _wayland_mouse_controller.click("left")
            elif button == 'right':
                _wayland_mouse_controller.click("right")
            elif button == 'middle':
                _wayland_mouse_controller.click("middle")
            else:
                print(f"Bouton de souris '{button}' inconnu")
                break
            if clicks > 1:
                time.sleep(interval)

def move_mouse_to(x_screen, y_screen, duration=0):
    """
    Déplace la souris à la position (x_screen, y_screen).
    Pas de duration pour wayland
    """
    print(f"Déplacement de la souris à ({x_screen}, {y_screen})")
    if IS_MACOS or IS_XORG:
        pyautogui.moveTo(x_screen, y_screen, duration=duration)
    elif IS_WAYLAND:
        _wayland_mouse_controller.move(x_screen, y_screen)



    
def main_loop(screen_width, screen_height):
    
    frame_index = 0
    sondages_global = []  # Ensemble des sondages qui ont été vus

    # Variablede mémorisation réponse_dev
    last_names_reponse_dev = [] # Stocke les derniers ID de répondants. Utile pour vérifier la fin de scroll en reponse dev
    temp_last_names_reponse_dev = [] # Stocke les derniers noms de répondants. Utile pour vérifier la fin de scroll en reponse dev
    last_option_clicked = None #Sert pour l'association aux reponse_dev

    while True and frame_index < 100:  # Limite de frames pour éviter une boucle infinie
        # Analyse de l'écran
        frame = read_screen()
        frame_index += 1
        composition = analyse_frames(frame)

        print(f"\nPasse {frame_index} -- Frame lue, analyse en cours...\n")

        if frame is None:
            break  # passage à l'enregistrement des données

        # DEBUG -- enregistrement de la frame et de la composition
        if not os.path.exists("debug/analyses"):
            os.makedirs("debug/analyses")
        if not os.path.exists("debug/raw"):
            os.makedirs("debug/raw")
        enregistrer_frame(debug_exporter_composition_as_frame(composition, screen_width, screen_height), f"debug/analyses/compo{frame_index}.png")
        enregistrer_frame(debug_overlay_frame_composition(composition, frame, screen_width, screen_height), f"debug/analyses/overlay{frame_index}.png")
        enregistrer_frame(frame, f"debug/raw/frame{frame_index}.png")
        
        if composition.reponse_dev_mode():

            reponses_dev_graph = composition.get_racines_reponse_dev()[0]
            if reponses_dev_graph is None:
                print("UB: reponse dev graph non détectée mais mode reponse_dev actif")
                break

            end_of_resp_list = False
            while not end_of_resp_list:
                current_repondents_on_screen = [] # Pour vérifier qu'on est pas au bout

                for compo in reponses_dev_graph.fils: # Lecture et ajout des gens
                    if compo.is_personne_sondee():
                        personne = OCR(compo, frame)
                        if personne is None or personne == "":
                            print("Personne sondée sans nom, ignorée")
                            continue
                        else:
                            print(f"Personne sondée: {personne}")
                            id_personne = PeopleDatabase.get_instance().get_id_from_name(personne)
                            if id_personne is None:
                                print(f"Personne sondée non existante. Ajout à la BDD")
                                id_personne = PeopleDatabase.get_instance().add_person_from_name(personne)
                            else:
                                print(f"Personne sondée existante: {id_personne}")

                            current_repondents_on_screen.append(id_personne)

                            # Ajout de l'ID sur l'option qui correspond
                            if last_option_clicked:
                                last_option_clicked.ajouter_respondent(id_personne)
                            else:
                                print("Warning: last_option_clicked is None, cannot associate respondent.")

                # Comparaison des répondants précédents et actuels pour vérifier le scroll
                if set(current_repondents_on_screen) == set(temp_last_names_reponse_dev):
                    print("Fin de la liste des répondants!")
                    end_of_resp_list = True
                else:
                    # Si il y en a des nouveaux, on update la liste et on scrolle pour voir si on est au bout
                    temp_last_names_reponse_dev = current_repondents_on_screen.copy()
                    print(f"Scrollage en mode reponse_dev. Nouveaux: {len(current_repondents_on_screen)}")
                    # On place la souris au milieu de la reponse dev avant de scroller
                    x, y, w, h = reponses_dev_graph.position
                    print(f"Position de la reponse_dev: {x}, {y}, {w}, {h}")
                    move_mouse_to(x, y)
                    # On scrolle vers le bas
                    scroll_down("small")
                    frame = read_screen()
                    composition = analyse_frames(frame)
                    reponses_dev_graph = composition.get_racines_reponse_dev()[0]

                    if reponses_dev_graph is None:
                        print("UB si la liste de reponses dev est étonnamment vide")
                        end_of_resp_list = True

            temp_last_names_reponse_dev = []

            # Sortie de la boucle de lecture, fermeture de fenetre
            bouton_fermer = [compo for compo in composition.get_all_composants() if compo.is_bouton_fermer_reponse()]
            if bouton_fermer:
                bouton_fermer = bouton_fermer[0]
                x, y, w, h = bouton_fermer.position
                print(f"Position du bouton fermer: {x}, {y}, {w}, {h}")
                simulate_click((x + WINDOW_TOP_LEFT_X)*OFFSET_FACTOR_X, (y + WINDOW_TOP_LEFT_Y)*OFFSET_FACTOR_Y)
                print("Clic sur 'bouton_fermer_reponse'.")
                time.sleep(0.2)
                frame = read_screen()
                composition = analyse_frames(frame)
                last_option_clicked = None


        elif composition.sondage_mode(): # Evite d'etre en mode sondage et de rester "bloqué" entre 2 sondages non complets
            # Le sondage est-il complet?
            sondages_graph = composition.get_racines_sondage()
            sondage_complet = None 
            for sg in sondages_graph:
                if verifier_vision_sondage(sg, frame):
                    sondage_complet = sg.id
                    print(f"Sondage complet trouvé: {sg.id}")
                    break
                else:
                    print(f"Trouvé: sondage incomplet à la passe {frame_index}")
                    scroll_down("small")
                    continue

            if sondage_complet is not None:

                clicked_on_voir_reponses = False # Pilote le scroll plus tard

                # On clique sur le bouton "voir tout"
                bouton_voir_tout = [fils for fils in sg.fils if fils.is_bouton_voir_tout()]
                if bouton_voir_tout is not None and len(bouton_voir_tout)>=1:
                    if isinstance(bouton_voir_tout, list):
                        bouton_voir_tout = bouton_voir_tout[0]
                    x,y,w,h = bouton_voir_tout.position
                    print(bouton_voir_tout.position)
                    simulate_click((x + WINDOW_TOP_LEFT_X)*OFFSET_FACTOR_X, (y + WINDOW_TOP_LEFT_Y)*OFFSET_FACTOR_Y)

                # On lit l'auteur, son texte total et ses options de réponse
                sm = sondage_m()
                descr = OCR(sg, frame)
                if descr is None or descr == "":
                    print("Sondage sans description, ignoré")
                    scroll_down("small")
                    break
                else:
                    sm.ajouter_description(descr)

                auteur = [fils for fils in sg.fils if fils.is_auteur_sondage()]
                if auteur:
                    sm.ajouter_auteur(OCR(auteur[0], frame))
                else:
                    break # Pas d'auteur, on passe au sondage suivant
                
                options = [fils for fils in sg.fils if fils.is_option_reponse()]
                for option in options:
                    # Tout pourri quand le texte a 2 lignes, TODO refaire cette fonction
                    om = option_m()
                    descr_option = OCR(option, frame)
                    om.ajouter_description(descr_option)
                    # Recherche du taux
                    if option.fils is not None:
                        voir_rep_op = [vro for vro in option.fils if vro.is_voir_reponses_option()]
                        if len(voir_rep_op) == 1:
                            om.taux = OCR(voir_rep_op[0], frame)
                            print(f"Taux: {om.taux}")
                    option_deja_vue = sm.ajouter_option(om)

                    if not option_deja_vue:
                        print(f"Nouvelle option de réponse détectée: '{descr_option}'")
                        last_option_clicked = om 

                        if len(voir_rep_op) == 1:
                            vro_component = voir_rep_op[0]
                            x, y, w, h = vro_component.position
                            print(f"Position du bouton 'voir_reponses_option': ({x}, {y}), taille: ({w}, {h})")
                            print(f"Clic sur 'voir_reponses_option' pour '{descr_option}' à ({x}, {y}).")
                            simulate_click((x + WINDOW_TOP_LEFT_X)*OFFSET_FACTOR_X, (y + WINDOW_TOP_LEFT_Y)*OFFSET_FACTOR_Y)
                            
                            clicked_on_voir_reponses = True
                            time.sleep(0.5) # Délai pour laisser le temps de charger les options
                            break
                        else:
                            print(f"Pas de bouton 'voir_reponses_option' trouvé pour l'option '{descr_option}'.")
                    else:
                        print(f"Option de réponse déjà vue: '{descr_option}'")


                # On vérifie si le sondage existe déjà dans la base de données (on ne vérif que les 10 derniers)
                sondage_m_prec = None
                for sondage in sondages_global[-10:]:
                    if correlation_sondage(sondage, sm):
                        sondage_m_prec = sondage
                        print("Sondage precedent trouvé")
                        break

                if sondage_m_prec is not None: # Si on a trouvé un sondage précédent
                    # On met à jour le sondage existant
                    print(f"Sondage déjà existant: {sondage_m_prec.id}, mise à jour")
                    for desc in sm.description.get_all_versions(): # Ajout de la ou les descriptions vues
                        sondage_m_prec.ajouter_description(desc)

                    if len(sm.options) > len(sondage_m_prec.options):
                        for option in sm.options:
                            sondage_m_prec.ajouter_option(option) # sondage_m gère l'ajout des options et la comparaison avec celles qu'il a deja

                else: # Le sondage n'existe pas, ajout à la liste globale
                    print(f"Nouveau sondage trouvé: {sm.get_description()}, auteur: {sm.get_auteur()}")
                    sondages_global.append(sm)

                scroll_down("small")

            else: # Si on n'est ni en mode reponse ni en mode sondage (?)
                scroll_down("small")

    # Export
    sondages_global = nettoyer_sondages(sondages_global)
    data = {"sondages": [sondage.to_dict_smpl() for sondage in sondages_global]}
    write_to_json_file(data, "debug/sondages.json")
                


if __name__ == "__main__":

    dotenv.load_dotenv()


    # Offsets de debug de la fenetre navigateur capturée
    WINDOW_TOP_LEFT_X = 0 # pos x=0
    WINDOW_TOP_LEFT_Y = 25 # pos y = 0

    # Correction à la main TODO: capter d'ou ca vient
    OFFSET_DEBUG_Y = 0 # -100 
    OFFSET_FACTOR_X = 1 #0.93 # Vraie position selon X = position détectée * OFFSET_FACTOR_X
    OFFSET_FACTOR_Y = 1 #0.94 # Idem

    # Compatibilité Wayland + Xorg
    IS_MACOS = sys.platform == 'darwin'
    IS_LINUX = sys.platform == 'linux'
    IS_XORG = False
    IS_WAYLAND = False

    if IS_LINUX:
        xdg_session_type = os.environ.get('XDG_SESSION_TYPE')
        if xdg_session_type == 'wayland':
            IS_WAYLAND = True
            print("Environnement: Wayland")
        elif xdg_session_type == 'x11':
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

    if IS_MACOS or IS_XORG:
        try:
            import pyautogui
        except ImportError:
            print("Erreur import pyatogui")
            sys.exit(1)
    elif IS_WAYLAND:
        try:
            from wayland_automation.mouse_controller import Mouse
            _wayland_mouse_controller = Mouse()
        except ImportError:
            print("Erreur import wayland-automation")
            sys.exit(1)
        except Exception as e:
            print(f"Erreur init wayland-automation : {e}")
            sys.exit(1)


    current_verbosity_level = 2

    screen_width = int(os.getenv("RESOLUTION_WIDTH"))
    screen_height = int(os.getenv("RESOLUTION_HEIGHT"))

    print(f"Résolution de l'écran: {screen_width}x{screen_height}")

    main_loop(screen_width, screen_height)

    print("Finito")
    






