 
import cv2
import os
import random
import argparse

def extract_frames(video_path, output_dir, nb_frames=50, name=0):
  if not os.path.exists(output_dir):
      os.makedirs(output_dir)

  cap = cv2.VideoCapture(video_path)
  if not cap.isOpened():
      print("Error opening video file")
      return

  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  if total_frames <= 0:
      # Fallback si le header WebM est mal lu
      print("Erreur : Impossible de déterminer le nombre de frames.")
      return

  # Tri des indices obligatoire pour optimiser le seek de OpenCV
  frame_indices = sorted(random.sample(range(total_frames), min(nb_frames, total_frames)))
  print(frame_indices)

  nom_frame = name
  for idx in frame_indices:
      cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
      ret, frame = cap.read()
      if not ret:
          # Si le seek échoue (fréquent en WebM), on passe à la suivante
          continue

      frame_name = f"frame_{nom_frame}.jpg"
      output_path = os.path.join(output_dir, frame_name)
      cv2.imwrite(output_path, frame)
      print(f"Saved {output_path}")
      nom_frame += 1

  cap.release()
  print("Extraction finito")
  return nom_frame

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Sépare une vidéo en frames pour annotation.")
  parser.add_argument("--video", type=str, default="train.mov", help="Chemin vers vidéo")
  parser.add_argument("--out_dir", type=str, default="framesep_out", help="Chemin dossier de sortie")
  parser.add_argument("--image_count", type=int, default="300", help="Nombre d'images à sélectionner")
  args = parser.parse_args()

  extract_frames(args.video, args.out_dir, args.image_count)
