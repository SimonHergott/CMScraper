import shutil
import random
from pathlib import Path

from PIL import Image

DATASETS_DIR = Path(__file__).parent
OUTPUT_DIR = DATASETS_DIR / "merged_dataset"
VAL_RATIO = 0.1
CROP_MIN = 0.15
CROP_MAX = 0.25

CLASSES = [
    "auteur_sondage",
    "bouton_fermer_reponse",
    "bouton_voir_tout",
    "option_reponse",
    "personne_sondee",
    "reponse_dev",
    "sondage",
    "voir_reponses_option",
]

random.seed(0)


def crop_cms232(img_path: Path, label_path: Path | None, out_img: Path, out_label: Path):
    img = Image.open(img_path)
    w, h = img.size

    crop_l_ratio = random.uniform(CROP_MIN, CROP_MAX)
    crop_r_ratio = random.uniform(CROP_MIN, CROP_MAX)
    crop_left = int(w * crop_l_ratio)
    crop_right = int(w * (1 - crop_r_ratio))
    cropped = img.crop((crop_left, 0, crop_right, h))
    cropped.save(out_img)

    if label_path and label_path.exists():
        with open(label_path) as f:
            lines = f.readlines()

        new_w_ratio = 1 - crop_l_ratio - crop_r_ratio
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id, xc, yc, bw, bh = parts
            xc_f = float(xc)
            if xc_f < crop_l_ratio or xc_f > (1 - crop_r_ratio):
                continue
            xc_f = (xc_f - crop_l_ratio) / new_w_ratio
            bw_f = float(bw) / new_w_ratio
            xc_f = max(0.0, min(1.0, xc_f))
            bw_f = max(0.0, min(1.0, bw_f))
            new_lines.append(f"{cls_id} {xc_f:.10f} {yc} {bw_f:.10f} {bh}\n")

        with open(out_label, "w") as f:
            f.writelines(new_lines)


def copy_normal(img_path: Path, label_path: Path | None, out_img: Path, out_label: Path):
    shutil.copy2(img_path, out_img)
    if label_path and label_path.exists():
        shutil.copy2(label_path, out_label)


def process_dataset(
    dataset_name: str,
    images_dir: Path,
    labels_dir: Path,
    out_train_img: Path,
    out_train_label: Path,
    out_val_img: Path,
    out_val_label: Path,
):
    image_files = sorted(images_dir.glob("*"))
    image_stems = {f.stem for f in image_files}

    label_stems = {f.stem for f in labels_dir.glob("*.txt")}

    valid_stems = sorted(image_stems & label_stems)
    if not valid_stems:
        print(f"  -> 0 images with labels, skipping")
        return

    random.shuffle(valid_stems)
    split_idx = max(1, int(len(valid_stems) * (1 - VAL_RATIO)))
    train_stems = valid_stems[:split_idx]
    val_stems = valid_stems[split_idx:]

    print(f"  {dataset_name}: {len(train_stems)} train, {len(val_stems)} val (from {len(valid_stems)} annotated)")

    crop_fn = crop_cms232 if "232" in dataset_name else copy_normal

    for stem in train_stems:
        img_src = images_dir / f"{stem}.jpg"
        label_src = labels_dir / f"{stem}.txt"
        out_img = out_train_img / f"{dataset_name}_{stem}.jpg"
        out_label = out_train_label / f"{dataset_name}_{stem}.txt"
        crop_fn(img_src, label_src, out_img, out_label)

    for stem in val_stems:
        img_src = images_dir / f"{stem}.jpg"
        label_src = labels_dir / f"{stem}.txt"
        out_img = out_val_img / f"{dataset_name}_{stem}.jpg"
        out_label = out_val_label / f"{dataset_name}_{stem}.txt"
        crop_fn(img_src, label_src, out_img, out_label)


def main():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    dirs = {
        "train_img": OUTPUT_DIR / "images" / "train",
        "val_img": OUTPUT_DIR / "images" / "val",
        "train_label": OUTPUT_DIR / "labels" / "train",
        "val_label": OUTPUT_DIR / "labels" / "val",
    }
    for d in dirs.values():
        d.mkdir(parents=True)

    datasets = [
        ("CMS25img", DATASETS_DIR / "CMS25img"),
        ("CMS232img", DATASETS_DIR / "CMS232img"),
    ]

    for name, ds_dir in datasets:
        print(f"Processing {name}...")
        process_dataset(
            dataset_name=name,
            images_dir=ds_dir / "images",
            labels_dir=ds_dir / "labels",
            out_train_img=dirs["train_img"],
            out_train_label=dirs["train_label"],
            out_val_img=dirs["val_img"],
            out_val_label=dirs["val_label"],
        )

    yaml_path = OUTPUT_DIR / "data.yaml"
    nc = len(CLASSES)
    yaml_content = (
        f"# {OUTPUT_DIR.name}\n"
        f"nc: {nc}\n"
        f"names:\n"
    )
    for i, name in enumerate(CLASSES):
        yaml_content += f"  {i}: {name}\n"
    yaml_content += (
        f"\n"
        f"train: {OUTPUT_DIR}/images/train\n"
        f"val: {OUTPUT_DIR}/images/val\n"
    )
    yaml_path.write_text(yaml_content)

    n_train = len(list(dirs["train_img"].iterdir()))
    n_val = len(list(dirs["val_img"].iterdir()))
    print(f"\nDone! {n_train} train, {n_val} val images -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
