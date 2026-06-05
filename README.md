# CMScraper : computer vision based scraper for polls in FB groups

## What is it?
CMScraper is designed as a mean to automatically gather data from polls on FB groups. It is being developed to tackle the tedious task of collecting data by hand to analyze polls results as a non admin.

## What is it not?
- A general purpose scraper for FB or anything else
- A spam bot that can post in groups or perform any other action besides viewing specific aspects

## How does it work?
CMScraper works by viewing pages like a human: although FB has put considerable means into obfuscating the code of its pages to prevent scraping with a simple regex, its human interface is remarkably clear and readable. As such, it can be read by a robot using basic computer vision techniques and minimal training. The robot can then extract the semantic parts of the page, and navigate it simulating a classic mouse pointer.

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) installed and running locally
- A YOLO detection model (see below)

## Setup

```bash
# Install Python dependencies
pip install -r requirements.txt

# Pull the vision model used for OCR (via Ollama)
ollama pull qwen3.5:9b-q4_K_M
```

Create a `.env` file from the template:

```bash
cp .env.template .env
```

### Configuration

| Variable | Description | Default |
|---|---|---|
| `MODEL` | Path to the YOLO detection model (`.pt`) | — |
| `INDICE_CONF` | Confidence threshold for general detections | `0.7` |
| `INDICE_CONF_AUTEUR` | Lower confidence threshold for author detection (model is weaker on these) | `0.5` |
| `VERBOSE` | Verbosity level 0–3 | `0` |
| `DEBUG` | Enable debug mode (saves screenshots, overlays with detected components) | `false` |
| `DEBUG_FOLDER` | Directory for debug output | `debug` |

### Model download

Download a suitable YOLO model from [this link](https://cloud.simonhergott.fr/s/sAR7WHyqFcFfkof). There are 2 models, based on YOLOv11 Nano and Medium.

## Launch

```bash
python CMScraper.py
```

The script reads the screen via `mss`, runs YOLO detection to identify poll components, then uses a vision language model (Qwen via Ollama) for OCR instead of Tesseract. It scrolls through the Facebook group page, extracts poll questions, options, and respondents, and saves results to `debug/sondages.json` and `debug/people_db.json`.

## Design
CMScraper relies on YOLO for semantic segmentation of the interface it sees, and simple Python to handle the logic. The polls are stored in a JSON file, and specific names of people are put in a database to prevent multiplying persons in case of an OCR bug.
YOLOv11 has been trained on ~100 manually annotated frames: the training datasets are not shared for obvious privacy reasons, but the weights are given. After reading the frames and detecting the interesting parts, some tests are performed to verify the integrity of the detected poll to prevent a degradation of resulting data integrity.

### Class diagram

![Class diagram](assets_doc/contraintes_composition.png)