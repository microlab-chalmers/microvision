<div align="center">
  <a href="https://www.snd.se" target="_blank"> <img width="50%" src="assets/logo_black.png" alt="MicroVision"></a>
</div>

<hr>
<div align="center">
    <a href="https://arxiv.org/abs/2603.18192"><img src="https://img.shields.io/badge/arXiv-2603.18192-b31b1b.svg?logo=arxiv" alt="arXiv:MY-INDEX"></a>
    <a href="https://snd.se/en/catalogue/dataset/2025-74"><img src="https://img.shields.io/badge/SND-2025--74-blue.svg?logo=data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTgiPz4KPCEtLSBHZW5lcmF0b3I6IEFkb2JlIElsbHVzdHJhdG9yIDIyLjAuMCwgU1ZHIEV4cG9ydCBQbHVnLUluIC4gU1ZHIFZlcnNpb246IDYuMDAgQnVpbGQgMCkgIC0tPgo8c3ZnIHZlcnNpb249IjEuMSIgaWQ9IkxheWVyXzEiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgeG1sbnM6eGxpbms9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkveGxpbmsiIHg9IjBweCIgeT0iMHB4IgoJIHZpZXdCb3g9IjAgMCAxMDAuOSA4Ni4zIiBzdHlsZT0iZW5hYmxlLWJhY2tncm91bmQ6bmV3IDAgMCAxMDAuOSA4Ni4zOyIgeG1sOnNwYWNlPSJwcmVzZXJ2ZSI+CjxzdHlsZSB0eXBlPSJ0ZXh0L2NzcyI+Cgkuc3Qye2ZpbGw6IzY0OUREMjt9Cgkuc3Qze2ZpbGw6I0U0NDYyQzt9Cgkuc3Q0e2ZpbGw6IzFGMzk2Mzt9Cjwvc3R5bGU+CjxnPgoJPHBhdGggY2xhc3M9InN0MiIgZD0iTTM5LjYsMTIuMmw0LjIsOC45YzYuOS0zLjMsMTUuMi0zLjMsMjIuNCwwLjdjMTEuOSw2LjUsMTYuMywyMS41LDkuOCwzMy41QzY5LjUsNjcuMiw1NC41LDcxLjYsNDIuNSw2NQoJCUMzNS41LDYxLjEsMzEsNTQuNCwzMCw0N2wtOS43LDEuNGMxLjUsMTAuMyw3LjcsMTkuOCwxNy42LDI1LjJjMTYuNyw5LjEsMzcuNiwzLDQ2LjgtMTMuN2M5LjEtMTYuNywzLTM3LjctMTMuNy00Ni44CgkJQzYwLjksNy42LDQ5LjMsNy43LDM5LjYsMTIuMnoiLz4KCTxwYXRoIGNsYXNzPSJzdDMiIGQ9Ik00NC4yLDM4LjRsLTI5LjYsNC4zYy0wLjItMS40LTAuMy0yLjgtMC4zLTQuM2MwLTExLjksNi45LTIyLjEsMTctMjdMNDQuMiwzOC40eiIvPgoJPHBhdGggY2xhc3M9InN0NCIgZD0iTTU0LjUsNDMuNWwtNS43LTEyLjFjMy43LTEuNyw4LjEtMS43LDExLjksMC40YzYuNCwzLjUsOC43LDExLjUsNS4yLDE3LjlTNTQuNCw1OC40LDQ4LDU0LjkKCQljLTMuNy0yLTYtNS45LTYuNi05LjgiLz4KPC9nPgo8L3N2Zz4K" alt="SND Dataset"></a>
    <!-- <a href="https://colab.research.google.com/test.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> -->
</div>

# MicroVision: An Open Dataset and Benchmark Models for Detecting Vulnerable Road Users and Micromobility Vehicles

Welcome to the repository for code related to the MicroVision dataset. More info about the dataset can be obtained from our [preprint](https://arxiv.org/abs/2603.18192).

<!-- <center>
<iframe width="560" height="315" src="https://www.youtube.com/embed/sdZJUZWMX-Q?si=hoxdzt9gLekago-N" title="YouTube video player" frameborder="0" allowfullscreen></iframe>
</center> -->

## Dataset

The MicroVision dataset is provided in three files, which can be downloaded at SND: [https://researchdata.se/en/catalogue/dataset/2025-74/1](https://researchdata.se/en/catalogue/dataset/2025-74/1).

- `images.zip`: Contains all images
- `labels.zip`: Contains all labels in YOLO format (object box coordinates and classes)
- `meta.csv`: Contains meta data e.g. needed to reproduce the splits used for our benchmark model

Additionally, we provide weights for some common state-of-the-art object-detection architectures:

- `microvision_yolo11.pt`: Weights file for the YOLO11-X model
- `microvision_fasterrcnn.pth`: Weights file for the Detectron2 Faster R-CNN models
- `microvision_rfdetr.pth`: Weights file for the RF-DETR large model (resolution 1232 px)

## Toolkit

### Installation

Clone the toolkit from GitHub
```
git clone https://github.com/microlab-chalmers/microvision
cd microvision
```

We recommend using [uv](https://docs.astral.sh/uv/) to install the microvision package and its dependencies in a virtual environment
```
uv sync
```

This should install the package and its dependencies. You can verify the installation by trying to import the `microvision` package without error:

```
uv run python -c "import microvision"
```

### Scripts
We provide some scripts to work with the data (stored in the `scripts` folder).

#### Prepare train/val/test split
The dataset is not split when downloading. The files can be split using the `split_data.py` tool, as follows:

```
python scripts\split_data.py \
  --path_data "test" \ # path to data folder (includes the "images" and "labels" folders)
  --path_meta "test/meta.csv" \ # path to meta.csv file with the split info
  --path_out "test/splits" \ # path to output folder
  --split_config "split" \ # split configuration, e.g. train/val/test (optional)
  --override # delete any existing files at path_out (optional)
```

#### Convert annotations from YOLO to COCO JSON 

This script can be used to convert annotations from the available YOLO11 format to COCO JSON, for example, to train Detectron2 or Roboflow models.

```
python scripts\yolo2coco.py \
  --path_yolo "test" \ # path to data folder (contains the "images" and "labels" folders)
  --path_json "test\test.json" path to output JSON file
```

#### Annotator agreement
The annotator-agreement results can be reproduced by running the following script:

```
uv run scripts\agreement.py
```

The script analyzes the annotations by three annotators, provided under `data/agreement`.

### Model training

#### YOLO11
The YOLO11 model was trained with ultralytics:

```
yolo detect train \
    data=microvision.yaml \
    model=yolo11x.pt \
    exist_ok=True \
    save=True \
    name="microvision_x_1280" \
    epochs=100 \
    imgsz=1280 \
    batch=32 \
    plots=True \
    device=0,1,2,3 # if running parallel
```

With `microvision.yaml`:

```
# Number of classes
nc: 5

# Class names
names: ["pedestrian", "bicycle", "cyclist", "e-scooter", "e-scooterist"] 

# Path to splits
train: .../train
val: .../val
test: .../test
```

#### Faster R-CNN

To be updated
<!-- The Faster R-CNN model was trained with Detectron2. -->

#### RF-DETR

To be updated

## Contact
Alexander Rasch
<div>
  <a href="mailto:alexander.rasch@chalmers.se"><img src="assets/contact/email-open.svg" height="24pt" alt="Alexander Rasch Email id"></a>
  <img src="assets/contact/logo-transparent.svg" width="1%" alt="space">
  <a href="https://orcid.org/0000-0001-6868-8364"><img src="assets/contact/orcid.svg" height="24pt" alt="Alexander Rasch ORCID"></a>
  <img src="assets/contact/logo-transparent.svg" width="1%" alt="space">
  <a href="https://github.com/feuerblitz7/"><img src="assets/contact/github-repo.svg" height="24pt" alt="Rahul Pai GitHub"></a>
  <img src="assets/contact/logo-transparent.svg" width="1%" alt="space">
  <a href="https://se.linkedin.com/in/alex-rasch"><img src="assets/contact/linkedin.svg" height="24pt" alt="Rahul Pai Linkedin"></a>

</div>

Rahul Rajendra Pai
<div>
  <a href="mailto:rahul.pai@chalmers.se"><img src="assets/contact/email-open.svg" height="24pt" alt="Rahul Pai Email id"></a>
  <img src="assets/contact/logo-transparent.svg" width="1%" alt="space">
  <a href="https://orcid.org/0000-0002-1516-6930"><img src="assets/contact/orcid.svg" height="24pt" alt="Rahul Pai ORCID"></a>
  <img src="assets/contact/logo-transparent.svg" width="1%" alt="space">
  <a href="https://github.com/Rahul-Pi/"><img src="assets/contact/github-repo.svg" height="24pt" alt="Rahul Pai GitHub"></a>
  <img src="assets/contact/logo-transparent.svg" width="1%" alt="space">
  <a href="https://se.linkedin.com/in/rahul-pai"><img src="assets/contact/linkedin.svg" height="24pt" alt="Rahul Pai Linkedin"></a>
  <img src="assets/contact/logo-transparent.svg" width="1%" alt="space">
</div>

<br>

## License
The dataset and models from this project are licensed under the [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) license.

## Citation
If you use the dataset or the models in your research, please cite the following paper:

```bibtex
@misc{microvision2026,
      title={MicroVision: An Open Dataset and Benchmark Models for Detecting Vulnerable Road Users and Micromobility Vehicles}, 
      author={Alexander Rasch and Rahul Rajendra Pai},
      year={2026},
      eprint={2603.18192},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.18192}, 
}
```

## Acknowledgments
We would like to thank Shiyi Qiu, Mahin Garg, and Anton Broman, for helping with processing and annotating the data, and [Marco Dozza](https://www.chalmers.se/en/persons/dozza/) for valuable discussions and funding acquisition.

The computations were enabled by resources provided by the [National Academic Infrastructure for Supercomputing in Sweden (NAISS)](https://www.naiss.se/), partially funded by the Swedish Research Council through grant agreement no. 2022-06725.

This work was carried out in the project MicroVision, funded by Vinnova (Sweden's innovation agency), the Swedish Energy Agency, and Formas (a Swedish research council for sustainable development), through the DriveSweden program (reference number [2023-01047](https://www.drivesweden.net/en/project/microvision-development-testing-and-demonstration-real-time-support-system-electric-vehicle)).