# Age Detector

A small local web application that estimates a person's **apparent age** from a
face image. Face detection uses InsightFace's RetinaFace detector, while age
estimation uses the dedicated MiVOLO v2 continuous-regression model.

The app accepts an uploaded image, an image copied to the system clipboard, or
a snapshot from the browser's camera. It finds faces locally, estimates age for
each detected face, and returns the results in the browser. Images are
processed in memory and are not stored.

> **Project note:** This was a fun project made for **$0.10 USD** using
> **MiniMax M3.1 Flash**.

> Age estimation is approximate and measures visible characteristics. It cannot
> determine a person's actual age and should not be used for identity, access
> control, employment, healthcare, or other consequential decisions.

## Requirements

- Python 3.9 or newer
- A compiler may be needed when installing InsightFace's dependencies

## Setup

```sh
cd ~/Code/OpenCodeProjects/age-detector
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The first run downloads the InsightFace detector and the MiVOLO age model into
the local cache. The age model is about 112 MB and is checksum-verified before
use.

## Run

```sh
source .venv/bin/activate
flask --app app run --debug --port 5001
```

Open <http://127.0.0.1:5001> and upload an image or use the camera.

> macOS AirTunes/AirPlay Receiver may occupy port `5000` on IPv6 and return a
> `403 Forbidden` response for `http://localhost:5000`. Using port `5001`
> avoids that conflict.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `AGE_MODEL_NAME` | `buffalo_l` | InsightFace detector model pack |
| `AGE_MODEL_ROOT` | `~/.insightface` | InsightFace model cache |
| `AGE_MODEL_PATH` | `~/.cache/age-detector/mivolo_v2_age.onnx` | MiVOLO model path |

## Privacy

- The server binds to localhost by default.
- Uploaded images are decoded into memory and are never written to disk.
- The page does not transmit images to third-party services.
- The browser asks for camera permission only when the user starts the camera.

## Model notes

InsightFace's bundled `genderage` model was previously used for age
estimation. Although convenient, it performs poorly for exact age regression
and produced implausible results for children and young adults. MiVOLO v2 was
trained specifically for age regression and reports approximately 3.7 years
mean absolute error on its benchmark. This figure is an aggregate benchmark
measurement, not a guaranteed error for an individual image.

## Credits and licenses

This project uses:

- [InsightFace](https://github.com/deepinsight/insightface) for face detection
- [MiVOLO](https://github.com/WildChlamydia/MiVOLO) for age estimation
- The ONNX export published as
  [`Sam-Apostel/mivolo-v2-age-onnx`](https://huggingface.co/Sam-Apostel/mivolo-v2-age-onnx)

Model weights are downloaded separately and are not included in this repository.
Review each model's license and terms before use in another project.
