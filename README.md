# Webloom

Simple web application implementing the Loom interface using the tiny `sshleifer/tiny-gpt2` model by default.

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Running

Start the development server:

```bash
python run.py
```

Set the `LOOM_DEVICE` environment variable to `cuda` or `mps` to use a GPU. By
default the model runs on CPU for maximum compatibility.

The application will start on `http://localhost:8001`.

Open the URL above in a web browser and use the interface to generate text.

Use the length slider to control how many tokens are added to the prompt for each continuation.
Use the temperature slider to adjust the randomness of the next continuation.
Use the variants slider to choose how many alternate continuations to generate.
Use the model dropdown to switch between GPT-2 variants.
Hover over a generated continuation to see which model produced it.
Changing the model may take a moment the first time; the "Weave" button will
show "Loading..." until the model is ready.

## Docker

To build the Docker image run:

```bash
docker build -t webloom .
```

Start a container exposing port 8001:

```bash
docker run -p 8001:8001 webloom
```

Then open `http://localhost:8001` in your browser.
