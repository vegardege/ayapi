# AyAPI

AyAPI is a thin [FastAPI](https://fastapi.tiangolo.com/) wrapper for
python-only AI features. Its purpose is to make these functions
available to an SPA.

Source code is shared in the spirit of sharing, but you most likely
want to create your own wrapper instead of trying to reuse this one.
It's very specific for my needs at the time of writing.

## Installation

If you want to build this version of the API, fork the repository and
run:

```bash
docker build -t ayapi:latest .
```

You can then run the API with:

```bash
docker run -d \
    --name ayapi \
    -p 8000:8000 \
    -v $HOME/.cache:/root/.cache \
    ayapi:latest
```

During development you can also run it directly with:

```bash
poetry run fastapi dev src/ayapi/main.py
```

## Usage

Currently, the API only supports one endpoint:

```bash
curl -X POST http://localhost:8000/embeddings \
  -H "Content-Type: application/json" \
  -d '{
        "model": "Qwen/Qwen3-Embedding-0.6B",
        "sentences": [
          "apple",
          "pear",
          "orange",
          "yellow",
          "manufacture"
        ]
      }'
```

This will return a JSON document:

```json
{
  "embeddings": [
    [
      0.5817796587944031,
      1.8998889923095703,
      -3.6402902603149414,
      ...
    ],
    ...
  ]
}
```

The embeddings returned are in the same order as the sentences in the request.
