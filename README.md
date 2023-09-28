# python-embedding-api

This repo contains a simple HTTP API that returns embeddings for a given sentence.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.


## Table of Contents
1. [Pre-requisites](#pre-requisites)
2. [Usage](#usage)
3. [API Documentation](#api-documentation)

## Pre-requisites

Install Pyenv/python

```bash
# Pyenv installation for mac
brew update && brew install pyenv

# Python installation through pyenv
pyenv install 3.10
```

Install poetry

```bash
curl -sSL https://install.python-poetry.org | python3 -
```


## Usage
To install project dependencies
```bash
poetry install
```

To start the API application
```bash
poetry run poe api
```


## Run tests

```bash
poetry run poe pytest
```

## API Documentation

### Local
With the service running, visit `localhost:5000/docs` to view the API docs.
