# DeepDriveMD
DeepDriveMD implemented with colmena

## Development
Locally:
```
python -m venv env
source env/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements/dev.txt
pip install -r requirements/requirements.txt
pip install -e .
```
To run dev tools (isort, flake8, black, mypy): `make`
