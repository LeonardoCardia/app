# Detector de acne

Aplicação Streamlit que usa um modelo YOLO para localizar possíveis lesões de
acne em uma imagem enviada pelo usuário.

> Projeto experimental. O resultado não substitui a avaliação de um
> dermatologista.

## Estrutura

```text
.
├── models/
│   └── best.pt
├── inference.py
├── streamlit_app.py
└── requirements.txt
```

## Execução local

Com Python 3.10 ou superior:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

No Windows, ative o ambiente com:

```powershell
.venv\Scripts\Activate.ps1
```

## Execução na EC2

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
streamlit run streamlit_app.py \
  --server.address 0.0.0.0 \
  --server.port 8501 \
  --server.headless true
```

O modelo é carregado de `models/best.pt`, relativo ao próprio aplicativo.
