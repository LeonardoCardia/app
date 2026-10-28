"""Interface Streamlit para o detector de acne."""

from __future__ import annotations

from io import BytesIO

import streamlit as st
from PIL import Image, UnidentifiedImageError

from inference import AcneDetector


st.set_page_config(
    page_title="Detector de acne",
    page_icon="🔎",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def load_detector() -> AcneDetector:
    """Mantém o modelo em memória entre as execuções da página."""

    return AcneDetector()


st.title("Detector de acne")
st.write(
    "Envie uma foto e o modelo marcará as regiões em que identificou possíveis "
    "lesões de acne."
)

with st.sidebar:
    st.header("Configurações")
    confidence = st.slider(
        "Confiança mínima",
        min_value=0.10,
        max_value=0.90,
        value=0.15,
        step=0.05,
        help=(
            "Valores maiores reduzem falsos positivos, mas podem deixar de "
            "mostrar algumas lesões. O valor inicial maximiza o F1 na validação."
        ),
    )
    st.info(
        "Este projeto é experimental e não substitui a avaliação de um "
        "dermatologista."
    )

uploaded_file = st.file_uploader(
    "Selecione uma imagem",
    type=("jpg", "jpeg", "png", "webp"),
    help="Formatos aceitos: JPG, PNG e WEBP.",
)

if uploaded_file is None:
    st.caption("A imagem é processada somente durante esta sessão.")
    st.stop()

try:
    image = Image.open(BytesIO(uploaded_file.getvalue()))
    image.load()
except (UnidentifiedImageError, OSError):
    st.error("Não foi possível abrir o arquivo. Envie uma imagem válida.")
    st.stop()

try:
    with st.spinner("Analisando a imagem..."):
        result = load_detector().predict(image, confidence=confidence)
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()
except Exception as exc:  # A interface deve mostrar uma falha legível ao usuário.
    st.error(f"Não foi possível realizar a análise: {exc}")
    st.stop()

if result.has_acne:
    st.error("Possíveis sinais de acne foram detectados.", icon="🔎")
else:
    st.success("Nenhum sinal de acne foi detectado com o limiar selecionado.")

metric_col1, metric_col2 = st.columns(2)
metric_col1.metric("Regiões detectadas", result.count)
metric_col2.metric(
    "Maior confiança",
    f"{result.max_confidence:.1%}" if result.max_confidence is not None else "—",
)

original_col, result_col = st.columns(2)
with original_col:
    st.subheader("Imagem original")
    st.image(image, width="stretch")
with result_col:
    st.subheader("Resultado do modelo")
    st.image(result.annotated_image, width="stretch")

st.caption(
    "A ausência de detecções não descarta acne. Iluminação, enquadramento, tom "
    "de pele e qualidade da foto podem afetar o resultado."
)

