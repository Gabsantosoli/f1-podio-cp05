"""Aplicação Streamlit: probabilidade de pódio na Fórmula 1 (CheckPoint05 - DSSC/FIAP).

Usa o mesmo pipeline salvo pelo notebook (``model/pipeline_final.joblib``), que
contém a mesma preparação de dados de ``f1_pipeline.py``.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

import f1_pipeline as f1  # noqa: F401  (necessário para desserializar o pipeline)

BASE_DIR = Path(__file__).parent
MANUAL = "Entrada manual"

st.set_page_config(page_title="Pódio na F1", layout="wide")


@st.cache_resource
def carregar_modelo():
    modelo = joblib.load(BASE_DIR / "model" / "pipeline_final.joblib")
    metadados = json.loads((BASE_DIR / "model" / "metadata.json").read_text(encoding="utf-8"))
    return modelo, metadados


@st.cache_data
def carregar_casos():
    return pd.read_csv(BASE_DIR / "data" / "casos_consistencia.csv")


modelo, meta = carregar_modelo()
casos = carregar_casos()
categorias = meta["categorias"]

PADRAO = {
    "grid": 5, "posicao_classificacao": 5.0, "largada_pit_lane": 0, "ano": 2025, "etapa": 10,
    "pontos_piloto_antes": 80.0, "posicao_piloto_antes": 5.0, "pontos_equipe_antes": 150.0,
    "posicao_equipe_antes": 3.0, "podios_piloto_ult5": 1, "podios_equipe_ult5": 2,
    "largadas_carreira": 100, "equipe": "mercedes", "motor": "mercedes",
    "circuito": "interlagos", "tipo_circuito": "RACE",
}

# ------------------------------------------------------------------ barra lateral
st.sidebar.header("Preenchimento")
escolha = st.sidebar.selectbox(
    "Carregar um caso do teste de consistência",
    [MANUAL] + casos["caso"].tolist(),
    key="caso",
    help="Os casos vêm do conjunto de teste do notebook e servem para o teste de paridade.",
)
caso = None if escolha == MANUAL else casos.loc[casos["caso"] == escolha].iloc[0]
valores = PADRAO if caso is None else caso.to_dict()
sufixo = escolha  # chaves dos widgets mudam com o caso, para recarregar os valores


def pct(x):
    """Percentual com duas casas e vírgula decimal (ex.: 38,84%)."""
    return f"{x * 100:.2f}%".replace(".", ",")


def num(x, casas=3):
    return f"{x:.{casas}f}".replace(".", ",")


def ausente(v):
    return v is None or (isinstance(v, float) and np.isnan(v))


def opcao(lista, valor):
    return lista.index(valor) if valor in lista else 0


st.sidebar.divider()
st.sidebar.subheader("Sobre o modelo")
st.sidebar.markdown(
    f"**Algoritmo:** {meta['modelo']}, com hiperparâmetros ajustados pelo {meta['configuracao']}.\n\n"
    f"**Nota do modelo:** {num(meta['metricas_teste']['roc_auc'], 2)} de 1,00 (ROC-AUC). "
    f"Comparando um piloto que subiu ao pódio com um que não subiu, o modelo dá a maior chance "
    f"para o que subiu em cerca de {meta['metricas_teste']['roc_auc']:.0%} das vezes.\n\n"
    f"Essa nota foi medida em corridas que o modelo não viu no treino, e ficou igual à do "
    f"desenvolvimento ({num(meta['auc_cv_media'], 2)}).\n\n"
    f"**Quando prevê pódio:** a partir de {pct(meta['limiar'])} de chance.\n\n"
    f"**Dados:** corridas de 2014 a 2026 da base F1DB."
)

# ------------------------------------------------------------------ entradas
st.title("Qual a chance de pódio?")
st.write(
    "Informe a situação do piloto **antes da largada**. O modelo estima a probabilidade de ele "
    "terminar a corrida entre os três primeiros."
)
if caso is not None:
    st.info(
        f"Caso carregado: **{caso['driverId']}** em **{caso['circuito']}** ({caso['date'][:10]}). "
        f"Resultado real: **{caso['positionText']}**."
    )

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("Corrida e equipe")
    circuito = st.selectbox("Circuito", categorias["circuito"],
                            index=opcao(categorias["circuito"], valores["circuito"]), key=f"circuito_{sufixo}")
    tipo_padrao = valores["tipo_circuito"] if caso is not None else meta["tipo_por_circuito"].get(circuito, "RACE")
    tipo_circuito = st.selectbox("Tipo de circuito", categorias["tipo_circuito"],
                                 index=opcao(categorias["tipo_circuito"], tipo_padrao),
                                 key=f"tipo_{sufixo}_{circuito}")
    ano = st.number_input("Temporada", 2014, 2026, int(valores["ano"]), key=f"ano_{sufixo}")
    etapa = st.number_input("Etapa", 1, 24, int(valores["etapa"]), key=f"etapa_{sufixo}")
    equipe = st.selectbox("Equipe", categorias["equipe"],
                          index=opcao(categorias["equipe"], valores["equipe"]), key=f"equipe_{sufixo}")
    motor = st.selectbox("Motor", categorias["motor"],
                         index=opcao(categorias["motor"], valores["motor"]), key=f"motor_{sufixo}")

with col2:
    st.subheader("Largada")
    pit_lane = st.checkbox("Largou do pit lane", bool(valores["largada_pit_lane"]), key=f"pit_{sufixo}")
    grid = st.number_input("Posição de largada (grid)", 1, 27, int(valores["grid"]), key=f"grid_{sufixo}",
                           help="Para largada do pit lane, use o último lugar do grid + 1.")
    sem_classif = st.checkbox("Sem tempo na classificação", ausente(valores["posicao_classificacao"]),
                              key=f"semclass_{sufixo}")
    posicao_classificacao = None if sem_classif else st.number_input(
        "Posição na classificação", 1, 26,
        5 if ausente(valores["posicao_classificacao"]) else int(valores["posicao_classificacao"]),
        key=f"classif_{sufixo}")
    largadas_carreira = st.number_input("Largadas na carreira", 0, 450, int(valores["largadas_carreira"]),
                                        key=f"largadas_{sufixo}")

with col3:
    st.subheader("Campeonato e forma")
    pontos_piloto = st.number_input("Pontos do piloto antes da corrida", 0.0, 700.0,
                                    float(valores["pontos_piloto_antes"]), step=1.0, key=f"ptsp_{sufixo}")
    sem_pos_piloto = st.checkbox("Piloto sem posição no campeonato (1ª etapa/estreia)",
                                 ausente(valores["posicao_piloto_antes"]), key=f"sempp_{sufixo}")
    posicao_piloto = None if sem_pos_piloto else st.number_input(
        "Posição do piloto no campeonato", 1, 30,
        1 if ausente(valores["posicao_piloto_antes"]) else int(valores["posicao_piloto_antes"]),
        key=f"posp_{sufixo}")
    pontos_equipe = st.number_input("Pontos da equipe antes da corrida", 0.0, 1000.0,
                                    float(valores["pontos_equipe_antes"]), step=1.0, key=f"ptse_{sufixo}")
    sem_pos_equipe = st.checkbox("Equipe sem posição no campeonato (1ª etapa)",
                                 ausente(valores["posicao_equipe_antes"]), key=f"sempe_{sufixo}")
    posicao_equipe = None if sem_pos_equipe else st.number_input(
        "Posição da equipe no campeonato", 1, 12,
        1 if ausente(valores["posicao_equipe_antes"]) else int(valores["posicao_equipe_antes"]),
        key=f"pose_{sufixo}")
    podios_piloto = st.number_input("Pódios do piloto nas últimas 5 corridas", 0, 5,
                                    int(valores["podios_piloto_ult5"]), key=f"podp_{sufixo}")
    podios_equipe = st.number_input("Pódios da equipe nas últimas 5 corridas", 0, 10,
                                    int(valores["podios_equipe_ult5"]), key=f"pode_{sufixo}")

# ------------------------------------------------------------------ previsão
entrada = pd.DataFrame([{
    "grid": grid,
    "posicao_classificacao": np.nan if posicao_classificacao is None else posicao_classificacao,
    "largada_pit_lane": int(pit_lane),
    "ano": ano,
    "etapa": etapa,
    "pontos_piloto_antes": pontos_piloto,
    "posicao_piloto_antes": np.nan if posicao_piloto is None else posicao_piloto,
    "pontos_equipe_antes": pontos_equipe,
    "posicao_equipe_antes": np.nan if posicao_equipe is None else posicao_equipe,
    "podios_piloto_ult5": podios_piloto,
    "podios_equipe_ult5": podios_equipe,
    "largadas_carreira": largadas_carreira,
    "equipe": equipe,
    "motor": motor,
    "circuito": circuito,
    "tipo_circuito": tipo_circuito,
}])[meta["features"]].astype({c: float for c in f1.NUM_FEATURES})

probabilidade = float(modelo.predict_proba(entrada)[0, 1])
st.session_state["probabilidade"] = probabilidade
previsao_podio = probabilidade >= meta["limiar"]

# Faixas de leitura da probabilidade:
# verde    -> acima do limiar de decisão (o modelo prevê pódio);
# amarelo  -> abaixo do limiar, mas acima da taxa média de pódios (chance real, porém incerta);
# vermelho -> abaixo da taxa média de pódios (chance menor que a de um piloto qualquer).
FAIXAS = [
    (meta["limiar"], "Alta chance de pódio", "Previsão: pódio", "#1e8449", "#e9f7ef"),
    (meta["taxa_base"], "Chance moderada", "Previsão: fora do pódio, mas acima da média", "#9a7d0a", "#fef9e7"),
    (0.0, "Baixa chance de pódio", "Previsão: fora do pódio", "#c0392b", "#fdedec"),
]
_, rotulo, previsao_texto, cor, fundo = next(f for f in FAIXAS if probabilidade >= f[0])


def cartao(conteudo, cor, fundo):
    """Bloco colorido usado para o resultado e para a comparação com o resultado real."""
    st.markdown(
        f'<div style="border-left:6px solid {cor};background:{fundo};color:#1f2933;'
        f'padding:16px 20px;border-radius:6px;margin-bottom:12px">{conteudo}</div>',
        unsafe_allow_html=True,
    )


st.divider()
res1, res2 = st.columns([1, 1])
with res1:
    cartao(
        f'<div style="font-size:0.9rem;color:#52606d">Probabilidade de pódio</div>'
        f'<div style="font-size:2.6rem;font-weight:700;color:{cor};line-height:1.2">{pct(probabilidade)}</div>'
        f'<div style="font-size:1.15rem;font-weight:600;color:{cor}">{rotulo}</div>'
        f'<div style="font-size:0.95rem;margin-bottom:10px">{previsao_texto}</div>'
        f'<div style="height:10px;background:#d9dee3;border-radius:5px">'
        f'<div style="width:{probabilidade * 100:.1f}%;height:10px;background:{cor};border-radius:5px"></div></div>',
        cor, fundo,
    )
    st.caption(
        f"Verde: a partir de {pct(meta['limiar'])} (limiar escolhido na validação cruzada). "
        f"Amarelo: entre {pct(meta['taxa_base'])} (taxa média de pódios) e o limiar. "
        f"Vermelho: abaixo da taxa média."
    )
with res2:
    if caso is not None:
        acertou = int(previsao_podio) == int(caso["podio"])
        resultado_real = "pódio" if caso["podio"] == 1 else "fora do pódio"
        cor_real, fundo_real = ("#1e8449", "#e9f7ef") if acertou else ("#c0392b", "#fdedec")
        cartao(
            f'<div style="font-weight:600;color:{cor_real}">'
            f'{"O modelo acertou" if acertou else "O modelo errou"} este caso</div>'
            f'<div>Resultado real: <b>{resultado_real}</b> (posição: {caso["positionText"]})</div>',
            cor_real, fundo_real,
        )
        diferenca = abs(probabilidade - caso["probabilidade"])
        paridade_ok = diferenca < 1e-6
        cartao(
            f'<div style="font-weight:600;color:{"#1e8449" if paridade_ok else "#c0392b"}">'
            f'Teste de paridade {"aprovado" if paridade_ok else "com divergência"}</div>'
            f'<div>Notebook: {num(caso["probabilidade"], 6)} | Aplicação: {num(probabilidade, 6)} | '
            f'Diferença: {diferenca:.1e}</div>',
            "#1e8449" if paridade_ok else "#c0392b", "#e9f7ef" if paridade_ok else "#fdedec",
        )
    else:
        st.write("Carregue um caso do teste de consistência na barra lateral para comparar a previsão "
                 "com o resultado real e com o notebook.")

with st.expander("Dados enviados ao pipeline"):
    st.dataframe(entrada, hide_index=True)
