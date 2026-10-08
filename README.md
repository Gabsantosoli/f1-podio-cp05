# Pódio na Fórmula 1: Random Forest × XGBoost × LightGBM

**CheckPoint05 — Data Science & Statistical Computing (FIAP 2026) — Prof. Jones Egydio**

| Integrante | RM |
|---|---|
| Thiago Nascimento | 561967 |
| Vinicius Herreira | 565662 |
| Gustavo Rocha | 564152 |
| Gabriel Santos | 562419 |

- **Aplicação Streamlit:** PREENCHER_COM_URL_DA_APLICACAO
- **Repositório:** PREENCHER_COM_URL_DO_REPOSITORIO

## Problema

> Dado o que se sabe sobre um piloto **antes da largada** (posição no grid, classificação, situação no campeonato, forma recente, equipe, motor e circuito), qual a probabilidade de ele terminar a corrida no **pódio**?

- **Tipo:** classificação binária (`podio` = 1 se terminou em 1º, 2º ou 3º).
- **Unidade observacional:** um piloto em uma corrida.
- **Período:** era híbrida da F1 (2014 até a 15ª etapa de 2026), com 5.423 observações após o tratamento e 14,8% de pódios.
- **Métrica principal:** ROC-AUC, escolhida pelo desbalanceamento das classes e porque a pergunta é de ordenação. Métricas auxiliares: Average Precision, F1, precision, recall e acurácia.

## Dados

- **Fonte:** [F1DB](https://github.com/f1db/f1db), release **v2026.15.1**, com licença [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- O arquivo `data/f1db-csv.zip` está versionado no repositório. Se ele for removido, o notebook baixa automaticamente a mesma release a partir de
  `https://github.com/f1db/f1db/releases/download/v2026.15.1/f1db-csv.zip` (função `get_f1db_zip` em `f1_pipeline.py`).

## Resultados

| Configuração | AUC treino | AUC CV (média ± dp) | Gap |
|---|---|---|---|
| Referência ingênua (ordenar só pelo grid) | — | 0,923 | — |
| Random Forest — baseline | 1,000 | 0,937 ± 0,011 | 0,063 |
| XGBoost — baseline | 1,000 | 0,925 ± 0,013 | 0,075 |
| LightGBM — baseline | 1,000 | 0,932 ± 0,011 | 0,068 |
| Random Forest — Grid Search / Optuna | 0,957 / 0,952 | 0,943 / 0,943 | 0,014 / 0,009 |
| XGBoost — Grid Search | 0,954 | 0,942 ± 0,009 | 0,012 |
| LightGBM — Grid Search / Optuna | 0,963 / 0,954 | 0,939 / 0,943 | 0,024 / 0,011 |
| **XGBoost — Optuna (modelo final)** | **0,950** | **0,944 ± 0,009** | **0,006** |

**Teste final (usado uma única vez):** ROC-AUC **0,942** (IC 95% por bootstrap: 0,93–0,96), contra 0,918 da referência só com o grid. O ganho sobre a referência tem IC 95% de +0,01 a +0,04, sem incluir o zero. Average Precision 0,718; recall 0,74 e precision 0,62 com o limiar de 0,39 escolhido na validação cruzada.

Na aplicação, a probabilidade aparece em três faixas de cor. No teste, a taxa real de pódio em cada faixa foi:

| Faixa | Critério | Taxa real de pódio no teste |
|---|---|---|
| Verde | probabilidade ≥ limiar (38,8%) | ~62% |
| Amarelo | entre a taxa média (14,8%) e o limiar | ~27% |
| Vermelho | abaixo da taxa média | < 2% |

Principais conclusões:
- Os baselines sobreajustavam (AUC de treino ≈ 1,0). O tuning, tanto no Grid Search quanto no Optuna, levou a modelos mais simples e regularizados, que generalizam melhor.
- A posição de largada e a de classificação são as variáveis mais importantes. A posição no campeonato e os pódios recentes (do piloto e da equipe) ajustam essa chance.
- O desempenho no teste ficou dentro do esperado pela validação cruzada.
- O teste de paridade confirmou que a aplicação reproduz exatamente as previsões do notebook.

## Estrutura do repositório

```
├── Checkpoint05_F1_Podio.ipynb   # notebook completo (exercícios 1 a 7)
├── f1_pipeline.py                # preparação de dados compartilhada (notebook + app)
├── app.py                        # aplicação Streamlit
├── requirements.txt              # dependências com versões fixadas
├── model/
│   ├── pipeline_final.joblib     # pipeline final (preparação + XGBoost)
│   └── metadata.json             # hiperparâmetros, limiar, métricas e categorias
└── data/
    ├── f1db-csv.zip              # base F1DB v2026.15.1
    └── casos_consistencia.csv    # casos do teste de consistência/paridade
```

## Como executar

Requisito: Python 3.12 ou superior.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate  |  Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install jupyter
```

**Notebook:** abra `Checkpoint05_F1_Podio.ipynb` e execute *Run All* (~6 minutos). O notebook regenera `model/` e `data/casos_consistencia.csv`.

**Aplicação:**

```bash
streamlit run app.py
```

Na barra lateral, escolha um caso do teste de consistência para comparar a previsão com a do notebook (teste de paridade), ou preencha os dados manualmente.

**Deploy no Streamlit Community Cloud:** *New app* → selecione este repositório, branch `main` e arquivo `app.py`. Em *Advanced settings*, escolha a mesma versão de Python usada no treino, para que o artefato `.joblib` seja carregado com as mesmas versões das bibliotecas.

## Créditos dos dados

Dados de [F1DB](https://github.com/f1db/f1db), licenciados sob CC BY 4.0. Este projeto não é afiliado à Formula 1.
