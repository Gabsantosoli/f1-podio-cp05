# Pódio na Fórmula 1: Random Forest, XGBoost e LightGBM

CheckPoint05 de Data Science & Statistical Computing, FIAP 2026. Prof. Jones Egydio.

| Integrante | RM |
|---|---|
| Thiago Nascimento | 561967 |
| Vinicius Herreira | 565662 |
| Gustavo Rocha | 564152 |
| Gabriel Santos | 562419 |

**Aplicação Streamlit:** https://f1-podio-cp05.streamlit.app/

**Repositório:** https://github.com/Gabsantosoli/f1-podio-cp05

## Problema

Antes da largada de uma corrida de Fórmula 1, qual a probabilidade de um piloto terminar no pódio?

Para responder, usamos informações conhecidas antes da corrida: posição de largada, resultado da classificação, situação do piloto e da equipe no campeonato, pódios recentes, equipe, motor e circuito. É um problema de classificação binária (pódio ou não). Cada observação é um piloto em uma corrida.

Usamos as temporadas de 2014 a 2026 (até a 15ª etapa), com 5.423 observações e 14,8% de pódios. A métrica principal é a ROC-AUC, porque as classes são desbalanceadas.

## Dados

Os dados vêm da [F1DB](https://github.com/f1db/f1db), versão v2026.15.1, com licença CC BY 4.0.

O arquivo `data/f1db-csv.zip` já está no repositório. Se ele não estiver na pasta, o notebook baixa automaticamente a mesma versão em:
`https://github.com/f1db/f1db/releases/download/v2026.15.1/f1db-csv.zip`

## Como executar

É preciso ter Python 3.12 ou superior.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install jupyter
```

No Linux ou macOS, troque a segunda linha por `source .venv/bin/activate`.

Para rodar o notebook, abra `Checkpoint05_F1_Podio.ipynb` e execute todas as células. Leva cerca de 6 minutos.

Para rodar a aplicação:

```bash
streamlit run app.py
```

## Resumo dos resultados

| Modelo | ROC-AUC na validação cruzada |
|---|---|
| Referência: ordenar só pela posição de largada | 0,923 |
| Random Forest (melhor configuração) | 0,943 |
| LightGBM (melhor configuração) | 0,943 |
| XGBoost ajustado com Optuna (modelo final) | 0,944 |

No conjunto de teste, o modelo final teve ROC-AUC de 0,942, praticamente igual à validação cruzada.

Os modelos com os parâmetros padrão decoraram os dados de treino (ROC-AUC de 1,0 no treino). Depois do ajuste com Grid Search e Optuna, os modelos ficaram mais simples e generalizaram melhor. A posição de largada e a classificação são as variáveis mais importantes, e a situação no campeonato e os pódios recentes ajustam a probabilidade.

A aplicação usa o mesmo modelo do notebook. Testamos 6 casos e as previsões foram idênticas nos dois.
