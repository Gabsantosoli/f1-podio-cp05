"""Preparação de dados do projeto "Pódio na Fórmula 1" (CheckPoint05 - DSSC/FIAP).

Este módulo é compartilhado entre o notebook e a aplicação Streamlit, garantindo
que as duas pontas usem exatamente a mesma preparação de dados:

- ``build_dataset``: lê as tabelas da F1DB e monta uma linha por piloto por
  corrida, contendo apenas informações conhecidas ANTES da largada e o alvo
  ``podio`` (1 se o piloto terminou entre os três primeiros).
- ``add_derived_features``: variáveis derivadas calculadas dentro do pipeline.
- ``make_preprocessor``: imputação e codificação, ajustadas somente no treino.
"""

from pathlib import Path
import urllib.request
import zipfile

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

F1DB_VERSION = "v2026.15.1"
F1DB_URL = f"https://github.com/f1db/f1db/releases/download/{F1DB_VERSION}/f1db-csv.zip"
FIRST_YEAR = 2014  # início da era dos motores híbridos turbo V6

# Equipes que mudaram de nome, mas mantiveram a mesma estrutura (fábrica, sede e pessoal).
TEAM_LINEAGE = {
    "force-india": "aston-martin",
    "racing-point": "aston-martin",
    "aston-martin": "aston-martin",
    "sauber": "sauber-audi",
    "alfa-romeo": "sauber-audi",
    "kick-sauber": "sauber-audi",
    "audi": "sauber-audi",
    "toro-rosso": "racing-bulls",
    "alphatauri": "racing-bulls",
    "rb": "racing-bulls",
    "racing-bulls": "racing-bulls",
    "lotus-f1": "alpine",
    "renault": "alpine",
    "alpine": "alpine",
    "marussia": "manor",
    "manor": "manor",
}

# Motores rebatizados por patrocínio: o fabricante real é o que importa.
ENGINE_NORMALIZE = {
    "tag-heuer": "renault",
    "toro-rosso": "renault",
    "bwt-mercedes": "mercedes",
    "rbpt": "honda",
    "honda-rbpt": "honda",
}

# Colunas da tabela de resultados que só existem DEPOIS da corrida (data leakage).
LEAKAGE_COLUMNS = [
    "positionDisplayOrder", "positionNumber", "positionText", "laps", "time",
    "timeMillis", "timePenalty", "timePenaltyMillis", "gap", "gapMillis",
    "gapLaps", "interval", "intervalMillis", "reasonRetired", "points",
    "positionsGained", "pitStops", "fastestLap", "driverOfTheDay", "grandSlam",
]

NUM_FEATURES = [
    "grid",
    "posicao_classificacao",
    "largada_pit_lane",
    "ano",
    "etapa",
    "pontos_piloto_antes",
    "posicao_piloto_antes",
    "pontos_equipe_antes",
    "posicao_equipe_antes",
    "podios_piloto_ult5",
    "podios_equipe_ult5",
    "largadas_carreira",
]
CAT_FEATURES = ["equipe", "motor", "circuito", "tipo_circuito"]
FEATURES = NUM_FEATURES + CAT_FEATURES
DERIVED_FEATURES = ["perda_grid"]
TARGET = "podio"


def get_f1db_zip(path="data/f1db-csv.zip"):
    """Retorna o caminho do zip da F1DB, baixando a versão fixada se necessário."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(F1DB_URL, path)
    return path


def load_tables(zip_path):
    """Lê as tabelas da F1DB usadas no projeto."""
    names = ["races", "races-race-results", "races-driver-standings",
             "races-constructor-standings", "circuits"]
    with zipfile.ZipFile(zip_path) as z:
        return {n: pd.read_csv(z.open(f"f1db-{n}.csv"), low_memory=False) for n in names}


def _rolling_prior_sum(df, key, value, window=5):
    """Soma de ``value`` nas ``window`` corridas ANTERIORES de cada ``key`` (shift evita vazamento)."""
    return df.groupby(key)[value].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).sum()
    ).fillna(0)


def build_dataset(zip_path, first_year=FIRST_YEAR, return_raw=False):
    """Monta a base modelável: uma linha por piloto que largou em uma corrida.

    Retorna um DataFrame com colunas de identificação (não usadas no modelo),
    as variáveis pré-corrida em ``FEATURES`` e o alvo ``podio``.
    """
    t = load_tables(zip_path)
    races = t["races"][["id", "year", "round", "date", "circuitId", "circuitType"]]
    races = races.rename(columns={"id": "raceId"})
    res = t["races-race-results"].merge(
        races[["raceId", "date", "circuitId", "circuitType"]], on="raceId", how="left"
    )
    res["date"] = pd.to_datetime(res["date"])
    res = res.sort_values(["date", "positionDisplayOrder"]).reset_index(drop=True)
    raw = res.copy()

    # Alvo: pódio oficial (DSQ, DNF etc. não possuem positionNumber -> não é pódio).
    res["podio"] = (res["positionNumber"] <= 3).astype(int)

    # Categorias consistentes: linhagem da equipe e fabricante real do motor.
    res["equipe"] = res["constructorId"].map(TEAM_LINEAGE).fillna(res["constructorId"])
    res["motor"] = res["engineManufacturerId"].replace(ENGINE_NORMALIZE)

    # Largada do pit lane (PL): posição = último lugar do grid daquela corrida.
    res["largada_pit_lane"] = (res["gridPositionText"] == "PL").astype(int)
    starters = res.groupby("raceId")["gridPositionNumber"].transform("max")
    res["grid"] = np.where(res["largada_pit_lane"] == 1, starters + 1, res["gridPositionNumber"])

    # Histórico calculado sobre toda a série (inclusive antes de 2014) e sempre deslocado.
    started = res["grid"].notna().astype(int)
    res["largadas_carreira"] = started.groupby(res["driverId"]).cumsum() - started
    res["podios_piloto_ult5"] = _rolling_prior_sum(res, "driverId", "podio")
    team_race = (res.groupby(["raceId", "date", "equipe"], as_index=False)["podio"].sum()
                 .sort_values("date"))
    team_race["podios_equipe_ult5"] = _rolling_prior_sum(team_race, "equipe", "podio")
    res = res.merge(team_race[["raceId", "equipe", "podios_equipe_ult5"]],
                    on=["raceId", "equipe"], how="left")

    # Campeonato ANTES da corrida = classificação após a etapa anterior do mesmo ano.
    ds = t["races-driver-standings"][["year", "round", "driverId", "positionNumber", "points"]].copy()
    ds["round"] += 1
    ds = ds.rename(columns={"positionNumber": "posicao_piloto_antes", "points": "pontos_piloto_antes"})
    cs = t["races-constructor-standings"][["year", "round", "constructorId", "positionNumber", "points"]].copy()
    cs["round"] += 1
    cs = cs.rename(columns={"positionNumber": "posicao_equipe_antes", "points": "pontos_equipe_antes"})
    res = res.merge(ds, on=["year", "round", "driverId"], how="left")
    res = res.merge(cs, on=["year", "round", "constructorId"], how="left")
    # Sem registro prévio (1ª etapa ou estreia): 0 pontos; a posição fica ausente e é imputada no pipeline.
    res[["pontos_piloto_antes", "pontos_equipe_antes"]] = (
        res[["pontos_piloto_antes", "pontos_equipe_antes"]].fillna(0)
    )

    res = res.rename(columns={
        "qualificationPositionNumber": "posicao_classificacao",
        "year": "ano", "round": "etapa",
        "circuitId": "circuito", "circuitType": "tipo_circuito",
    })

    res = res[res["ano"] >= first_year]
    if return_raw:
        return res, raw[raw["year"] >= first_year]
    id_cols = ["raceId", "date", "driverId", "constructorId", "positionText", "gridPositionText"]
    return res[id_cols + FEATURES + [TARGET]].reset_index(drop=True)


def add_derived_features(X):
    """Variáveis derivadas, calculadas dentro do pipeline (idênticas no notebook e no app)."""
    X = X.copy()
    # Posições perdidas entre a classificação e o grid (punições por troca de peças etc.).
    X["perda_grid"] = X["grid"] - X["posicao_classificacao"]
    return X


def _derived_names(transformer, input_features):
    """Nomes das colunas após ``add_derived_features`` (necessário para ``get_feature_names_out``)."""
    return np.asarray(list(input_features) + DERIVED_FEATURES, dtype=object)


def make_preprocessor():
    """Imputação + one-hot, aprendidos apenas nos dados de treino de cada fold."""
    num = Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True))])
    cat = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    prep = ColumnTransformer(
        [("num", num, NUM_FEATURES + DERIVED_FEATURES), ("cat", cat, CAT_FEATURES)],
        verbose_feature_names_out=False,
    )
    return Pipeline([
        ("derivadas", FunctionTransformer(add_derived_features, feature_names_out=_derived_names)),
        ("prep", prep),
    ])


def make_pipeline(model):
    """Pipeline completo: preparação + estimador."""
    return Pipeline([("preparacao", make_preprocessor()), ("modelo", model)])
