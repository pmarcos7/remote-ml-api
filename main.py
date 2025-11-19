# ============================================================
# main.py - API de Regressão Linear Remota com Azure Storage + Banco Gratuito
# ============================================================

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse
from azure.storage.blob import BlobServiceClient
from azure.data.tables import TableServiceClient
import pandas as pd
import joblib
import io
import os
import uuid
from datetime import datetime
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from fastapi.middleware.cors import CORSMiddleware

# ============================================================
# 1. CRIE O APP PRIMEIRO
# ============================================================
app = FastAPI(title="ML Remote API with Azure Storage + Banco Gratuito")


# ============================================================
# 2. AGORA CONFIGURE O CORS (usando a variável 'app' que existe)
# ============================================================
origins = [
    "http://localhost:8080",  # onde vamos servir o frontend
    "http://127.0.0.1:8080",
    "http://localhost:9000",  # se você acessar swagger nessa porta
    "http://127.0.0.1:9000",
    "null",                   # <<< IMPORTANTE: Para permitir 'file://' (HTML local)
    "*"                       # opcional: permitir todos (apenas dev)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 3. AGORA CONTINUE COM O RESTO DO SEU CÓDIGO
# ============================================================
# CONFIGURAÇÃO DO AZURE STORAGE (BLOB + TABLE)
# ============================================================

AZURE_ACCOUNT_NAME = "storagecriptografia"
AZURE_ACCOUNT_KEY  = "q6GznJ3k4CV1Bg2rdgC2gFGuMVfF3xp5zNFVy1sF3LXYGtJfXF0oxbKsG08DkkYBulHNth4znRsF+AStym1w8Q=="
AZURE_CONTAINER    = "meucontainer"
TABLE_NAME         = "Treinos"

connection_string = (
    f"DefaultEndpointsProtocol=https;"
    f"AccountName={AZURE_ACCOUNT_NAME};"
    f"AccountKey={AZURE_ACCOUNT_KEY};"
    f"EndpointSuffix=core.windows.net"
)

# BLOB STORAGE
blob_service = BlobServiceClient.from_connection_string(connection_string)
container = blob_service.get_container_client(AZURE_CONTAINER)

# TABLE STORAGE (BANCO GRATUITO)
table_service = TableServiceClient.from_connection_string(connection_string)
try:
    table_client = table_service.create_table_if_not_exists(TABLE_NAME)
except:
    table_client = table_service.get_table_client(TABLE_NAME)


def registrar_treino(mae, rmse, r2):
    entity = {
        "PartitionKey": "Treinos",
        "RowKey": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2)
    }
    table_client.create_entity(entity)


def listar_treinos():
    return list(table_client.list_entities())


def ultimo_treino():
    itens = list(table_client.list_entities())
    if not itens:
        return None
    itens.sort(key=lambda x: x["timestamp"], reverse=True)
    return itens[0]


# ============================================================
# HELPERS PARA ARQUIVOS NO BLOB
# ============================================================

def upload_to_blob(blob_name: str, data: bytes):
    container.upload_blob(name=blob_name, data=data, overwrite=True)

def download_from_blob(blob_name: str) -> bytes:
    blob = container.get_blob_client(blob_name)
    return blob.download_blob().readall()

def delete_blob(blob_name: str):
    try:
        container.get_blob_client(blob_name).delete_blob()
    except:
        pass


# ============================================================
# FUNÇÃO: construir lags
# ============================================================

def build_lags(df, lags=5, target="time"):
    if all(f"lag{i}" in df.columns for i in range(1, lags + 1)):
        X = df[[f"lag{i}" for i in range(1, lags + 1)]]
        y = df[target]
        return X, y

    if target not in df.columns:
        raise ValueError("A coluna 'time' não foi encontrada no CSV.")

    s = df[target].astype(float)

    data = {f"lag{i}": s.shift(i) for i in range(1, lags + 1)}
    data[target] = s
    new_df = pd.DataFrame(data).dropna().reset_index(drop=True)

    X = new_df[[f"lag{i}" for i in range(1, lags + 1)]]
    y = new_df[target]

    return X, y


# ============================================================
# ENDPOINT: UPLOAD TRAIN
# ============================================================

@app.post("/upload/train")
async def upload_train(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    upload_to_blob("train_upload.csv", contents)
    return {"status": "ok", "rows": len(df), "columns": list(df.columns)}


# ============================================================
# ENDPOINT: TRAIN
# ============================================================

@app.post("/train")
async def train_model(lags: int = Form(5), cv_splits: int = Form(5)):

    df = pd.read_csv(io.BytesIO(download_from_blob("train_upload.csv")))

    X, y = build_lags(df, lags=lags)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    tscv = TimeSeriesSplit(n_splits=cv_splits)
    maes, rmses, r2s = [], [], []

    for tr, val in tscv.split(X_scaled):
        Xtr, Xval = X_scaled[tr], X_scaled[val]
        ytr, yval = y.iloc[tr], y.iloc[val]

        m = LinearRegression()
        m.fit(Xtr, ytr)
        preds = m.predict(Xval)

        maes.append(mean_absolute_error(yval, preds))
        rmses.append(mean_squared_error(yval, preds, squared=False))
        r2s.append(r2_score(yval, preds))

    model = LinearRegression()
    model.fit(X_scaled, y)

    # salvar no blob
    b = io.BytesIO()
    joblib.dump(model, b)
    upload_to_blob("model.joblib", b.getvalue())

    b = io.BytesIO()
    joblib.dump(scaler, b)
    upload_to_blob("scaler.joblib", b.getvalue())

    # registrar no banco
    registrar_treino(np.mean(maes), np.mean(rmses), np.mean(r2s))

    metrics = {
        "MAE": float(np.mean(maes)),
        "RMSE": float(np.mean(rmses)),
        "R2": float(np.mean(r2s))
    }

    return {"status": "trained", "metrics": metrics}


# ============================================================
# ENDPOINT: UPLOAD TEST
# ============================================================

@app.post("/upload/test")
async def upload_test(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    upload_to_blob("test_upload.csv", contents)
    return {"status": "ok", "rows": len(df), "columns": list(df.columns)}


# ============================================================
# ENDPOINT: PREDICT
# ============================================================

@app.post("/predict")
async def predict():

    model = joblib.load(io.BytesIO(download_from_blob("model.joblib")))
    scaler = joblib.load(io.BytesIO(download_from_blob("scaler.joblib")))
    df = pd.read_csv(io.BytesIO(download_from_blob("test_upload.csv")))

    has_labels = "time" in df.columns

    X, y = build_lags(df, lags=5)
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)

    out = pd.DataFrame({"predicted": preds})

    if has_labels:
        out["actual"] = y.values[:len(preds)]
        out["error"] = out["actual"] - out["predicted"]

    b = io.BytesIO()
    out.to_csv(b, index=False)
    upload_to_blob("predictions.csv", b.getvalue())

    return {"status": "ok"}


# ============================================================
# ENDPOINT: DOWNLOAD PREDICTIONS
# ============================================================

@app.get("/download/predictions")
async def download_predictions():
    data = download_from_blob("predictions.csv")
    path = "/tmp/predictions.csv"
    with open(path, "wb") as f:
        f.write(data)
    return FileResponse(path, filename="predictions.csv")


# ============================================================
# ENDPOINT: LOGS (BANCO)
# ============================================================

@app.get("/logs")
async def logs():
    return {"logs": listar_treinos()}


# ============================================================
# ENDPOINT: MÉTRICAS DO ÚLTIMO TREINO
# ============================================================

@app.get("/metrics/last")
async def last_metrics():
    item = ultimo_treino()
    if item is None:
        return {"message": "Nenhum treino encontrado"}
    return item


# ============================================================
# RESET
# ============================================================

@app.post("/reset")
async def reset():
    for f in ["train_upload.csv", "test_upload.csv", "model.joblib", "scaler.joblib", "predictions.csv"]:
        delete_blob(f)
    return {"status": "reset"}