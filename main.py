# ============================================================
# main.py - API de Regressão Linear Remota (Azure Blob + Table Storage)
# (com criptografia Fernet simples integrada)
# ============================================================

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from azure.storage.blob import BlobServiceClient
from azure.data.tables import TableServiceClient
import pandas as pd
import joblib
import io
import os
import uuid
from datetime import datetime
import numpy as np
import traceback

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from fastapi.middleware.cors import CORSMiddleware
from cryptography.fernet import Fernet # Mantido o import aqui, mas é usado abaixo

# ============================================================
# CONFIGURAÇÕES E VARIÁVEIS
# ============================================================
AZURE_ACCOUNT_NAME = os.getenv("AZURE_ACCOUNT_NAME")
AZURE_ACCOUNT_KEY = os.getenv("AZURE_ACCOUNT_KEY")
AZURE_CONTAINER = os.getenv("AZURE_CONTAINER", "meucontainer")
TABLE_NAME = os.getenv("TABLE_NAME", "Treinos")
PREDICTIONS_TABLE_NAME = os.getenv("PREDICTIONS_TABLE_NAME", "Predicoes")
RAW_CHART_BLOB_NAME = "chart_data_raw.csv" # Novo nome para o arquivo RAW de treino/teste
RAW_PREDICTIONS_BLOB_NAME = "predictions_raw.csv" # Novo nome para o arquivo RAW de previsões

if not AZURE_ACCOUNT_NAME or not AZURE_ACCOUNT_KEY:
    raise RuntimeError("Defina AZURE_ACCOUNT_NAME e AZURE_ACCOUNT_KEY como secrets.")

connection_string = (
    f"DefaultEndpointsProtocol=https;"
    f"AccountName={AZURE_ACCOUNT_NAME};"
    f"AccountKey={AZURE_ACCOUNT_KEY};"
    f"EndpointSuffix=core.windows.net"
)

# ============================================================
# INICIALIZAÇÃO DO BLOB E TABLE STORAGE
# ============================================================
blob_service = BlobServiceClient.from_connection_string(connection_string)
blob_container = blob_service.get_container_client(AZURE_CONTAINER)
try:
    blob_container.create_container()
except Exception:
    pass

table_service = TableServiceClient.from_connection_string(connection_string)
table_client = table_service.create_table_if_not_exists(TABLE_NAME)
predictions_table_client = table_service.create_table_if_not_exists(PREDICTIONS_TABLE_NAME)

# ============================================================
# HELPERS (não-criptográficos)
# ============================================================
def safe_float(x):
    try:
        return float(x)
    except Exception:
        try:
            return float(np.asarray(x).item())
        except Exception:
            return None

def delete_blob(blob_name: str):
    try:
        blob_container.get_blob_client(blob_name).delete_blob()
    except Exception:
        pass

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

def registrar_treino(mae, rmse, r2):
    entity = {
        "PartitionKey": "Treinos",
        "RowKey": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        "MAE": safe_float(mae),
        "RMSE": safe_float(rmse),
        "R2": safe_float(r2)
    }
    table_client.create_entity(entity)

def registrar_predicao(training_id, input_row, predicted_value):
    entity = {
        "PartitionKey": training_id or "unknown",
        "RowKey": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        "PredictedValue": safe_float(predicted_value),
        **{f"lag{k}": safe_float(v) for k, v in input_row.items() if str(k).startswith('lag')}
    }
    predictions_table_client.create_entity(entity)

# ============================================================
# FASTAPI
# ============================================================
app = FastAPI(title="ML Remote API with Azure Storage")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HTML_DASHBOARD = "<h1>ML Remote API funcionando</h1>"

# Rota principal (agora abaixo, depois da definição de todas as rotas)

# ============================================================
# CRIPTOGRAFIA - implementação simples com Fernet
# ============================================================
KEY_BLOB_NAME = "fernet.key"

# Funções raw para manipular blobs sem criptografia (usadas para chave e para dados de gráfico RAW)
def raw_upload_blob(blob_name: str, data: bytes):
    try:
        blob_client = blob_container.get_blob_client(blob_name)
        blob_client.upload_blob(data, overwrite=True)
    except Exception as e:
        raise RuntimeError(f"Erro raw upload blob {blob_name}: {e}")

def raw_download_blob(blob_name: str) -> bytes:
    try:
        blob_client = blob_container.get_blob_client(blob_name)
        if not blob_client.exists():
            raise RuntimeError(f"Blob {blob_name} não existe")
        return blob_client.download_blob().readall()
    except Exception as e:
        raise RuntimeError(f"Erro raw download blob {blob_name}: {e}")

# Gera ou carrega chave (usa raw_* para não depender de decrypt)
def get_crypto_key():
    try:
        key = raw_download_blob(KEY_BLOB_NAME)
        return Fernet(key)
    except Exception:
        # cria nova chave e sobe (raw)
        key = Fernet.generate_key()
        raw_upload_blob(KEY_BLOB_NAME, key)
        return Fernet(key)

# inicializa o objeto Fernet (global)
FERNET = get_crypto_key()

# ------------------------------------------------------
# Sobrescrevemos as funções upload/download usadas pelo app (criptografadas)
# ------------------------------------------------------
def upload_to_blob(blob_name: str, data: bytes):
    """ Envia bytes criptografados para o blob. """
    try:
        encrypted = FERNET.encrypt(data)
        blob_client = blob_container.get_blob_client(blob_name)
        blob_client.upload_blob(encrypted, overwrite=True)
    except Exception as e:
        raise RuntimeError(f"Erro upload blob {blob_name}: {e}")

def download_from_blob(blob_name: str) -> bytes:
    """ Faz download do blob e descriptografa antes de retornar bytes. """
    try:
        blob_client = blob_container.get_blob_client(blob_name)
        if not blob_client.exists():
            raise RuntimeError(f"Blob {blob_name} não encontrado")
        encrypted = blob_client.download_blob().readall()
        return FERNET.decrypt(encrypted)
    except Exception as e:
        raise RuntimeError(f"Erro download blob {blob_name}: {e}")

# Auxiliar: ler blob (raw) apenas para obter tamanho criptografado sem decrypt
def get_encrypted_blob_bytes(blob_name: str) -> bytes:
    try:
        blob_client = blob_container.get_blob_client(blob_name)
        if not blob_client.exists():
            raise RuntimeError("blob não existe")
        return blob_client.download_blob().readall()
    except Exception as e:
        raise RuntimeError(f"Erro lendo blob (raw) {blob_name}: {e}")

# ============================================================
# ROTAS: upload, train, predict (usando upload_to_blob/download_from_blob criptografados)
# ============================================================

@app.post("/upload/train")
async def upload_train(file: UploadFile = File(...)):
    contents = await file.read()
    # valida CSV (apenas leitura)
    df = pd.read_csv(io.BytesIO(contents))
    # salva criptografado no blob (para treino seguro)
    upload_to_blob("train_upload.csv", contents)
    return {"status": "ok", "rows": len(df), "columns": list(df.columns), "security": "encrypted"}

@app.post("/upload/test")
async def upload_test(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    # salva criptografado no blob (para predict seguro)
    upload_to_blob("test_upload.csv", contents)
    return {"status": "ok", "rows": len(df), "columns": list(df.columns), "security": "encrypted"}

# --- NOVO: Rota para upload de dados NÃO-CRIPTOGRAFADOS (RAW) para gráficos ---
@app.post("/upload/chart_data")
async def upload_chart_data(file: UploadFile = File(...)):
    """ Salva o arquivo CSV diretamente no blob (sem criptografia) para uso em gráficos. """
    contents = await file.read()
    try:
        pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao ler CSV: {e}")

    # salva SEM criptografia
    raw_upload_blob(RAW_CHART_BLOB_NAME, contents)
    return {"status": "ok", "security": "raw/no_encryption", "blob_name": RAW_CHART_BLOB_NAME}
# --------------------------------------------------------------------------------

@app.post("/train")
async def train_model(lags: int = Form(5), cv_splits: int = Form(5)):
    # baixa e descriptografa o CSV de treino
    data = download_from_blob("train_upload.csv")
    df = pd.read_csv(io.BytesIO(data))
    X, y = build_lags(df, lags=lags)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    maes, rmses, r2s = [], [], []
    for tr, val in tscv.split(X_scaled):
        m = LinearRegression()
        m.fit(X_scaled[tr], y.iloc[tr])
        preds = m.predict(X_scaled[val])
        maes.append(mean_absolute_error(y.iloc[val], preds))
        rmses.append(np.sqrt(mean_squared_error(y.iloc[val], preds)))
        r2s.append(r2_score(y.iloc[val], preds))
    model = LinearRegression()
    model.fit(X_scaled, y)
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    model_blob_name = f"model_{ts}.joblib"
    scaler_blob_name = f"scaler_{ts}.joblib"
    b = io.BytesIO(); joblib.dump(model, b); upload_to_blob(model_blob_name, b.getvalue())
    b2 = io.BytesIO(); joblib.dump(scaler, b2); upload_to_blob(scaler_blob_name, b2.getvalue())
    metrics = {"MAE": float(np.mean(maes)), "RMSE": float(np.mean(rmses)), "R2": float(np.mean(r2s))}
    registrar_treino(metrics["MAE"], metrics["RMSE"], metrics["R2"])
    return {"status": "trained", "metrics": metrics, "model_blob": model_blob_name}

@app.post("/predict")
async def predict(lags: int = Form(5)):
    # pega o model mais recente (lista de blobs criptografados com prefix model_)
    blobs = list(blob_container.list_blobs(name_starts_with="model_"))
    if not blobs: raise HTTPException(status_code=404, detail="Modelo não encontrado")
    blobs_sorted = sorted(blobs, key=lambda b: b.last_modified or datetime.min, reverse=True)
    model_blob = blobs_sorted[0].name
    scaler_blob = list(blob_container.list_blobs(name_starts_with="scaler_"))
    scaler_blob = sorted(scaler_blob, key=lambda b: b.last_modified or datetime.min, reverse=True)[0].name if scaler_blob else None
    # carrega descriptografado
    model = joblib.load(io.BytesIO(download_from_blob(model_blob)))
    scaler = joblib.load(io.BytesIO(download_from_blob(scaler_blob))) if scaler_blob else None

    data = download_from_blob("test_upload.csv")
    df = pd.read_csv(io.BytesIO(data))
    X, y = build_lags(df, lags=lags)
    X_scaled = scaler.transform(X) if scaler else X
    preds = model.predict(X_scaled)
    out = pd.DataFrame({"predicted": preds})
    if "time" in df.columns:
        out["actual"] = y.values[:len(preds)]
        out["error"] = out["actual"] - out["predicted"]

    # salva CSV CRIPTOGRAFADO (segurança/backup)
    b = io.BytesIO(); out.to_csv(b, index=False); upload_to_blob("predictions.csv", b.getvalue())

    # --- NOVO: salva CSV de previsões NÃO-CRIPTOGRAFADO (RAW) para o gráfico ---
    b_raw = io.BytesIO(); out.to_csv(b_raw, index=False); raw_upload_blob(RAW_PREDICTIONS_BLOB_NAME, b_raw.getvalue())
    # --------------------------------------------------------------------------

    training_id = "last_run"
    for i, row in out.iterrows():
        registrar_predicao(training_id, X.iloc[i].to_dict(), float(row["predicted"]))

    return {"status": "ok", "n": len(preds), "raw_predictions_blob": RAW_PREDICTIONS_BLOB_NAME}

# ============================================================
# DOWNLOAD de previsões (descriptografado) - Original
# ============================================================
@app.get("/download/predictions")
async def download_predictions():
    try:
        data = download_from_blob("predictions.csv")
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
    path = "/tmp/predictions.csv"
    with open(path, "wb") as f:
        f.write(data)
    return FileResponse(path, filename="predictions.csv")

# Nova rota: faz download de qualquer blob descriptografado por nome (usado pelo frontend para "baixar descriptografado")
@app.get("/download/decrypted/{filename}")
async def download_decrypted(filename: str):
    """
    Baixa o blob criptografado e retorna bytes descriptografados para download.
    filename: nome do blob (ex: train_upload.csv, test_upload.csv, predictions.csv)
    """
    try:
        data = download_from_blob(filename)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
    # stream para o cliente
    return StreamingResponse(io.BytesIO(data), media_type="application/octet-stream",
                             headers={"Content-Disposition": f"attachment; filename={filename}"})

# --- NOVO: Rota para download de dados NÃO-CRIPTOGRAFADOS (RAW) para gráficos ---
@app.get("/download/chart_data/{filename}")
async def download_chart_data(filename: str):
    """
    Baixa um arquivo de dados RAW (sem criptografia).
    filename pode ser 'chart_data_raw.csv' ou 'predictions_raw.csv'.
    """
    if filename not in [RAW_CHART_BLOB_NAME, RAW_PREDICTIONS_BLOB_NAME]:
        raise HTTPException(status_code=400, detail="Nome de arquivo RAW inválido.")
    try:
        # usa a função raw_download_blob
        data = raw_download_blob(filename)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Dados de gráfico RAW ({filename}) não encontrados: {e}")

    # stream para o cliente
    return StreamingResponse(io.BytesIO(data), media_type="application/octet-stream",
                             headers={"Content-Disposition": f"attachment; filename={filename}"})
# --------------------------------------------------------------------------------

# ============================================================
# Predições (Table) e Logs - mantidos
# ============================================================
@app.get("/predictions/table")
async def predictions_table():
    try:
        items = list(predictions_table_client.list_entities(results_per_page=50))
        predictions_list = []
        for item in items:
            try:
                ts = item.timestamp.isoformat() if hasattr(item.timestamp, 'isoformat') else str(item.timestamp)
            except Exception:
                ts = str(item.timestamp)
            predicted = safe_float(getattr(item, "PredictedValue", None))
            input_lags = {}
            for k, v in item.items():
                if str(k).startswith("lag"):
                    input_lags[k] = safe_float(v)
            predictions_list.append({
                "RowKey": getattr(item, "RowKey", ""),
                "PartitionKey": getattr(item, "PartitionKey", ""),
                "timestamp": ts,
                "PredictedValue": predicted,
                "InputLags": input_lags
            })
        predictions_list.sort(key=lambda x: x["timestamp"], reverse=True)
        return {"predictions_table": predictions_list}
    except Exception as e:
        print("Erro /predictions/table:", traceback.format_exc())
        return {"predictions_table": [], "error": str(e)}

@app.get("/logs")
async def logs():
    items = list(table_client.list_entities())
    return {"logs": items}

@app.get("/metrics/last")
async def last_metrics():
    itens = list(table_client.list_entities())
    if not itens: return {"message": "Nenhum treino encontrado"}
    itens.sort(key=lambda x: x["timestamp"], reverse=True)
    return itens[0]

@app.post("/reset")
async def reset():
    # Adicionando os novos arquivos RAW no reset
    files_to_delete = ["train_upload.csv", "test_upload.csv", "model.joblib", "scaler.joblib", "predictions.csv", RAW_CHART_BLOB_NAME, RAW_PREDICTIONS_BLOB_NAME]
    for f in files_to_delete:
        delete_blob(f)
    return {"status": "reset"}

# ============================================================
# ROTAS DE CRIPTO - info e stats (mantidas)
# ============================================================
@app.get("/crypto/info")
async def crypto_info():
    """
    Retorna informações básicas sobre a chave e se o Fernet está ativo.
    """
    try:
        key = raw_download_blob(KEY_BLOB_NAME)
        key_str = key.decode("utf-8")
        return {
            "key_file": KEY_BLOB_NAME,
            "key_length": len(key_str),
            "key_preview": key_str[:32] + "..." if len(key_str) > 32 else key_str,
            "fernet_active": True
        }
    except Exception as e:
        return {"fernet_active": False, "error": str(e)}

@app.get("/crypto/stats")
async def crypto_stats():
    """
    Retorna tamanhos criptografados vs descriptografados para alguns arquivos de exemplo.
    """
    sample_files = ["train_upload.csv", "test_upload.csv", "predictions.csv"]
    stats = {}
    for f in sample_files:
        try:
            blob_client = blob_container.get_blob_client(f)
            if not blob_client.exists():
                stats[f] = {"exists": False}
                continue
            # pega tamanho criptografado (propriedade do blob)
            try:
                props = blob_client.get_blob_properties()
                encrypted_size = props.size
            except Exception:
                encrypted_size = None
            # além disso, fazemos download e descriptografamos para medir tamanho claro
            try:
                encrypted_bytes = blob_client.download_blob().readall()
                clear_bytes = FERNET.decrypt(encrypted_bytes)
                clear_size = len(clear_bytes)
                encrypted_len_actual = len(encrypted_bytes)
                stats[f] = {
                    "exists": True,
                    "encrypted_size_reported": encrypted_size,
                    "encrypted_size_downloaded": encrypted_len_actual,
                    "decrypted_size": clear_size
                }
            except Exception as e:
                stats[f] = {"exists": True, "error_decrypting_or_downloading": str(e)}
        except Exception as e:
            stats[f] = {"exists": False, "error": str(e)}
    return {"stats": stats}

# ============================================================
# FRONTEND EMBUTIDO (Mantido, mas com ajustes para as novas rotas)
# ============================================================
API_URL = "https://remote-ml-api.mangorock-79845fa8.centralus.azurecontainerapps.io"
CHART_DATA_RAW = "chart_data_raw.csv"
PREDICTIONS_RAW = "predictions_raw.csv"

HTML_TEMPLATE = """
<!doctype html>
<html lang="pt-BR">
<head>
    <meta charset="utf-8" />
    <title>ML Remote — Dashboard</title>
    <meta name="viewport" content="width=device-width,initial-scale=1" />
    <style>
        body{font-family:Inter,system-ui,Segoe UI,Arial;padding:18px;background:#f6f8fb;color:#111}
        h1{margin:0 0 10px}
        .box{background:#fff;border-radius:8px;padding:14px;margin-bottom:12px;box-shadow:0 1px 4px rgba(10,10,10,0.06)}
        label{display:block;margin:8px 0 6px;font-weight:600}
        button{padding:8px 12px;border-radius:6px;border:0;background:#2563eb;color:#fff;cursor:pointer;margin-right:5px;margin-bottom:5px;} /* Ajuste para botões */
        input[type=file]{padding:6px}
        table{width:100%;border-collapse:collapse;margin-top:8px}
        th,td{padding:6px;border-bottom:1px solid #eee;text-align:left;font-size:13px}
        .row{display:flex;gap:12px;flex-wrap:wrap}
        .col{flex:1;min-width:240px}
        pre{background:#0b1220;color:#dbeafe;padding:10px;border-radius:6px;overflow:auto}
    </style>
</head>
<body>
    <h1>ML Remote — Dashboard</h1>

    <div class="box row">
        <div class="col">
            <h3>1) Upload e Treino</h3>
            <label>Arquivo de treino (.csv)</label>
            <input id="trainFile" type="file" accept=".csv" />
            <div style="margin-top:8px">
                <button onclick="uploadTrain()">Upload Train (Cript.)</button>
                <button onclick="uploadChartData('trainFile', '__CHART_DATA_RAW__')">Upload Gráfico (RAW)</button>
                <button onclick="train()">Treinar</button>
            </div>
            <div id="trainResult" style="margin-top:8px"></div>
        </div>

        <div class="col">
            <h3>2) Upload teste e Previsão</h3>
            <label>Arquivo de teste (.csv)</label>
            <input id="testFile" type="file" accept=".csv" />
            <div style="margin-top:8px">
                <button onclick="uploadTest()">Upload Teste (Cript.)</button>
                <button onclick="predict()">Prever</button>
                <br>
                <button onclick="downloadPredictions()">Baixar Previsões (Cript. - Descript.)</button>
                <button onclick="downloadRawData('__PREDICTIONS_RAW__')">Baixar Previsões (RAW)</button>
                <button onclick="downloadRawData('__CHART_DATA_RAW__')">Baixar Treino/Teste (RAW)</button>
            </div>
            <div id="predictResult" style="margin-top:8px"></div>
        </div>

        <div class="col">
            <h3>3) Logs e Métricas</h3>
            <div style="display:flex;gap:8px;align-items:center;margin-bottom:8px">
                <button onclick="getLastMetrics()">Último treino (Table)</button>
                <button onclick="getLogs()">Ver logs (Table)</button>
            </div>
            <div style="display:flex;gap:8px;align-items:center">
                <button onclick="getPredictionsTable()">Ver predições (Table)</button>
            </div>
            <div id="metrics" style="margin-top:8px"></div>
        </div>
    </div>

    <div class="box">
        <h3>Criptografia</h3>
        <button onclick="getCryptoInfo()">Ver detalhes da criptografia</button>
        <button onclick="getCryptoStats()">Tamanhos (criptografado vs descriptografado)</button>
        <pre id="cryptoInfo">Nenhuma informação carregada ainda.</pre>
    </div>

    <div class="box">
        <h3>Preview das previsões (Descript.)</h3>
        <div id="predPreview">Nenhuma previsão gerada ainda.</div>
    </div>

    <div class="box">
        <h3>Console</h3>
        <pre id="console">Pronto.</pre>
    </div>

<script>
const API_BASE = "__API_URL__";
const CHART_DATA_RAW = "__CHART_DATA_RAW__";
const PREDICTIONS_RAW = "__PREDICTIONS_RAW__";

function log(msg){
    const c = document.getElementById('console');
    c.textContent = `${new Date().toISOString()} — ${msg}\n` + c.textContent;
}

// ----------------------------------------------------
// Novas Funções RAW
// ----------------------------------------------------
async function uploadChartData(fileId, targetBlob){
    const f = document.getElementById(fileId).files[0];
    if(!f){ alert(`Selecione o CSV (${fileId})`); return; }
    const fd = new FormData();
    fd.append('file', f);
    log('Enviando dados para gráfico (será salvo RAW)...');
    const res = await fetch(`${API_BASE}/upload/chart_data`, { method:'POST', body: fd });
    const j = await res.json();
    log('Upload chart data (RAW): ' + JSON.stringify(j));
}

async function downloadRawData(filename){
    const a = document.createElement('a');
    a.href = `${API_BASE}/download/chart_data/${filename}`;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    log('Solicitado download RAW: ' + filename);
}
// ----------------------------------------------------

async function uploadTrain(){
    const f = document.getElementById('trainFile').files[0];
    if(!f){ alert('Selecione o CSV de treino'); return; }
    const fd = new FormData();
    fd.append('file', f);
    log('Enviando treino (será criptografado no servidor)...');
    const res = await fetch(`${API_BASE}/upload/train`, { method:'POST', body: fd });
    const j = await res.json();
    log('Upload train: ' + JSON.stringify(j));
    document.getElementById('trainResult').innerText = JSON.stringify(j);
}

async function uploadTest(){
    const f = document.getElementById('testFile').files[0];
    if(!f){ alert('Selecione o CSV de teste'); return; }
    const fd = new FormData();
    fd.append('file', f);
    log('Enviando teste (será criptografado no servidor)...');
    const res = await fetch(`${API_BASE}/upload/test`, { method:'POST', body: fd });
    const j = await res.json();
    log('Upload test: ' + JSON.stringify(j));
    document.getElementById('predictResult').innerText = JSON.stringify(j);
}

async function train(){
    log('Iniciando treino...');
    const res = await fetch(`${API_BASE}/train`, { method:'POST',
        headers: {"Content-Type": "application/x-www-form-urlencoded"},
        body: new URLSearchParams({ lags: 5, cv_splits: 5 })
    });
    const j = await res.json();
    log('Treino: ' + JSON.stringify(j));
    document.getElementById('trainResult').innerText = JSON.stringify(j);
}

async function predict(){
    log('Rodando predict...');
    const res = await fetch(`${API_BASE}/predict`, { method:'POST' });
    const j = await res.json();
    log('Predict: ' + JSON.stringify(j));
    document.getElementById('predictResult').innerText = JSON.stringify(j);
    await showPredictionsPreview();
}

async function downloadPredictions(){
    // baixa o arquivo predictions.csv criptografado e o servidor o descriptografa
    const a = document.createElement('a');
    a.href = `${API_BASE}/download/predictions`;
    a.download = 'predictions_decrypted.csv';
    document.body.appendChild(a);
    a.click();
    a.remove();
    log('Solicitado download (predictions.csv) - Descriptografado.');
}

async function downloadDecrypted(filename){
    const a = document.createElement('a');
    a.href = `${API_BASE}/download/decrypted/${filename}`;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    log('Solicitado download descriptografado: ' + filename);
}

async function getCryptoInfo(){
    try{
        const res = await fetch(`${API_BASE}/crypto/info`);
        const j = await res.json();
        document.getElementById('cryptoInfo').innerText = JSON.stringify(j, null, 2);
        log('Crypto info carregada.');
    }catch(e){
        log('Erro crypto info: ' + e);
    }
}

async function getCryptoStats(){
    try{
        const res = await fetch(`${API_BASE}/crypto/stats`);
        const j = await res.json();
        document.getElementById('cryptoInfo').innerText = JSON.stringify(j, null, 2);
        log('Crypto stats carregada.');
    }catch(e){
        log('Erro crypto stats: ' + e);
    }
}

async function showPredictionsPreview(){
    try{
        const res = await fetch(`${API_BASE}/download/predictions`);
        if(!res.ok){ log('Nenhuma previsão disponível.'); return; }
        const txt = await res.text();
        const lines = txt.trim().split('\n').slice(0, 11).join('\n');
        document.getElementById('predPreview').innerText = lines;
    }catch(e){
        log('Erro preview: ' + e);
    }
}

async function getLogs(){
    const res = await fetch(`${API_BASE}/logs`);
    const j = await res.json();
    document.getElementById('metrics').innerText = JSON.stringify(j, null, 2);
    log('Logs carregados.');
}

async function getPredictionsTable(){
    const res = await fetch(`${API_BASE}/predictions/table`);
    const j = await res.json();
    document.getElementById('metrics').innerText = JSON.stringify(j, null, 2);
    log('Tabela de predições carregada.');
}

async function getLastMetrics(){
    const res = await fetch(`${API_BASE}/metrics/last`);
    const j = await res.json();
    document.getElementById('metrics').innerText = JSON.stringify(j, null, 2);
    log('Últimas métricas carregadas.');
}

log('Frontend pronto. API base: ' + API_BASE);
</script>
</body>
</html>
"""

HTML_DASHBOARD = HTML_TEMPLATE.replace("__API_URL__", API_URL).replace("__CHART_DATA_RAW__", RAW_CHART_BLOB_NAME).replace("__PREDICTIONS_RAW__", RAW_PREDICTIONS_BLOB_NAME)

@app.get("/", response_class=HTMLResponse)
async def serve_frontend_embedded():
    return HTMLResponse(content=HTML_DASHBOARD, status_code=200)

# ============================================================
# FIM DO ARQUIVO
# ============================================================