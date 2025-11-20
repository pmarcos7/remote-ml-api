# ============================================================
# main.py - API de Regressão Linear Remota com Azure Storage + Banco Gratuito
# ============================================================

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse, HTMLResponse 
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
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    "http://localhost:9000",
    "http://127.0.0.1:9000",
    "null",
    "*" # Permissão ampla, útil para ambientes de teste como o ACA
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 3. FRONTEND EMBUTIDO E ROTA RAIZ (CÓDIGO NOVO E CORRIGIDO)
# ============================================================

# ATENÇÃO: SUBSTITUA ESTE VALOR pela URL completa do seu Container App!
API_URL = "https://remote-ml-api.mangorock-79845fa8.centralus.azurecontainerapps.io" 

# HTML_DASHBOARD NÃO É MAIS UMA F-STRING, USA .replace() PARA EVITAR CONFLITOS DE CHAVES {}
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
        button{padding:8px 12px;border-radius:6px;border:0;background:#2563eb;color:#fff;cursor:pointer}
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
                <button onclick="uploadTrain()">Upload Train</button>
                <button onclick="train()">Treinar</button>
            </div>
            <div id="trainResult" style="margin-top:8px"></div>
        </div>

        <div class="col">
            <h3>2) Upload teste e Previsão</h3>
            <label>Arquivo de teste (.csv)</label>
            <input id="testFile" type="file" accept=".csv" />
            <div style="margin-top:8px">
                <button onclick="uploadTest()">Upload Test</button>
                <button onclick="predict()">Prever</button>
                <button onclick="downloadPredictions()">Baixar Previsões</button>
            </div>
            <div id="predictResult" style="margin-top:8px"></div>
        </div>

        <div class="col">
            <h3>3) Logs e Métricas</h3>
            <div style="display:flex;gap:8px;align-items:center">
                <button onclick="getLastMetrics()">Último treino</button>
                <button onclick="getLogs()">Ver logs</button>
                <button onclick="exportLogsCSV()">Exportar logs (CSV)</button>
            </div>
            <div id="metrics" style="margin-top:8px"></div>
        </div>
    </div>

    <div class="box">
        <h3>Preview das previsões</h3>
        <div id="predPreview">Nenhuma previsão gerada ainda.</div>
    </div>

    <div class="box">
        <h3>Console</h3>
        <pre id="console">Pronto.</pre>
    </div>

<script>
// CORREÇÃO AQUI: A URL da API é injetada via Python antes de ser servida
const API_BASE = "__API_URL__"; 

function log(msg){
    const c = document.getElementById('console');
    // CORRIGIDO: Removido caracteres de escape desnecessários para o navegador
    c.textContent = `${new Date().toISOString()} — ${msg}\\n` + c.textContent; 
}

// O restante das suas funções JS (uploadTrain, train, etc.) usam API_BASE corretamente.

async function uploadTrain(){
    const f = document.getElementById('trainFile').files[0];
    if(!f){ alert('Selecione o CSV de treino'); return; }
    const fd = new FormData();
    fd.append('file', f);
    log('Enviando treino...');
    const res = await fetch(`${API_BASE}/upload/train`, { method:'POST', body: fd });
    const j = await res.json();
    log('Upload train: ' + JSON.stringify(j));
    document.getElementById('trainResult').innerText = JSON.stringify(j);
}

async function train(){
    log('Iniciando treino...');
    const res = await fetch(`${API_BASE}/train`, { method:'POST', body: new URLSearchParams({lags:5, cv_splits:5}) });
    const j = await res.json();
    log('Treino finalizado: ' + JSON.stringify(j));
    document.getElementById('trainResult').innerText = JSON.stringify(j);
    getLastMetrics();
}

async function uploadTest(){
    const f = document.getElementById('testFile').files[0];
    if(!f){ alert('Selecione o CSV de teste'); return; }
    const fd = new FormData();
    fd.append('file', f);
    log('Enviando teste...');
    const res = await fetch(`${API_BASE}/upload/test`, { method:'POST', body: fd });
    const j = await res.json();
    log('Upload test: ' + JSON.stringify(j));
    document.getElementById('predictResult').innerText = JSON.stringify(j);
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
    const url = `${API_BASE}/download/predictions`;
    log('Baixando ' + url);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'predictions.csv';
    document.body.appendChild(a);
    a.click();
    a.remove();
}

async function getLastMetrics(){
    try{
        const res = await fetch(`${API_BASE}/metrics/last`);
        const j = await res.json();
        log('Último treino: ' + JSON.stringify(j));
        document.getElementById('metrics').innerText = JSON.stringify(j, null, 2);
    }catch(e){
        log('Erro ao buscar último treino: ' + e);
    }
}

let logsCache = [];
async function getLogs(){
    const res = await fetch(`${API_BASE}/logs`);
    const j = await res.json();
    logsCache = j.logs || [];
    log('Logs recebidos: ' + logsCache.length);
    const html = ['<table><thead><tr><th>RowKey</th><th>timestamp</th><th>MAE</th><th>RMSE</th><th>R2</th></tr></thead><tbody>'];
    for(const it of logsCache){
        html.push(`<tr><td>${it.RowKey}</td><td>${it.timestamp}</td><td>${it.MAE}</td><td>${it.RMSE}</td><td>${it.R2}</td></tr>`);
    }
    html.push('</tbody></table>');
    document.getElementById('metrics').innerHTML = html.join('');
}

function exportLogsCSV(){
    if(!logsCache || logsCache.length===0){ alert('Sem logs para exportar'); return; }
    const cols = ['RowKey','timestamp','MAE','RMSE','R2'];
    const lines = [cols.join(',')];
    for(const it of logsCache){
        const row = cols.map(c => JSON.stringify(it[c] ?? '')).join(',');
        lines.push(row);
    }
    const blob = new Blob([lines.join('\\n')], {type:'text/csv;charset=utf-8;'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'logs_treinos.csv';
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
    log('Logs exportados (CSV).');
}

async function showPredictionsPreview(){
    try{
        const res = await fetch(`${API_BASE}/download/predictions`);
        if(!res.ok){ log('Nenhuma previsão disponível.'); return; }
        const txt = await res.text();
        const lines = txt.trim().split('\\n').slice(0, 11).join('\\n');
        document.getElementById('predPreview').innerText = lines;
    }catch(e){
        log('Erro preview: ' + e);
    }
}

log('Frontend pronto. API base: ' + API_BASE);
</script>
</body>
</html>
"""

HTML_DASHBOARD = HTML_TEMPLATE.replace("__API_URL__", API_URL)

# Rota Raiz para servir o HTML
@app.get("/", response_class=HTMLResponse)
async def serve_frontend_embedded():
    # Retorna o HTML_DASHBOARD (o seu frontend)
    return HTMLResponse(content=HTML_DASHBOARD, status_code=200)


# ============================================================
# 4. CONFIGURAÇÃO DO AZURE STORAGE (BLOB + TABLE)
# ============================================================

import os


ACCOUNT_NAME = os.getenv("AZURE_ACCOUNT_NAME")
ACCOUNT_KEY = os.getenv("AZURE_ACCOUNT_KEY")
AZURE_CONTAINER = "meucontainer"
TABLE_NAME = "Treinos"


# Removendo a lógica desnecessária de AZURE_CONNECTION_STRING e apenas verificando
# se as variáveis necessárias para a Connection String existem.
if not ACCOUNT_NAME or not ACCOUNT_KEY:
    # Este erro será disparado se as variáveis do ACA não estiverem sendo injetadas corretamente
    raise RuntimeError(
        "Azure credentials not found. Certifique-se de que as variáveis "
        "AZURE_ACCOUNT_NAME e AZURE_ACCOUNT_KEY estão configuradas como SEGREDOS."
    )

connection_string = (
    f"DefaultEndpointsProtocol=https;"
    f"AccountName={ACCOUNT_NAME};"
    f"AccountKey={ACCOUNT_KEY};"
    f"EndpointSuffix=core.windows.net"
)

# BLOB STORAGE
# A falha pode ocorrer aqui se a connection_string for inválida
try:
    blob_service = BlobServiceClient.from_connection_string(connection_string)
    container = blob_service.get_container_client(AZURE_CONTAINER)
except Exception as e:
    raise RuntimeError(f"Falha crítica ao conectar ao Blob Storage: {e}")

# TABLE STORAGE (BANCO GRATUITO)
try:
    table_service = TableServiceClient.from_connection_string(connection_string)
    # Tentar criar a tabela, se falhar, apenas obtê-la
    table_client = table_service.create_table_if_not_exists(TABLE_NAME)
except Exception as e:
    raise RuntimeError(f"Falha crítica ao conectar ao Table Storage: {e}")


def registrar_treino(mae, rmse, r2):
    entity = {
        "PartitionKey": "Treinos",
        "RowKey": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        "MAE": float(mae),
        "RMSE": float(float(rmse)),
        "R2": float(r2)
    }
    # ESTA CHAMADA É QUE DEVE ESTAR FALHANDO NO 500
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
    
    # Esta linha faz upload para o Azure Blob Storage, que pode ser o ponto de falha 500.
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