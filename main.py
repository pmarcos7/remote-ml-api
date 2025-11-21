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
import traceback

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from fastapi.middleware.cors import CORSMiddleware

# ------------------------------------------------------
# CONFIG
# ------------------------------------------------------
AZURE_ACCOUNT_NAME = os.getenv("AZURE_ACCOUNT_NAME")
AZURE_ACCOUNT_KEY = os.getenv("AZURE_ACCOUNT_KEY")
AZURE_CONTAINER = os.getenv("AZURE_CONTAINER", "meucontainer")
TABLE_NAME = os.getenv("TABLE_NAME", "Treinos")
PREDICTIONS_TABLE_NAME = os.getenv("PREDICTIONS_TABLE_NAME", "Predicoes")

if not AZURE_ACCOUNT_NAME or not AZURE_ACCOUNT_KEY:
    raise RuntimeError("Azure credentials missing.")

connection_string = (
    f"DefaultEndpointsProtocol=https;"
    f"AccountName={AZURE_ACCOUNT_NAME};"
    f"AccountKey={AZURE_ACCOUNT_KEY};"
    f"EndpointSuffix=core.windows.net"
)

# ------------------------------------------------------
# CLIENTES AZURE
# ------------------------------------------------------
blob_service = BlobServiceClient.from_connection_string(connection_string)
blob_container = blob_service.get_container_client(AZURE_CONTAINER)
try:
    blob_container.create_container()
except:
    pass

table_service = TableServiceClient.from_connection_string(connection_string)
table_client = table_service.create_table_if_not_exists(TABLE_NAME)
predictions_table_client = table_service.create_table_if_not_exists(PREDICTIONS_TABLE_NAME)

# ------------------------------------------------------
# HELPERS
# ------------------------------------------------------
def safe_float(x):
    try:
        return float(x)
    except:
        try:
            return float(np.asarray(x).item())
        except:
            return None

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

def registrar_predicao(training_id, input_row, predicted):
    entity = {
        "PartitionKey": training_id or "unknown",
        "RowKey": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        "PredictedValue": safe_float(predicted),
        **{f"lag{k}": safe_float(v) for k, v in input_row.items()}
    }
    predictions_table_client.create_entity(entity)

def upload_to_blob(name, data):
    blob_container.upload_blob(name=name, data=data, overwrite=True)

def download_from_blob(name):
    return blob_container.get_blob_client(name).download_blob().readall()

# ------------------------------------------------------
# build_lags
# ------------------------------------------------------
def build_lags(df, lags=5, target="time"):
    s = df[target].astype(float)
    data = {f"lag{i}": s.shift(i) for i in range(1, lags + 1)}
    data[target] = s
    new_df = pd.DataFrame(data).dropna().reset_index(drop=True)
    X = new_df[[f"lag{i}" for i in range(1, lags + 1)]]
    y = new_df[target]
    return X, y

# ------------------------------------------------------
# FASTAPI
# ------------------------------------------------------
app = FastAPI(title="Remote ML API — TABLE STORAGE ONLY")

origins = ["*", "null", "http://localhost:8080"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/", response_class=HTMLResponse)
async def home():
    return "<h2>API Machine Learning Remota</h2>"

# ------------------------------------------------------
# UPLOADS
# ------------------------------------------------------
@app.post("/upload/train")
async def upload_train(file: UploadFile = File(...)):
    data = await file.read()
    df = pd.read_csv(io.BytesIO(data))
    upload_to_blob("train_upload.csv", data)
    return {"status": "ok", "rows": len(df)}

@app.post("/upload/test")
async def upload_test(file: UploadFile = File(...)):
    data = await file.read()
    df = pd.read_csv(io.BytesIO(data))
    upload_to_blob("test_upload.csv", data)
    return {"status": "ok", "rows": len(df)}

# ------------------------------------------------------
# TREINAR MODELO
# ------------------------------------------------------
@app.post("/train")
async def train(lags: int = Form(5), cv_splits: int = Form(5)):

    df = pd.read_csv(io.BytesIO(download_from_blob("train_upload.csv")))
    X, y = build_lags(df, lags)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    tscv = TimeSeriesSplit(n_splits=cv_splits)
    maes, rmses, r2s = [], [], []

    for tr, val in tscv.split(X_scaled):
        Xtr, Xval = X_scaled[tr], X_scaled[val]
        ytr, yval = y.iloc[tr], y.iloc[val]

        model = LinearRegression()
        model.fit(Xtr, ytr)
        preds = model.predict(Xval)

        maes.append(mean_absolute_error(yval, preds))
        rmses.append(np.sqrt(mean_squared_error(yval, preds)))
        r2s.append(r2_score(yval, preds))

    model = LinearRegression()
    model.fit(X_scaled, y)

    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    mname = f"model_{ts}.joblib"
    sname = f"scaler_{ts}.joblib"

    b = io.BytesIO(); joblib.dump(model, b)
    upload_to_blob(mname, b.getvalue())

    b = io.BytesIO(); joblib.dump(scaler, b)
    upload_to_blob(sname, b.getvalue())

    metrics = {
        "MAE": float(np.mean(maes)),
        "RMSE": float(np.mean(rmses)),
        "R2": float(np.mean(r2s))
    }

    registrar_treino(metrics["MAE"], metrics["RMSE"], metrics["R2"])

    return {"status": "trained", "metrics": metrics}

# ------------------------------------------------------
# PREVER
# ------------------------------------------------------
@app.post("/predict")
async def predict(training_id: str = Form("default"), lags: int = Form(5)):

    # encontra modelo mais recente
    blobs = list(blob_container.list_blobs(name_starts_with="model_"))
    blobs_sorted = sorted(blobs, key=lambda b: b.last_modified, reverse=True)
    mname = blobs_sorted[0].name

    sblobs = list(blob_container.list_blobs(name_starts_with="scaler_"))
    s_sorted = sorted(sblobs, key=lambda b: b.last_modified, reverse=True)
    sname = s_sorted[0].name

    model = joblib.load(io.BytesIO(download_from_blob(mname)))
    scaler = joblib.load(io.BytesIO(download_from_blob(sname)))

    df = pd.read_csv(io.BytesIO(download_from_blob("test_upload.csv")))
    X, y = build_lags(df, lags)
    X_scaled = scaler.transform(X)

    preds = model.predict(X_scaled)

    out = pd.DataFrame({"predicted": preds})
    if "time" in df.columns:
        out["actual"] = y.values
        out["error"] = out["actual"] - out["predicted"]

    # salva CSV
    b = io.BytesIO()
    out.to_csv(b, index=False)
    upload_to_blob("predictions.csv", b.getvalue())

    # registra predições na tabela
    for i, row in out.iterrows():
        registrar_predicao(training_id, X.iloc[i].to_dict(), float(row["predicted"]))

    return {"status": "ok", "n": len(preds)}

# ------------------------------------------------------
# DOWNLOAD CSV
# ------------------------------------------------------
@app.get("/download/predictions")
async def download_predictions():
    data = download_from_blob("predictions.csv")
    temp = "/tmp/predictions.csv"
    with open(temp, "wb") as f:
        f.write(data)
    return FileResponse(temp, filename="predictions.csv")

# ------------------------------------------------------
# LISTAR LOGS DE TREINOS (TABLE)
# ------------------------------------------------------
@app.get("/logs")
async def logs():
    return {"logs": list(table_client.list_entities())}

# ------------------------------------------------------
# LISTAR PREDIÇÕES (TABLE)
# ------------------------------------------------------
@app.get("/predictions/table")
async def predicoes_table():
    return {"predicoes": list(predictions_table_client.list_entities())}



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
        <!-- UPLOAD E TREINO -->
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

        <!-- UPLOAD E PREDIÇÃO -->
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

        <!-- LOGS E MÉTRICAS -->
        <div class="col">
            <h3>3) Logs e Métricas</h3>
            <div style="display:flex;gap:8px;align-items:center;margin-bottom:8px">
                <button onclick="getLastMetrics()">Último treino (Table)</button>
                <button onclick="getLogs()">Logs Treino (Table)</button>
            </div>
            <div style="display:flex;gap:8px;align-items:center;margin-bottom:8px">
                <button onclick="getTablePredictions()">Predições (Table)</button>
            </div>
            <div id="metrics" style="margin-top:8px"></div>
        </div>
    </div>

    <div class="box">
        <h3>AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA</h3>
        <div id="predPreview">Nenhuma previsão gerada ainda.</div>
    </div>

    <div class="box">
        <h3>Console</h3>
        <pre id="console">Pronto.</pre>
    </div>

<script>
const API_BASE = "__API_URL__";

function log(msg){
    const c = document.getElementById('console');
    c.textContent = `${new Date().toISOString()} — ${msg}\n` + c.textContent;
}

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

async function train() {
    log('Iniciando treino...');
    try {
        const res = await fetch(`${API_BASE}/train`, {
            method: 'POST',
            headers: { "Content-Type": "application/x-www-form-urlencoded" },
            body: new URLSearchParams({ lags: 5, cv_splits: 5 })
        });
        const text = await res.text();
        log('Resposta bruta: ' + text);
        const j = JSON.parse(text);
        document.getElementById('trainResult').innerText = JSON.stringify(j);
        getLastMetrics();
    } catch (e) {
        log("ERRO no front: " + e);
    }
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

    const html = ['<table><thead><tr><th>RowKey</th><th>timestamp</th><th>MAE</th><th>RMSE</th><th>R2</th></tr></thead><tbody>'];

    for(const it of logsCache){
        html.push(`<tr>
            <td>${it.RowKey}</td>
            <td>${it.timestamp}</td>
            <td>${it.MAE}</td>
            <td>${it.RMSE}</td>
            <td>${it.R2}</td>
        </tr>`);
    }

    html.push('</tbody></table>');
    document.getElementById('metrics').innerHTML = '<h4>Treinos (Azure Table Storage)</h4>' + html.join('');
}

async function getTablePredictions(){
    try{
        const res = await fetch(`${API_BASE}/predictions/table`);
        const j = await res.json();
        const predictions = j.predictions_table || [];

        if (predictions.length === 0) {
            document.getElementById('metrics').innerHTML = '<h4>Predições (Azure Table Storage)</h4><p>Nenhuma predição encontrada.</p>';
            return;
        }

        const html = ['<table><thead><tr><th>RowKey</th><th>Valor Previsto</th></tr></thead><tbody>'];

        for(const it of predictions){
            html.push(`<tr>
                <td>${it.RowKey}</td>
                <td>${it.PredictedValue}</td>
            </tr>`);
        }

        html.push('</tbody></table>');
        document.getElementById('metrics').innerHTML = '<h4>Predições (Azure Table Storage)</h4>' + html.join('');
    }catch(e){
        log('Erro ao buscar predições Table: ' + e.message);
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