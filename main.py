# ============================================================
# main.py - API de Regressão Linear Remota (Azure Blob + Table Storage)
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
import traceback
import typing

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from fastapi.middleware.cors import CORSMiddleware

# --------------------
# CONFIGURAÇÕES E VARIÁVEIS
# --------------------
AZURE_ACCOUNT_NAME = os.getenv("AZURE_ACCOUNT_NAME")
AZURE_ACCOUNT_KEY = os.getenv("AZURE_ACCOUNT_KEY")
AZURE_CONTAINER = os.getenv("AZURE_CONTAINER", "meucontainer")
TABLE_NAME = os.getenv("TABLE_NAME", "Treinos")
PREDICTIONS_TABLE_NAME = os.getenv("PREDICTIONS_TABLE_NAME", "Predicoes")

if not AZURE_ACCOUNT_NAME or not AZURE_ACCOUNT_KEY:
    raise RuntimeError("Defina AZURE_ACCOUNT_NAME e AZURE_ACCOUNT_KEY como secrets.")

connection_string = (
    f"DefaultEndpointsProtocol=https;"
    f"AccountName={AZURE_ACCOUNT_NAME};"
    f"AccountKey={AZURE_ACCOUNT_KEY};"
    f"EndpointSuffix=core.windows.net"
)

# --------------------
# INICIALIZAÇÃO DO BLOB E TABLE STORAGE
# --------------------
blob_service = BlobServiceClient.from_connection_string(connection_string)
blob_container = blob_service.get_container_client(AZURE_CONTAINER)
try:
    blob_container.create_container()
except Exception:
    pass

table_service = TableServiceClient.from_connection_string(connection_string)
table_client = table_service.create_table_if_not_exists(TABLE_NAME)
predictions_table_client = table_service.create_table_if_not_exists(PREDICTIONS_TABLE_NAME)

# --------------------
# HELPERS
# --------------------
def safe_float(x):
    try:
        return float(x)
    except Exception:
        try:
            return float(np.asarray(x).item())
        except Exception:
            return None

def upload_to_blob(blob_name: str, data: bytes):
    try:
        blob_container.upload_blob(name=blob_name, data=data, overwrite=True)
    except Exception as e:
        raise RuntimeError(f"Erro upload blob {blob_name}: {e}")

def download_from_blob(blob_name: str) -> bytes:
    try:
        blob = blob_container.get_blob_client(blob_name)
        downloader = blob.download_blob()
        return downloader.readall()
    except Exception as e:
        raise RuntimeError(f"Erro download blob {blob_name}: {e}")

def delete_blob(blob_name: str):
    try:
        blob_container.get_blob_client(blob_name).delete_blob()
    except Exception:
        pass

def list_blobs_with_prefix(prefix: str) -> typing.List[str]:
    return [b.name for b in blob_container.list_blobs(name_starts_with=prefix)]

def latest_blob_by_prefix(prefix: str) -> typing.Optional[str]:
    blobs = list(blob_container.list_blobs(name_starts_with=prefix))
    if not blobs:
        return None
    blobs_sorted = sorted(blobs, key=lambda b: b.last_modified or datetime.min, reverse=True)
    return blobs_sorted[0].name

def build_lags(df, lags=5, target="time"):
    # Se os lags já estiverem na tabela, usa direto
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

def registrar_treino(mae, rmse, r2, extra: dict = None):
    entity = {
        "PartitionKey": "Treinos",
        "RowKey": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        "MAE": safe_float(mae),
        "RMSE": safe_float(rmse),
        "R2": safe_float(r2)
    }
    if extra:
        # adiciona campos extras se houver (ex: lags, cv_splits)
        for k, v in extra.items():
            try:
                entity[k] = str(v)
            except Exception:
                entity[k] = "__UNSERIALIZABLE__"
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

# --------------------
# FASTAPI
# --------------------
app = FastAPI(title="ML Remote API with Azure Storage")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse(content=HTML_DASHBOARD, status_code=200)

# --------------------
# Upload
# --------------------
@app.post("/upload/train")
async def upload_train(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    upload_to_blob("train_upload.csv", contents)
    return {"status": "ok", "rows": len(df), "columns": list(df.columns)}

@app.post("/upload/test")
async def upload_test(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    upload_to_blob("test_upload.csv", contents)
    return {"status": "ok", "rows": len(df), "columns": list(df.columns)}

# --------------------
# Treino
# --------------------
@app.post("/train")
async def train_model(lags: int = Form(5), cv_splits: int = Form(5)):
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
    # Treina final
    model = LinearRegression()
    model.fit(X_scaled, y)
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    model_blob_name = f"model_{ts}.joblib"
    scaler_blob_name = f"scaler_{ts}.joblib"
    b = io.BytesIO(); joblib.dump(model, b); upload_to_blob(model_blob_name, b.getvalue())
    b2 = io.BytesIO(); joblib.dump(scaler, b2); upload_to_blob(scaler_blob_name, b2.getvalue())

    metrics = {"MAE": float(np.mean(maes)), "RMSE": float(np.mean(rmses)), "R2": float(np.mean(r2s))}
    # salva métricas como CSV no Blob com timestamp
    metrics_df = pd.DataFrame([metrics])
    bm = io.BytesIO()
    metrics_df.to_csv(bm, index=False)
    metrics_blob_name = f"train_results_{ts}.csv"
    upload_to_blob(metrics_blob_name, bm.getvalue())

    registrar_treino(metrics["MAE"], metrics["RMSE"], metrics["R2"], extra={"lags": lags, "cv_splits": cv_splits, "metrics_blob": metrics_blob_name})

    return {"status": "trained", "metrics": metrics, "model_blob": model_blob_name, "metrics_blob": metrics_blob_name}

# --------------------
# Predict
# --------------------
@app.post("/predict")
async def predict(lags: int = Form(5)):
    # pega o model mais recente
    model_blob = latest_blob_by_prefix("model_")
    if not model_blob:
        raise HTTPException(status_code=404, detail="Modelo não encontrado")
    scaler_blob = latest_blob_by_prefix("scaler_")
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

    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    pred_blob_name = f"predictions_{ts}.csv"
    # salva CSV com timestamp
    b = io.BytesIO(); out.to_csv(b, index=False); upload_to_blob(pred_blob_name, b.getvalue())
    # também salva/atualiza um alias 'predictions.csv' para facilitar o download direto
    upload_to_blob("predictions.csv", b.getvalue())

    training_id = ts  # usa timestamp do run como id único
    for i, row in out.iterrows():
        try:
            registrar_predicao(training_id, X.iloc[i].to_dict(), float(row["predicted"]))
        except Exception:
            # se falhar registrar, continua mas loga
            print("Erro registrar_predicao:", traceback.format_exc())

    return {"status": "ok", "n": len(preds), "predictions_blob": pred_blob_name}

@app.get("/download/predictions")
async def download_predictions():
    # tenta pegar o latest predictions_{ts}.csv, se não houver, usa predictions.csv
    latest_pred = latest_blob_by_prefix("predictions_")
    blob_to_download = latest_pred or ("predictions.csv" if "predictions.csv" in list_blobs_with_prefix("") else None)
    if not blob_to_download:
        raise HTTPException(status_code=404, detail="Nenhum arquivo de previsões encontrado")
    data = download_from_blob(blob_to_download)
    path = "/tmp/predictions.csv"
    with open(path, "wb") as f: f.write(data)
    return FileResponse(path, filename=os.path.basename(blob_to_download))

@app.get("/predictions/table")
async def predictions_table():
    try:
        items = list(predictions_table_client.list_entities(results_per_page=50))
        predictions_list = []
        for item in items:
            # Tenta converter timestamp para isoformat; se falhar, usa string ou None
            try:
                ts = item.timestamp.isoformat() if hasattr(item.timestamp, 'isoformat') else str(item.timestamp)
            except Exception:
                ts = str(item.timestamp)

            # PredictedValue seguro
            predicted = safe_float(getattr(item, "PredictedValue", None))

            # Input lags seguros
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

        # Ordena pelo timestamp mais recente
        predictions_list.sort(key=lambda x: x["timestamp"], reverse=True)
        return {"predictions_table": predictions_list}

    except Exception as e:
        # Log do traceback no servidor
        print("Erro /predictions/table:", traceback.format_exc())
        # Retorna JSON sempre válido com mensagem de erro
        return {"predictions_table": [], "error": str(e)}

@app.get("/logs")
async def logs():
    items = list(table_client.list_entities())
    return {"logs": items}

# endpoint stub para logs do Cosmos (referenciado pelo frontend)
@app.get("/logs/cosmos")
async def logs_cosmos():
    # caso queira integrar com CosmosDB, substitua este stub
    return {"logs_cosmos": []}

@app.get("/metrics/last")
async def last_metrics():
    itens = list(table_client.list_entities())
    if not itens: 
        return {"message": "Nenhum treino encontrado"}
    itens.sort(key=lambda x: x["timestamp"], reverse=True)
    return itens[0]

@app.post("/reset")
async def reset():
    # deleta blobs comuns e todos os modelos/predictions/metrics
    prefixes = ["train_upload.csv", "test_upload.csv", "predictions.csv"]
    # deleta blobs com prefixos model_, scaler_, predictions_, train_results_
    for b in blob_container.list_blobs():
        name = b.name
        if (name.startswith("model_") or name.startswith("scaler_") or name.startswith("predictions_") or name.startswith("train_results_") or name in prefixes):
            try:
                blob_container.get_blob_client(name).delete_blob()
            except Exception:
                pass
    # (opcional) limpar tabelas
    try:
        # limpa tabela Treinos
        for ent in table_client.list_entities():
            try:
                table_client.delete_entity(partition_key=ent["PartitionKey"], row_key=ent["RowKey"])
            except Exception:
                pass
        # limpa tabela Predicoes
        for ent in predictions_table_client.list_entities():
            try:
                predictions_table_client.delete_entity(partition_key=ent["PartitionKey"], row_key=ent["RowKey"])
            except Exception:
                pass
    except Exception:
        pass

    return {"status": "reset"}

# ============================================================
# 3. FRONTEND EMBUTIDO E ROTA RAIZ (CÓDIGO EMBUTIDO)
# ============================================================

# ATENÇÃO: SUBSTITUA ESTE VALOR pela URL completa do seu Container App (se for diferente)
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
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
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
        <h3>Preview das previsões</h3>
        <div id="predPreview">Nenhuma previsão gerada ainda.</div>
    </div>

    <div class="box">
        <h3>Gráfico - Previsões vs Valor Real</h3>
        <canvas id="predictionChart" height="120"></canvas>
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
            headers: {
                "Content-Type": "application/x-www-form-urlencoded"
            },
            body: new URLSearchParams({ lags: 5, cv_splits: 5 })
        });

        const text = await res.text();
        log('Resposta bruta: ' + text);

        const j = JSON.parse(text);
        log('Treino finalizado: ' + JSON.stringify(j));
        document.getElementById('trainResult').innerText = JSON.stringify(j);
        getLastMetrics();
    }
    catch (e) {
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
    await renderPredictionChart();
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

async function getPredictionsTable(){
    try{
        const res = await fetch(`${API_BASE}/predictions/table`);
        const j = await res.json();
        const preds = j.predictions_table || [];
        log('Predições Table recebidas: ' + preds.length);

        const html = ['<table><thead><tr><th>ID</th><th>Training ID</th><th>Timestamp</th><th>Predicted</th><th>Lags</th></tr></thead><tbody>'];
        for(const it of preds){
            const lags = Object.entries(it.InputLags || {}).map(([k,v]) => `${k}:${v}`).join(', ');
            html.push(`<tr><td>${it.RowKey}</td><td>${it.PartitionKey}</td><td>${it.timestamp}</td><td>${it.PredictedValue}</td><td>${lags}</td></tr>`);
        }
        html.push('</tbody></table>');
        document.getElementById('metrics').innerHTML = '<h4>Predições Table Storage</h4>' + html.join('');
    }catch(e){
        log('Erro ao buscar predições Table: ' + e);
        document.getElementById('metrics').innerText = 'Erro ao buscar predições Table: ' + e.message;
    }
}

let predictionChartInstance = null;

async function renderPredictionChart(){
    try{
        const res = await fetch(`${API_BASE}/download/predictions`);
        if(!res.ok){
            log('Sem CSV de previsões para gráfico.');
            return;
        }

        const csv = await res.text();
        const lines = csv.trim().split('\\n');
        if(lines.length < 2){
            log('CSV insuficiente para plotagem.');
            return;
        }

        const headers = lines[0].split(',');
        const predictedIndex = headers.indexOf('predicted');
        const actualIndex = headers.indexOf('actual');

        const labels = [];
        const predictedData = [];
        const actualData = [];

        lines.slice(1).forEach((line, i) => {
            const cols = line.split(',');
            labels.push(i + 1);
            if (predictedIndex >= 0) {
                predictedData.push(parseFloat(cols[predictedIndex]));
            }
            if (actualIndex >= 0) {
                actualData.push(parseFloat(cols[actualIndex]));
            }
        });

        const canvas = document.getElementById('predictionChart');
        if (!canvas){
            log('Canvas de gráfico não encontrado no HTML.');
            return;
        }
        const ctx = canvas.getContext('2d');

        if (predictionChartInstance) {
            predictionChartInstance.destroy();
        }

        predictionChartInstance = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Valor Previsto',
                        data: predictedData,
                        borderColor: '#2563eb',
                        backgroundColor: 'rgba(37,99,235,0.15)',
                        tension: 0.3
                    },
                    {
                        label: 'Valor Real',
                        data: actualData,
                        borderColor: '#16a34a',
                        backgroundColor: 'rgba(22,163,74,0.15)',
                        tension: 0.3
                    }
                ]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { display: true }
                },
                scales: {
                    x: { title: { display: true, text: 'Registro' } },
                    y: { title: { display: true, text: 'Valor' } }
                }
            }
        });

        log('Gráfico de previsões renderizado.');
    }catch(e){
        log('Erro ao gerar gráfico: ' + e);
    }
}

log('Frontend pronto. API base: ' + API_BASE);

document.addEventListener("DOMContentLoaded", () => {
    log("Dashboard carregado.");
});
</script>
</body>
</html>
"""

HTML_DASHBOARD = HTML_TEMPLATE.replace("__API_URL__", API_URL)

# Rota Raiz para servir o HTML
@app.get("/", response_class=HTMLResponse)
async def serve_frontend_embedded():
    return HTMLResponse(content=HTML_DASHBOARD, status_code=200)

# ============================================================
# 4. CONFIGURAÇÃO DO AZURE STORAGE (BLOB + TABLE)
# ============================================================