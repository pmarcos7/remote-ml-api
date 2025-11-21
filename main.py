# ============================================================
# main.py - API de Regressão Linear Remota com Azure Storage + Banco Gratuito
# ============================================================

# ============================================================
# main.py - CORRIGIDO - API de Regressão Linear Remota
# Correções importantes:
# - Evita sobrescrever variáveis (blob_container vs cosmos_container)
# - Verifica presence de env vars e falha com mensagens claras
# - Trata tipos numpy antes de gravar no Azure Table
# - Uso seguro do Cosmos DB (opcional) — só ativa se COSMOS_URI/COSMOS_KEY existirem
# - Tratamento de erros de I/O com mensagens e logs
# - Mantém comportamento original (upload, train, predict, download)
# ============================================================

# main.py - API de Regressão Linear Remota (corrigido, tudo em um bloco)

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

# Cosmos (opcional)
try:
    from azure.cosmos import CosmosClient
except Exception:
    CosmosClient = None  # se não estiver instalado, continuamos sem cosmos

# --------------------
# CONFIGURAÇÕES E VARIÁVEIS
# --------------------
AZURE_ACCOUNT_NAME = os.getenv("AZURE_ACCOUNT_NAME")
AZURE_ACCOUNT_KEY = os.getenv("AZURE_ACCOUNT_KEY")
AZURE_CONTAINER = os.getenv("AZURE_CONTAINER", "meucontainer")
TABLE_NAME = os.getenv("TABLE_NAME", "Treinos")

# Cosmos (opcional)
COSMOS_URI = os.getenv("COSMOS_URI")
COSMOS_KEY = os.getenv("COSMOS_KEY")
COSMOS_DB_NAME = os.getenv("COSMOS_DB_NAME", "remote-ml")
COSMOS_TRAININGS_CONTAINER = os.getenv("COSMOS_TRAININGS_CONTAINER", "training_runs")
COSMOS_PREDICTIONS_CONTAINER = os.getenv("COSMOS_PREDICTIONS_CONTAINER", "predictions")

# Verificações iniciais mínimas para Blob/Table
if not AZURE_ACCOUNT_NAME or not AZURE_ACCOUNT_KEY:
    raise RuntimeError(
        "Azure credentials not found. Defina AZURE_ACCOUNT_NAME e AZURE_ACCOUNT_KEY como secrets."
    )

connection_string = (
    f"DefaultEndpointsProtocol=https;"
    f"AccountName={AZURE_ACCOUNT_NAME};"
    f"AccountKey={AZURE_ACCOUNT_KEY};"
    f"EndpointSuffix=core.windows.net"
)

# --------------------
# INICIALIZAÇÃO DOS CLIENTES AZURE
# --------------------
try:
    blob_service = BlobServiceClient.from_connection_string(connection_string)
    blob_container = blob_service.get_container_client(AZURE_CONTAINER)
    # cria container se não existir (ignore erro se já existe)
    try:
        blob_container.create_container()
    except Exception:
        pass
except Exception as e:
    raise RuntimeError(f"Falha ao conectar ao Blob Storage: {e}")

try:
    table_service = TableServiceClient.from_connection_string(connection_string)
    table_client = table_service.create_table_if_not_exists(TABLE_NAME)
except Exception as e:
    raise RuntimeError(f"Falha ao conectar ao Table Storage: {e}")

# --------------------
# INICIALIZAÇÃO DO COSMOS (opcional)
# --------------------
cosmos_enabled = False
cosmos_client = None
cosmos_db = None
cosmos_trainings = None
cosmos_predictions = None

if COSMOS_URI and COSMOS_KEY and CosmosClient is not None:
    try:
        cosmos_client = CosmosClient(COSMOS_URI, credential=COSMOS_KEY)
        cosmos_db = cosmos_client.create_database_if_not_exists(id=COSMOS_DB_NAME)
        cosmos_trainings = cosmos_db.create_container_if_not_exists(
            id=COSMOS_TRAININGS_CONTAINER,
            partition_key="/id"  # partition key simples para runs
        )
        cosmos_predictions = cosmos_db.create_container_if_not_exists(
            id=COSMOS_PREDICTIONS_CONTAINER,
            partition_key="/training_id"
        )
        cosmos_enabled = True
    except Exception as e:
        # não falhar o app inteiro por causa do Cosmos, apenas logar
        print("Aviso: não foi possível inicializar Cosmos DB:", e)
        cosmos_enabled = False

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

def write_log_cosmos(log_type, message, data=None):
    if not cosmos_enabled:
        return
    try:
        item = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "type": log_type,
            "message": message,
            "data": data or {}
        }
        if "type" not in item or not item["type"]:
            item["type"] = "info"
        cosmos_trainings.upsert_item(item)  # uso train container para logs simples
    except Exception as e:
        print("Erro ao gravar log no Cosmos:", e)

def registrar_treino(mae, rmse, r2):
    mae_v = safe_float(mae)
    rmse_v = safe_float(rmse)
    r2_v = safe_float(r2)
    entity = {
        "PartitionKey": "Treinos",
        "RowKey": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        "MAE": mae_v,
        "RMSE": rmse_v,
        "R2": r2_v
    }
    try:
        table_client.create_entity(entity)
        write_log_cosmos("treino", "Treino registrado na Table Storage", {"MAE": mae_v, "RMSE": rmse_v, "R2": r2_v})
    except Exception as e:
        print("Erro ao registrar treino na Table Storage:", e)
        raise

# --------------------
# HELPERS BLOB
# --------------------
def upload_to_blob(blob_name: str, data: bytes):
    try:
        blob_container.upload_blob(name=blob_name, data=data, overwrite=True)
    except Exception as e:
        print(f"Erro upload blob {blob_name}: {e}")
        raise

def download_from_blob(blob_name: str) -> bytes:
    try:
        blob = blob_container.get_blob_client(blob_name)
        downloader = blob.download_blob()
        return downloader.readall()
    except Exception as e:
        print(f"Erro download blob {blob_name}: {e}")
        raise

def delete_blob(blob_name: str):
    try:
        blob_container.get_blob_client(blob_name).delete_blob()
    except Exception:
        pass

# --------------------
# build_lags
# --------------------
def build_lags(df, lags=5, target="time"):
    # se já tem colunas lag1..lagN, usa direto
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

# --------------------
# Cosmos helpers específicos (training_runs / predictions)
# --------------------
def save_training_run_to_cosmos(summary, model_params, metrics, model_blob_name, scaler_blob_name=None):
    if not cosmos_enabled:
        return None
    training_id = str(uuid.uuid4())
    item = {
        "id": training_id,
        "timestamp": datetime.utcnow().isoformat(),
        "inputSummary": summary,
        "modelParams": model_params,
        "metrics": metrics,
        "model_blob": model_blob_name,
        "scaler_blob": scaler_blob_name
    }
    try:
        cosmos_trainings.create_item(item)
    except Exception as e:
        print("Erro ao salvar training_run no Cosmos:", e)
        raise
    return training_id

def save_prediction_to_cosmos(training_id, input_row, output_value):
    if not cosmos_enabled:
        return None
    pred = {
        "id": str(uuid.uuid4()),
        "training_id": training_id,
        "timestamp": datetime.utcnow().isoformat(),
        "input": input_row,
        "output": float(output_value)
    }
    try:
        cosmos_predictions.create_item(pred)
    except Exception as e:
        print("Erro ao salvar prediction no Cosmos:", e)
        raise
    return pred["id"]

# --------------------
# FASTAPI APP
# --------------------
app = FastAPI(title="ML Remote API with Azure Storage + Cosmos + Table Storage")

origins = [
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    "http://localhost:9000",
    "http://127.0.0.1:9000",
    "null",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# pequeno HTML para teste
API_URL = os.getenv("API_URL", "/")
HTML_DASHBOARD = f"""
<!doctype html><html><head><meta charset="utf-8"><title>Remote ML</title></head><body>
<h2>Remote ML API</h2>
<p>Endpoints: /upload/train, /train, /upload/test, /predict, /download/predictions, /logs, /metrics/last</p>
<p>API base: {API_URL}</p>
</body></html>
"""

@app.get("/", response_class=HTMLResponse)
async def serve_frontend_embedded():
    return HTMLResponse(content=HTML_DASHBOARD, status_code=200)

# --------------------
# Endpoints: uploads
# --------------------
@app.post("/upload/train")
async def upload_train(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CSV inválido: {e}")
    try:
        upload_to_blob("train_upload.csv", contents)
    except Exception as e:
        print("Erro ao enviar treino para blob:", e)
        raise HTTPException(status_code=500, detail="Falha ao salvar arquivo de treino no Blob Storage")
    return {"status": "ok", "rows": len(df), "columns": list(df.columns)}

@app.post("/upload/test")
async def upload_test(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CSV inválido: {e}")
    try:
        upload_to_blob("test_upload.csv", contents)
    except Exception as e:
        print("Erro ao enviar teste para blob:", e)
        raise HTTPException(status_code=500, detail="Falha ao salvar arquivo de teste no Blob Storage")
    return {"status": "ok", "rows": len(df), "columns": list(df.columns)}

# --------------------
# Train endpoint
# --------------------
@app.post("/train")
async def train_model(lags: int = Form(5), cv_splits: int = Form(5)):
    try:
        data = download_from_blob("train_upload.csv")
        df = pd.read_csv(io.BytesIO(data))

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
            rmses.append(np.sqrt(mean_squared_error(yval, preds)))
            r2s.append(r2_score(yval, preds))

        model = LinearRegression()
        model.fit(X_scaled, y)

        # salvar model + scaler no blob com timestamp
        ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        model_blob_name = f"model_{ts}.joblib"
        scaler_blob_name = f"scaler_{ts}.joblib"

        b = io.BytesIO()
        joblib.dump(model, b)
        upload_to_blob(model_blob_name, b.getvalue())

        b2 = io.BytesIO()
        joblib.dump(scaler, b2)
        upload_to_blob(scaler_blob_name, b2.getvalue())

        # preparar resumo e params
        summary = {
            "rows": int(len(df)),
            "columns": list(df.columns),
            "target": "time",
            "count": int(len(y)),
            "min": float(y.min()),
            "max": float(y.max()),
            "mean": float(y.mean())
        }
        model_params = {
            "coef": [float(c) for c in model.coef_.tolist()],
            "intercept": float(model.intercept_)
        }
        metrics = {
            "MAE": float(np.mean(maes)),
            "RMSE": float(np.mean(rmses)),
            "R2": float(np.mean(r2s))
        }

        # salvar run no cosmos (opcional)
        training_id = None
        try:
            training_id = save_training_run_to_cosmos(summary, model_params, metrics, model_blob_name, scaler_blob_name)
        except Exception as e:
            print("Aviso: falha ao salvar training_run no Cosmos:", e)

        # registrar em Table Storage (log simples)
        try:
            registrar_treino(metrics["MAE"], metrics["RMSE"], metrics["R2"])
        except Exception as e:
            print("Aviso: falha ao registrar na Table Storage:", e)

        return {"status": "trained", "metrics": metrics, "training_id": training_id, "model_blob": model_blob_name}

    except HTTPException:
        raise
    except Exception as e:
        print("ERRO NO TREINO:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erro no processo de treino: {e}")

# --------------------
# Predict endpoint
# --------------------
@app.post("/predict")
async def predict(training_id: str = Form(None), lags: int = Form(5)):
    try:
        # decidir qual model/scaler usar
        model_blob = None
        scaler_blob = None
        if training_id and cosmos_enabled:
            # buscar run no cosmos
            query = f"SELECT * FROM c WHERE c.id = '{training_id}'"
            docs = list(cosmos_trainings.query_items(query=query, enable_cross_partition_query=True))
            if not docs:
                raise HTTPException(status_code=404, detail="training_id não encontrado no Cosmos")
            doc = docs[0]
            model_blob = doc.get("model_blob")
            scaler_blob = doc.get("scaler_blob")
        else:
            # fallback: pega o mais recente pelo nome (busca blob list)
            # tenta pegar "model_*.joblib" mais recente
            try:
                blobs = list(blob_container.list_blobs(name_starts_with="model_"))
                if blobs:
                    # escolher o mais recente por last_modified
                    blobs_sorted = sorted(blobs, key=lambda b: b.last_modified or datetime.min, reverse=True)
                    model_blob = blobs_sorted[0].name
                blobs_s = list(blob_container.list_blobs(name_starts_with="scaler_"))
                if blobs_s:
                    blobs_s_sorted = sorted(blobs_s, key=lambda b: b.last_modified or datetime.min, reverse=True)
                    scaler_blob = blobs_s_sorted[0].name
            except Exception:
                # se falhar, assume nomes padrão
                model_blob = model_blob or "model.joblib"
                scaler_blob = scaler_blob or "scaler.joblib"

        if not model_blob:
            raise HTTPException(status_code=404, detail="Modelo não encontrado no Blob Storage")

        model = joblib.load(io.BytesIO(download_from_blob(model_blob)))
        scaler = joblib.load(io.BytesIO(download_from_blob(scaler_blob))) if scaler_blob else None

        # ler test csv do blob
        data = download_from_blob("test_upload.csv")
        df = pd.read_csv(io.BytesIO(data))

        has_labels = "time" in df.columns
        X, y = build_lags(df, lags=lags)

        X_scaled = scaler.transform(X) if scaler is not None else X
        preds = model.predict(X_scaled)

        out = pd.DataFrame({"predicted": preds})
        if has_labels:
            out["actual"] = y.values[:len(preds)]
            out["error"] = out["actual"] - out["predicted"]

        # salvar CSV de previsões
        b = io.BytesIO()
        out.to_csv(b, index=False)
        upload_to_blob("predictions.csv", b.getvalue())

        # salvar cada predição no Cosmos
        saved = []
        if cosmos_enabled:
            for i, row in out.iterrows():
                input_row = X.iloc[i].to_dict()
                try:
                    pid = save_prediction_to_cosmos(training_id or "unknown", input_row, float(row["predicted"]))
                    saved.append(pid)
                except Exception as e:
                    print("Aviso: falha ao salvar prediction no Cosmos:", e)

        # opcional: gravar log no cosmos
        write_log_cosmos("predict", "Predição executada", {"n": len(preds), "training_id": training_id})

        return {"status": "ok", "n": len(preds), "saved_ids": saved}

    except HTTPException:
        raise
    except Exception as e:
        print("Erro no predict:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erro no predict: {e}")

# --------------------
# Download predictions
# --------------------
@app.get("/download/predictions")
async def download_predictions():
    try:
        data = download_from_blob("predictions.csv")
    except Exception:
        raise HTTPException(status_code=404, detail="Nenhuma previsão disponível")
    path = "/tmp/predictions.csv"
    with open(path, "wb") as f:
        f.write(data)
    return FileResponse(path, filename="predictions.csv")

# --------------------
# Logs e métricas
# --------------------
@app.get("/logs")
async def logs():
    try:
        items = list(table_client.list_entities())
        return {"logs": items}
    except Exception as e:
        print("Erro ao listar treinos:", e)
        raise HTTPException(status_code=500, detail="Falha ao listar logs")

@app.get("/metrics/last")
async def last_metrics():
    try:
        itens = list(table_client.list_entities())
        if not itens:
            return {"message": "Nenhum treino encontrado"}
        itens.sort(key=lambda x: x["timestamp"], reverse=True)
        return itens[0]
    except Exception as e:
        print("Erro ao buscar métricas:", e)
        raise HTTPException(status_code=500, detail="Falha ao obter última métrica")

# --------------------
# Reset
# --------------------
@app.post("/reset")
async def reset():
    for f in ["train_upload.csv", "test_upload.csv", "model.joblib", "scaler.joblib", "predictions.csv"]:
        try:
            delete_blob(f)
        except Exception:
            pass
    return {"status": "reset"}


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

        const text = await res.text();  // <-- pega qualquer coisa, até HTML
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