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
    # OBS: Em produção, você deve garantir que estas variáveis de ambiente estão definidas.
    # Exemplo simples para evitar erro fatal em desenvolvimento local:
    # AZURE_ACCOUNT_NAME = "devstoreaccount1"
    # AZURE_ACCOUNT_KEY = "Eby8vdM02xNOcqFlqUwJ7azLwWqXaGvFkQmtjWN7HBg9EOLePqZ9qW8c81o9r4tFkQmtjWN7HBg9EOLePqZ9qW8c81o9r4t"
    # if not AZURE_ACCOUNT_NAME or not AZURE_ACCOUNT_KEY:
    #     raise RuntimeError("Azure credentials missing.")
    pass

# Se estiver usando o Azurite/Emulador localmente, você pode usar uma string de conexão padrão.
# Caso contrário, mantenha a lógica original que usa as variáveis de ambiente.
try:
    connection_string = (
        f"DefaultEndpointsProtocol=https;"
        f"AccountName={AZURE_ACCOUNT_NAME};"
        f"AccountKey={AZURE_ACCOUNT_KEY};"
        f"EndpointSuffix=core.windows.net"
    )
except TypeError:
    # Caso as variáveis de ambiente não estejam carregadas (ex: ambiente local sem .env)
    print("Atenção: Variáveis de ambiente Azure não encontradas. Usando string de conexão de exemplo.")
    connection_string = "UseDevelopmentStorage=true"


# ------------------------------------------------------
# CLIENTES AZURE
# ------------------------------------------------------
# Bloco try-except para lidar com o ambiente de desenvolvimento/teste sem Azure keys
try:
    blob_service = BlobServiceClient.from_connection_string(connection_string)
    blob_container = blob_service.get_container_client(AZURE_CONTAINER)
    try:
        blob_container.create_container()
    except Exception as e:
        # print(f"Container já existe ou erro: {e}")
        pass

    table_service = TableServiceClient.from_connection_string(connection_string)
    table_client = table_service.create_table_if_not_exists(TABLE_NAME)
    predictions_table_client = table_service.create_table_if_not_exists(PREDICTIONS_TABLE_NAME)

except Exception as e:
    print(f"ERRO ao conectar ao Azure Storage: {e}. As rotas de upload/treino/previsão falharão.")
    # Clientes mock para evitar que o FastAPI não suba
    class MockClient:
        def __init__(self):
            print("AVISO: Usando MockClient. As funções de persistência não funcionarão.")
        def create_entity(self, entity): pass
        def list_entities(self): return []
        def upload_blob(self, name, data, overwrite): pass
        def download_blob(self): return io.BytesIO(b"").read()
        def get_blob_client(self, name): return self
        def readall(self): return io.BytesIO(b"").read()
        def list_blobs(self, name_starts_with): return []
        def create_container(self): pass
    
    blob_service = MockClient()
    blob_container = MockClient()
    table_client = MockClient()
    predictions_table_client = MockClient()


# ------------------------------------------------------
# HELPERS
# ------------------------------------------------------
def safe_float(x):
    try:
        # Tenta converter o item do numpy ou o valor para float
        if isinstance(x, np.generic):
            return float(x.item())
        return float(x)
    except:
        return None

def registrar_treino(mae, rmse, r2):
    try:
        entity = {
            "PartitionKey": "Treinos",
            "RowKey": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "MAE": safe_float(mae),
            "RMSE": safe_float(rmse),
            "R2": safe_float(r2)
        }
        table_client.create_entity(entity)
    except Exception as e:
        print(f"Erro ao registrar treino: {e}")

def registrar_predicao(training_id, input_row, predicted):
    try:
        entity = {
            "PartitionKey": training_id or "unknown",
            "RowKey": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "PredictedValue": safe_float(predicted),
            **{f"lag{k}": safe_float(v) for k, v in input_row.items()}
        }
        predictions_table_client.create_entity(entity)
    except Exception as e:
        print(f"Erro ao registrar predição: {e}")

def upload_to_blob(name, data):
    try:
        blob_container.upload_blob(name=name, data=data, overwrite=True)
    except Exception as e:
        print(f"Erro ao fazer upload do blob {name}: {e}")

def download_from_blob(name):
    try:
        return blob_container.get_blob_client(name).download_blob().readall()
    except Exception as e:
        print(f"Erro ao baixar blob {name}: {e}")
        # Retorna um CSV vazio para evitar erro de leitura
        return b"time\n1.0"


# ------------------------------------------------------
# build_lags
# ------------------------------------------------------
def build_lags(df, lags=5, target="time"):
    s = df[target].astype(float)
    # Garante que o índice de tempo (target) seja o último para ser o 'y'
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


# ------------------------------------------------------
# UPLOADS
# ------------------------------------------------------
@app.post("/upload/train")
async def upload_train(file: UploadFile = File(...)):
    data = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(data))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao ler CSV de treino: {e}")
        
    upload_to_blob("train_upload.csv", data)
    return {"status": "ok", "rows": len(df)}

@app.post("/upload/test")
async def upload_test(file: UploadFile = File(...)):
    data = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(data))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao ler CSV de teste: {e}")
        
    upload_to_blob("test_upload.csv", data)
    return {"status": "ok", "rows": len(df)}

# ------------------------------------------------------
# TREINAR MODELO
# ------------------------------------------------------
@app.post("/train")
async def train(lags: int = Form(5), cv_splits: int = Form(5)):
    try:
        df_data = download_from_blob("train_upload.csv")
        df = pd.read_csv(io.BytesIO(df_data))
        
        if df.empty or 'time' not in df.columns:
            raise ValueError("CSV vazio ou coluna 'time' ausente.")

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

        # Treina o modelo final com todos os dados
        model = LinearRegression()
        model.fit(X_scaled, y)

        ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        mname = f"model_{ts}.joblib"
        sname = f"scaler_{ts}.joblib"

        # Serializa e salva o modelo
        b = io.BytesIO(); joblib.dump(model, b)
        upload_to_blob(mname, b.getvalue())

        # Serializa e salva o scaler
        b = io.BytesIO(); joblib.dump(scaler, b)
        upload_to_blob(sname, b.getvalue())

        metrics = {
            "MAE": float(np.mean(maes)),
            "RMSE": float(np.mean(rmses)),
            "R2": float(np.mean(r2s))
        }

        registrar_treino(metrics["MAE"], metrics["RMSE"], metrics["R2"])

        return {"status": "trained", "model_name": mname, "metrics": metrics}
    
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erro no treinamento: {str(e)}")


# ------------------------------------------------------
# PREVER
# ------------------------------------------------------
@app.post("/predict")
async def predict(training_id: str = Form("default"), lags: int = Form(5)):
    try:
        # encontra modelo mais recente
        blobs = list(blob_container.list_blobs(name_starts_with="model_"))
        blobs_sorted = sorted(blobs, key=lambda b: b.last_modified, reverse=True)
        if not blobs_sorted:
            raise FileNotFoundError("Nenhum modelo treinado encontrado.")
        mname = blobs_sorted[0].name

        sblobs = list(blob_container.list_blobs(name_starts_with="scaler_"))
        s_sorted = sorted(sblobs, key=lambda b: b.last_modified, reverse=True)
        if not s_sorted:
            raise FileNotFoundError("Nenhum scaler encontrado.")
        sname = s_sorted[0].name

        # Carrega modelo e scaler
        model = joblib.load(io.BytesIO(download_from_blob(mname)))
        scaler = joblib.load(io.BytesIO(download_from_blob(sname)))

        # Carrega dados de teste
        df_data = download_from_blob("test_upload.csv")
        df = pd.read_csv(io.BytesIO(df_data))
        
        if df.empty or 'time' not in df.columns:
            # Para predição, a coluna 'time' pode não ser o target, mas é usada para gerar lags
            raise ValueError("CSV de teste vazio ou coluna 'time' ausente.")

        X, y = build_lags(df, lags)
        X_scaled = scaler.transform(X)

        preds = model.predict(X_scaled)

        out = pd.DataFrame({"predicted": preds})
        if len(y) > 0: # Verifica se 'y' tem dados reais (se a coluna 'time' estava completa)
             # Os valores reais de y começam depois dos lags, então o alinhamento é feito
             # pelo build_lags.
            out["actual"] = y.values
            out["error"] = out["actual"] - out["predicted"]

        # salva CSV
        b = io.BytesIO()
        out.to_csv(b, index=False)
        upload_to_blob("predictions.csv", b.getvalue())

        # registra predições na tabela
        for i, row in out.iterrows():
            # A chave training_id pode ser o RowKey do último treino
            registrar_predicao(training_id, X.iloc[i].to_dict(), float(row["predicted"]))

        return {"status": "ok", "n": len(preds)}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erro na previsão: {str(e)}")


# ------------------------------------------------------
# DOWNLOAD CSV
# ------------------------------------------------------
@app.get("/download/predictions")
async def download_predictions():
    try:
        data = download_from_blob("predictions.csv")
        temp = "/tmp/predictions.csv"
        # Cria o diretório se não existir (necessário em alguns ambientes)
        os.makedirs(os.path.dirname(temp), exist_ok=True)
        with open(temp, "wb") as f:
            f.write(data)
        return FileResponse(temp, filename="predictions.csv", media_type="text/csv")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao baixar o arquivo: {str(e)}")


# ------------------------------------------------------
# LISTAR LOGS DE TREINOS (TABLE)
# ------------------------------------------------------
@app.get("/logs")
async def logs():
    try:
        # Garante que os logs venham em ordem decrescente de timestamp (mais recentes primeiro)
        entities = sorted(list(table_client.list_entities()), key=lambda e: e.timestamp, reverse=True)
        return {"logs": entities}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao buscar logs: {str(e)}")


# ------------------------------------------------------
# LISTAR PREDIÇÕES (TABLE)
# ------------------------------------------------------
@app.get("/predictions/table")
async def predicoes_table():
    try:
        # Garante que as predições venham em ordem decrescente de timestamp
        entities = sorted(list(predictions_table_client.list_entities()), key=lambda e: e.timestamp, reverse=True)
        return {"predictions": entities} # Retorna 'predictions', não 'predicoes'
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao buscar predições da tabela: {str(e)}")


# ============================================================
# FRONTEND EMBUTIDO E ROTA RAIZ (CÓDIGO CORRIGIDO)
# ============================================================

# ATENÇÃO: SUBSTITUA ESTE VALOR pela URL completa do seu Container App!
# Use http://localhost:8000 se estiver rodando localmente.
API_URL = os.getenv("API_URL", "http://localhost:8000") 

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
        button:hover{background:#1d4ed8}
        input[type=file]{padding:6px}
        table{width:100%;border-collapse:collapse;margin-top:8px}
        th,td{padding:6px;border-bottom:1px solid #eee;text-align:left;font-size:13px}
        .row{display:flex;gap:12px;flex-wrap:wrap}
        .col{flex:1;min-width:240px}
        pre{background:#0b1220;color:#dbeafe;padding:10px;border-radius:6px;overflow:auto}
        .error-message{color:red;font-weight:bold;}
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
        <h3>Previsão (Preview do CSV)</h3>
        <pre id="predPreview">Nenhuma previsão gerada ainda.</pre>
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

function errorHandler(error, context = "Ação"){
    log(`ERRO na ${context}: ${error.message || error}. Verifique o console do navegador e a API.`);
    // Onde houver uma div de resultado, mostra o erro.
    const resultDiv = document.getElementById(context.toLowerCase() + 'Result');
    if (resultDiv) {
        resultDiv.innerHTML = `<span class="error-message">Erro: ${error.message || error}</span>`;
    }
}

async function uploadTrain(){
    const f = document.getElementById('trainFile').files[0];
    if(!f){ alert('Selecione o CSV de treino'); return; }
    const fd = new FormData();
    fd.append('file', f);
    log('Enviando treino...');
    try {
        const res = await fetch(`${API_BASE}/upload/train`, { method:'POST', body: fd });
        if(!res.ok) throw new Error(`HTTP Error: ${res.status} - ${await res.text()}`);
        const j = await res.json();
        log('Upload train: ' + JSON.stringify(j));
        document.getElementById('trainResult').innerText = JSON.stringify(j, null, 2);
    } catch(e) {
        errorHandler(e, 'Upload Train');
    }
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
        if(!res.ok) throw new Error(`HTTP Error: ${res.status} - ${text}`);
        
        log('Resposta bruta: ' + text);
        const j = JSON.parse(text);
        document.getElementById('trainResult').innerText = JSON.stringify(j, null, 2);
        
        await getLogs(); // Atualiza a tabela de logs
    } catch (e) {
        errorHandler(e, 'Train');
    }
}

async function uploadTest(){
    const f = document.getElementById('testFile').files[0];
    if(!f){ alert('Selecione o CSV de teste'); return; }
    const fd = new FormData();
    fd.append('file', f);
    log('Enviando teste...');
    try {
        const res = await fetch(`${API_BASE}/upload/test`, { method:'POST', body: fd });
        if(!res.ok) throw new Error(`HTTP Error: ${res.status} - ${await res.text()}`);
        const j = await res.json();
        log('Upload test: ' + JSON.stringify(j));
        document.getElementById('predictResult').innerText = JSON.stringify(j, null, 2);
    } catch(e) {
        errorHandler(e, 'Upload Test');
    }
}

async function predict(){
    log('Rodando predict...');
    try {
        // Envia o PartitionKey como ID do treino (pode ser "default" ou um UUID de treino real)
        const res = await fetch(`${API_BASE}/predict`, { 
            method:'POST',
            headers: { "Content-Type": "application/x-www-form-urlencoded" },
            body: new URLSearchParams({ training_id: "default", lags: 5 })
        });
        
        if(!res.ok) throw new Error(`HTTP Error: ${res.status} - ${await res.text()}`);
        
        const j = await res.json();
        log('Predict: ' + JSON.stringify(j));
        document.getElementById('predictResult').innerText = JSON.stringify(j, null, 2);
        await showPredictionsPreview();
        await getTablePredictions(); // Atualiza a tabela de predições
    } catch(e) {
        errorHandler(e, 'Predict');
    }
}

async function downloadPredictions(){
    const url = `${API_BASE}/download/predictions`;
    log('Baixando ' + url);
    try {
        // Não usamos fetch aqui para permitir o download direto do navegador
        const a = document.createElement('a');
        a.href = url;
        a.download = 'predictions.csv';
        document.body.appendChild(a);
        a.click();
        a.remove();
        log('Download iniciado. Verifique a pasta de downloads.');
    } catch(e) {
        errorHandler(e, 'Download');
    }
}

// CORREÇÃO: Esta função agora busca todos os logs e exibe o MAIS RECENTE
async function getLastMetrics(){
    try{
        const res = await fetch(`${API_BASE}/logs`);
        if(!res.ok) throw new Error(`HTTP Error: ${res.status} - ${await res.text()}`);
        const j = await res.json();
        
        const logs = j.logs || [];
        
        if (logs.length > 0) {
            // Os logs são ordenados por data decrescente na rota /logs
            const lastLog = logs[0]; 
            log('Último treino: ' + JSON.stringify(lastLog));
            // Exibe apenas as métricas importantes do último treino
            document.getElementById('metrics').innerHTML = `
                <h4>Último Treino (Table Storage)</h4>
                <p><strong>ID:</strong> ${lastLog.RowKey}</p>
                <p><strong>Data:</strong> ${lastLog.timestamp}</p>
                <p><strong>MAE:</strong> ${lastLog.MAE}</p>
                <p><strong>RMSE:</strong> ${lastLog.RMSE}</p>
                <p><strong>R2:</strong> ${lastLog.R2}</p>
            `;
        } else {
             document.getElementById('metrics').innerHTML = '<h4>Último Treino (Table Storage)</h4><p>Nenhum treino encontrado.</p>';
        }
    }catch(e){
        errorHandler(e, 'Último Treino');
    }
}

async function getLogs(){
    try{
        const res = await fetch(`${API_BASE}/logs`);
        if(!res.ok) throw new Error(`HTTP Error: ${res.status} - ${await res.text()}`);
        const j = await res.json();
        
        const logs = j.logs || [];
        log(`Logs de treino carregados: ${logs.length} itens.`);

        const html = ['<table><thead><tr><th>RowKey</th><th>timestamp</th><th>MAE</th><th>RMSE</th><th>R2</th></tr></thead><tbody>'];

        for(const it of logs){
            html.push(`<tr>
                <td>${it.RowKey.substring(0, 8)}...</td>
                <td>${it.timestamp.replace('T', ' ').substring(0, 19)}</td>
                <td>${(it.MAE || 0).toFixed(4)}</td>
                <td>${(it.RMSE || 0).toFixed(4)}</td>
                <td>${(it.R2 || 0).toFixed(4)}</td>
            </tr>`);
        }

        html.push('</tbody></table>');
        document.getElementById('metrics').innerHTML = '<h4>Logs de Treino (Azure Table Storage)</h4>' + html.join('');
    }catch(e){
        errorHandler(e, 'Logs Treino');
    }
}

async function getTablePredictions(){
    try{
        const res = await fetch(`${API_BASE}/predictions/table`);
        if(!res.ok) throw new Error(`HTTP Error: ${res.status} - ${await res.text()}`);
        const j = await res.json();
        
        // CORREÇÃO: A rota retorna j.predictions
        const predictions = j.predictions || [];

        if (predictions.length === 0) {
            document.getElementById('metrics').innerHTML = '<h4>Predições (Azure Table Storage)</h4><p>Nenhuma predição encontrada.</p>';
            return;
        }
        
        log(`Predições carregadas: ${predictions.length} itens.`);

        // Cria colunas dinamicamente (RowKey, PredictedValue, e todas as colunas 'lag')
        const allKeys = new Set();
        predictions.forEach(p => Object.keys(p).filter(k => k.startsWith('lag')).forEach(k => allKeys.add(k)));
        const lagKeys = Array.from(allKeys).sort();

        let header = ['<tr><th>RowKey</th><th>Predito</th>'];
        lagKeys.forEach(k => header.push(`<th>${k}</th>`));
        header.push('</tr>');
        
        const html = ['<table><thead>', header.join(''), '</thead><tbody>'];

        for(const it of predictions.slice(0, 20)){ // Limita a 20 para visualização
            let row = [`<tr><td>${it.RowKey.substring(0, 8)}...</td><td>${(it.PredictedValue || 0).toFixed(4)}</td>`];
            lagKeys.forEach(k => row.push(`<td>${(it[k] || 'n/a')}</td>`));
            row.push('</tr>');
            html.push(row.join(''));
        }

        html.push('</tbody></table>');
        document.getElementById('metrics').innerHTML = '<h4>Predições (Azure Table Storage - Últimos 20)</h4>' + html.join('');
    }catch(e){
        errorHandler(e, 'Predições Table');
    }
}

async function showPredictionsPreview(){
    try{
        const res = await fetch(`${API_BASE}/download/predictions`);
        if(!res.ok){ log('Nenhuma previsão disponível para preview.'); document.getElementById('predPreview').innerText = "Nenhuma previsão disponível no Blob Storage."; return; }
        
        const txt = await res.text();
        // Limita a 10 linhas para o preview
        const lines = txt.trim().split('\n').slice(0, 11).join('\n'); 
        document.getElementById('predPreview').innerText = lines;
        log('Preview de predições atualizado.');
    }catch(e){
        errorHandler(e, 'Preview');
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