import os
import re
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
import openai

# ==========================================================
# CONFIGURAÇÕES GERAIS E SECRETS
# ==========================================================
DB_DIR = "chroma_db"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

def get_config(key: str, default: str = "") -> str:
    """Busca em st.secrets primeiro (ideal para Streamlit Cloud), depois no SO."""
    if key in st.secrets:
        return st.secrets[key]
    return os.getenv(key, default)

LLM_API_KEY = get_config("LLM_API_KEY", "ollama")
LLM_BASE_URL = get_config("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_MODEL = get_config("LLM_MODEL", "llama3")
COLLECTION_NAME = get_config("COLLECTION_NAME", "morning_crypto")

# Busca o limite de contextos do secrets. Se falhar ou estiver vazio, usa 7 por padrão.
try:
    MAX_LLM_CONTEXTS = int(get_config("MAX_LLM_CONTEXTS", "7"))
except ValueError:
    MAX_LLM_CONTEXTS = 7
    
# ==========================================================
# DETECÇÃO DE BACKEND LLM
# ==========================================================
if "llm_backend" not in st.session_state:
    backend = None
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:114345", timeout=1) # Ajustado timeout url para localhost:11434
        backend = "ollama"
    except Exception:
        if LLM_API_KEY and LLM_API_KEY.lower() not in ["", "ollama"]:
            backend = "openai_compatible" # Nome genérico para cobrir Mistral, Groq, OpenRouter, OpenAI
    st.session_state.llm_backend = backend

LLM_BACKEND = st.session_state.llm_backend

# ==========================================================
# INTERFACE (SIDEBAR)
# ==========================================================
st.set_page_config(page_title="Oráculo Cripto", page_icon="🧠", layout="wide")
st.title("🧠 Oráculo Cripto")
st.caption("A IA busca, lê e sintetiza as transcrições do Morning Crypto.")

with st.sidebar:
    st.header("⚙️ Configurações de Busca")
    n_results = st.slider("Contextos recuperados (Top-K)", 3, 100, 7)
    similarity_threshold = st.slider("Aderência mínima", 0.0, 1.0, 0.30, 0.05)
    
    st.divider()
    st.header("🕵️ Modos")
    modo_varredura = st.toggle("Modo Varredura (Palavra-Chave)", value=False, 
                               help="Ignora a IA e a similaridade. Busca TODAS as menções exatas da palavra no banco (ideal para listar livros, projetos, etc).")
    
    st.divider()
    st.header("🤖 IA / LLM")
    if LLM_BACKEND is None:
        st.error("Nenhum backend LLM detectado. Configure as secrets ou ligue o Ollama.")
        use_llm = False
    else:
        use_llm = st.toggle("Ativar resposta com IA", value=True)
        st.caption(f"Backend: **{LLM_BACKEND}** | Modelo: `{LLM_MODEL}`")
        if st.button("Limpar Histórico de Chat"):
            st.session_state.messages = []
            st.rerun()
        
    st.divider()
    st.header("📺 Exibição")
    modo_digest = st.toggle("Modo Digest (só resposta + fontes)", value=True,
                            help="Esconde os trechos brutos; mostra só a resposta da IA com links para as fontes.")

# ==========================================================
# BANCO VETORIAL (Refatorado para estabilidade)
# ==========================================================
@st.cache_resource(show_spinner="Carregando memória vetorial (Client)...")
def get_chroma_client():
    return chromadb.PersistentClient(path=DB_DIR)

def get_collection():
    client = get_chroma_client()
    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL, device="cpu"
    )
    # get_or_create_collection evita crashes caso a coleção não exista ainda
    return client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=embedder)

try:
    collection = get_collection()
except Exception as e:
    st.error(f'❌ Falha crítica ao conectar ao ChromaDB: {e}')
    st.stop()

# ==========================================================
# HELPERS
# ==========================================================
def ts_to_seconds(ts: str) -> int:
    try:
        parts = ts.strip().split(":")
        if len(parts) == 3:
            h, m, s = map(int, parts)
            return h * 3600 + m * 60 + s
    except Exception:
        pass
    return 0

def build_yt_link(link: str, seconds: int) -> str:
    if not link or not link.startswith("http"):
        return ""
    sep = "&" if "?" in link else "?"
    return f"{link}{sep}t={seconds}s"

def build_markdown_export(query: str, answer: str, contexts: list, modo_varredura: bool) -> str:
    lines = ["# 🧠 Oráculo Cripto — Resultado", ""]
    if modo_varredura:
        lines.append(f"**Modo Varredura — palavra-chave:** `{query}`")
    else:
        lines.append(f"**Pergunta:** {query}")
    lines.append("")

    if answer:
        lines.append("## Resposta")
        lines.append("")
        lines.append(answer)
        lines.append("")

    lines.append(f"## Fontes consultadas ({len(contexts)})")
    lines.append("")
    for i, c in enumerate(contexts, 1):
        lines.append(f"### [{i}] {c['title']}")
        lines.append(f"- **Data:** {c['date']}")
        lines.append(f"- **Timestamp:** `{c['ts']}`")
        lines.append(f"- **Aderência:** {round(c['similarity'] * 100, 1)}%")
        if c["yt_link"]:
            lines.append(f"- **Link:** {c['yt_link']}")
        lines.append("")
        lines.append(f"> {c['text']}")
        lines.append("")

    lines.append("---")
    lines.append("_Gerado pelo Oráculo Cripto_")
    return "\n".join(lines)


def synthesize(query: str, contexts: list) -> str:
    if not contexts:
        return "Não encontrei informações relevantes nas transcrições para responder essa pergunta."
    
    client = openai.OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

    fontes = []
    for idx, c in enumerate(contexts, 1):
        fontes.append(
            f"[FONTE {idx}]\n"
            f"Título: {c['title']}\n"
            f"Data: {c['date']}\n"
            f"Timestamp: {c['ts']}\n"
            f"Trecho: {c['text']}"
        )
    context_block = "\n\n---\n\n".join(fontes)
    
    system_prompt = (
        "Você é um assistente especialista em criptomoedas, blockchain e tecnologia. "
        "Seu trabalho é responder à pergunta do usuário de forma clara, direta e completa.\n\n"
        "REGRAS IMPORTANTES:\n"
        "1. Use APENAS as informações das fontes fornecidas abaixo. Não invente dados.\n"
        "2. As transcrições contêm erros de ASR (ex: 'biscoito visconti' = Biscoint). Corrija-os no contexto.\n"
        "3. Se a informação não estiver nas fontes, diga honestamente que não sabe com base nas fontes.\n"
        "4. Cite as fontes usando [^N] associando aos números das fontes fornecidas.\n"
        "5. Responda em português do Brasil.\n\n"
    )
    
    user_prompt = f"### Pergunta do usuário\n{query}\n\n### Fontes recuperadas\n{context_block}\n\n### Resposta:"

    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.25,
            max_tokens=4000 # Ajuste conforme necessário.
        )
        return resp.choices[0].message.content.strip()
    
    # Captura de erros específicos e descritivos
    except openai.AuthenticationError:
         return "❌ Erro: Chave de API (API Key) inválida ou não configurada corretamente nas secrets."
    except openai.RateLimitError:
         return "⏳ Erro: Limite de uso da API excedido (Rate Limit). Aguarde um minuto e tente novamente."
    except openai.APIConnectionError:
         return "🔌 Erro: Falha ao conectar com o provedor do LLM. Verifique sua conexão com a internet ou se o Ollama está rodando."
    except Exception as e:
        return f"⚠️ Erro inesperado na síntese do LLM: {str(e)}"

# ==========================================================
# LÓGICA CORE DE BUSCA
# ==========================================================

def run_query(query, modo_varredura, n_results, similarity_threshold, use_llm, modo_digest):
    try:
        if modo_varredura:
            results = collection.get(
                where_document={"$contains": query.lower()},
                include=["documents", "metadatas"]
            )
            docs = results.get("documents", []) or []
            metas = results.get("metadatas", []) or []
            dists = [0.0] * len(docs)
        else:
            results = collection.query(
                query_texts=[query],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            dists = results.get("distances", [[]])[0]
    except Exception as e:
        return {"query": query, "modo_varredura": modo_varredura, "contexts": [],
                "answer": "", "llm_failed": False, "llm_attempted": False,
                "modo_digest": modo_digest, "error_message": f"❌ Erro na busca: {e}"}

    contexts = []
    for doc, meta, dist in zip(docs, metas, dists):
        sim = max(0.0, min(1.0, 1 - dist))
        if not modo_varredura and sim < similarity_threshold:
            continue

        ts = meta.get("start_time", "00:00:00")
        link = meta.get("link", "")
        title = meta.get("source", "Sem título")
        date = meta.get("date", "N/A")
        seconds = ts_to_seconds(ts)
        yt_link = build_yt_link(link, seconds)

        contexts.append({
            "title": title, "date": date, "ts": ts, "text": doc,
            "similarity": sim, "yt_link": yt_link, "source_file": meta.get("source", "")
        })

    if not contexts:
        msg = f"Nenhuma menção exata a '{query}' encontrada." if modo_varredura else "Os trechos não atingiram a aderência mínima."
        return {"query": query, "modo_varredura": modo_varredura, "contexts": [],
                "answer": "", "llm_failed": False, "llm_attempted": False,
                "modo_digest": modo_digest, "error_message": msg}

    answer = ""
    llm_failed = False
    llm_attempted = use_llm and bool(LLM_BACKEND) and not modo_varredura
    
    if llm_attempted:
        # Pega apenas os TOP N para o LLM, baseando na variável global configurável.
        answer = synthesize(query, contexts[:MAX_LLM_CONTEXTS])
        if answer.startswith("❌") or answer.startswith("⏳") or answer.startswith("🔌") or answer.startswith("⚠️"):
             llm_failed = True # Identifica que a resposta foi uma mensagem de erro capturada

    return {
        "query": query, "modo_varredura": modo_varredura, "contexts": contexts,
        "answer": answer, "llm_failed": llm_failed, "llm_attempted": llm_attempted,
        "modo_digest": modo_digest, "error_message": None,
        "llm_context_capped": llm_attempted and len(contexts) > MAX_LLM_CONTEXTS,
    }

# ==========================================================
# RENDERIZAÇÃO DE UI
# ==========================================================

def render_result(result, message_index):
    """Renderiza a resposta do assistente baseada no dicionário result."""
    if result["error_message"]:
        st.warning(result["error_message"])
        return

    modo_varredura = result["modo_varredura"]
    contexts = result["contexts"]
    answer = result["answer"]
    query_slug = re.sub(r"[^a-zA-Z0-9]+", "_", result["query"]).strip("_").lower()[:40]

    if modo_varredura:
        st.info(f"🎯 **{len(contexts)} menções encontradas!** (Modo Varredura ignora o Top-K e a IA)")

    effective_digest = False
    if answer:
        if result.get("llm_context_capped") and not result["llm_failed"]:
            st.caption(f"ℹ️ A IA sintetizou usando as {MAX_LLM_CONTEXTS} fontes mais relevantes de um total de {len(contexts)} recuperadas.")
        
        if result["llm_failed"]:
             # Mostra o erro formatado capturado no except
             st.error(answer)
             st.warning("Exibindo os trechos brutos recuperados abaixo devido à falha da IA.")
        else:
            st.markdown(answer)
            st.markdown("**🔗 Fontes citadas:**")
            cols = st.columns(min(len(contexts), 4))
            for idx, c in enumerate(contexts):
                with cols[idx % 4]:
                    if c["yt_link"]:
                        st.link_button(
                            f"▶️ {c['title'][:20]}... ({c['ts']})",
                            c["yt_link"],
                            help=f"Aderência: {round(c['similarity'] * 100, 1)}% | {c['date']}",
                            key=f"cited_{message_index}_{idx}"
                        )
        
        effective_digest = result["modo_digest"] and not modo_varredura and not result["llm_failed"]
        
    elif result["llm_attempted"] is False and result["modo_digest"] and not modo_varredura:
        st.info("Modo Digest ativo, mas IA desabilitada. Exibindo trechos brutos.")

    # ── FONTES / TRECHOS ──
    if effective_digest:
        with st.expander(f"Ver fontes consultadas ({len(contexts)})", expanded=False):
            for i, c in enumerate(contexts, 1):
                col1, col2 = st.columns([5, 1])
                with col1:
                    st.markdown(f"**[{i}] [{c['title']}]({c['yt_link'] or '#'})** — `{c['ts']}` | 📅 {c['date']} | 🎯 {round(c['similarity'] * 100, 1)}%")
                    st.caption(f"> {c['text'][:250]}{'...' if len(c['text']) > 250 else ''}")
                with col2:
                    if c["yt_link"]:
                        st.link_button("Assistir", c["yt_link"], key=f"src_link_{message_index}_{i}")
                st.divider()
    else:
        st.subheader(f"📚 Trechos recuperados ({len(contexts)})")
        for i, c in enumerate(contexts, 1):
            with st.container():
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"**{i}. [{c['title']}]({c['yt_link'] or '#'})**")
                    st.caption(f"📅 {c['date']}  •  ⏱️ `{c['ts']}`   •  🎯 Aderência: `{round(c['similarity'] * 100, 1)}%`")
                    with st.expander("Ver trecho completo", expanded=(i == 1 or modo_varredura)):
                        st.markdown(f"> {c['text']}")
                with col2:
                    if c["yt_link"]:
                        st.link_button("▶️ YouTube", c["yt_link"], use_container_width=True, key=f"det_link_{message_index}_{i}")
                st.divider()

    # ── EXPORTAR ──
    md_export = build_markdown_export(result["query"], answer if not result["llm_failed"] else "", contexts, modo_varredura)
    with st.expander("📤 Exportar este resultado", expanded=False):
        st.download_button(
            label="📥 Baixar em Markdown",
            data=md_export,
            file_name=f"oraculo_{query_slug}.md",
            mime="text/markdown",
            use_container_width=True,
            key=f"dl_md_{message_index}"
        )


# ==========================================================
# GERENCIAMENTO DE ESTADO E CHAT (Novo Histórico)
# ==========================================================

# Inicializa o histórico de mensagens se não existir
if "messages" not in st.session_state:
    st.session_state.messages = []

# Exibe as mensagens armazenadas na sessão atual
for idx, msg in enumerate(st.session_state.messages):
    if msg["role"] == "user":
        with st.chat_message("user"):
            if msg.get("modo_varredura"):
                 st.markdown(f"🕵️ **Varredura exata por:** `{msg['content']}`")
            else:
                 st.markdown(msg["content"])
    elif msg["role"] == "assistant":
        with st.chat_message("assistant"):
            render_result(msg["result_data"], message_index=idx)

# Captura nova entrada do usuário
placeholder_text = "🔍 Digite uma palavra-chave (ex: livro, IPFS)..." if modo_varredura else "💬 O que você quer saber?"
new_query = st.chat_input(placeholder_text)

if new_query:
    # 1. Adiciona e exibe a pergunta do usuário na UI imediatamente
    st.session_state.messages.append({
         "role": "user", 
         "content": new_query,
         "modo_varredura": modo_varredura
    })
    
    with st.chat_message("user"):
        if modo_varredura:
             st.markdown(f"🕵️ **Varredura exata por:** `{new_query}`")
        else:
             st.markdown(new_query)
    
    # 2. Processa o RAG e LLM mostrando um spinner
    with st.chat_message("assistant"):
        spinner_msg = f"Varrendo o banco por '{new_query}'..." if modo_varredura else "Consultando os oráculos..."
        with st.spinner(spinner_msg):
            # Roda a função principal
            result_data = run_query(new_query, modo_varredura, n_results, similarity_threshold, use_llm, modo_digest)
            
            # Adiciona o resultado gerado ao estado para sobreviver aos re-runs da página
            st.session_state.messages.append({
                "role": "assistant",
                "result_data": result_data
            })
            
            # Renderiza imediatamente a resposta na tela
            render_result(result_data, message_index=len(st.session_state.messages)-1)
            
elif len(st.session_state.messages) == 0:
    st.info("👋 Bem-vindo! Digite sua pergunta ou palavra-chave na barra inferior para começar.")
