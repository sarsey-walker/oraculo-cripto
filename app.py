import os
import re
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
import openai

# ==========================================================
# CONFIGURAÇÕES GERAIS
# ==========================================================
DB_DIR = "chroma_db"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
# v3.2: com o Top-K liberado até 100, mandar TODOS os trechos pro LLM pode
# estourar a janela de contexto de modelos locais (ex: llama3 via Ollama) e
# encarecer/atrasar chamadas à OpenAI. O Top-K continua controlando quantos
# trechos são recuperados/exibidos/exportados; este limite é só o teto do
# que efetivamente entra no prompt de síntese.
MAX_LLM_CONTEXTS = 20


def _secret(key: str, default: str = "") -> str:
    """
    v3.1: FIX - st.secrets.get(key, default) ainda estoura FileNotFoundError se
    não existir nenhum .streamlit/secrets.toml no projeto (é um comportamento
    documentado do Streamlit, não um bug nosso). Como isso é avaliado
    ANTES do os.getenv(...) rodar - Python não faz curto-circuito em
    argumento de função - o app quebrava na primeira linha mesmo pra quem só
    queria usar variável de ambiente e nunca criou um secrets.toml.
    """
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return default


LLM_API_KEY = os.getenv("LLM_API_KEY", _secret("LLM_API_KEY", "ollama"))
LLM_BASE_URL = os.getenv("LLM_BASE_URL", _secret("LLM_BASE_URL", "http://localhost:11434/v1"))
LLM_MODEL = os.getenv("LLM_MODEL", _secret("LLM_MODEL", "llama3"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", _secret("COLLECTION_NAME", "morning_crypto"))

# ==========================================================
# DETECÇÃO DE BACKEND LLM
# ==========================================================
if "llm_backend" not in st.session_state:
    backend = None
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:114345", timeout=1)
        backend = "ollama"
    except Exception:
        if LLM_API_KEY and LLM_API_KEY not in ["", "ollama"]:
            backend = "openai"
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
        st.info("LLM desabilitado. Rode o Ollama ou configure a API.")
        use_llm = False
    else:
        use_llm = st.toggle("Ativar resposta com IA", value=True)
        st.caption(f"Backend: **{LLM_BACKEND}** | Modelo: `{LLM_MODEL}`")
        
    st.divider()
    st.header("📺 Exibição")
    modo_digest = st.toggle("Modo Digest (só resposta + fontes)", value=True,
                            help="Esconde os trechos brutos; mostra só a resposta da IA com links para as fontes.")
    
    st.divider()
    st.caption("v3.2 — Top-K até 100, exportação em Markdown, secrets seguros e coleção configurável")

# ==========================================================
# BANCO VETORIAL
# ==========================================================
@st.cache_resource(show_spinner="Carregando memória vetorial...")
def load_db():
    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL, device="cpu"
    )
    client = chromadb.PersistentClient(path=DB_DIR)
    return client.get_collection(name=COLLECTION_NAME, embedding_function=embedder)

try:
    collection = load_db()
except Exception as e:
    st.error(f'❌ Erro real ao carregar o banco: {e}')
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
    """Monta um .md com a pergunta/varredura, a resposta da IA (se houver) e as fontes,
    pronto para ser baixado ou colado em outro lugar (Discord, Notion, Telegram, etc.)."""
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
    
    # Mágica: Ollama aceita a biblioteca da OpenAI mudando o base_url!
    if LLM_BACKEND == "ollama":
        client = openai.OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    else:
        client = openai.OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

    # Monta o bloco de contexto
    fontes = []
    for idx, c in enumerate(contexts, 1):
        fontes.append(
            f"[FONTE {idx}]\n"
            f"Título: {c['title']}\n"
            f"Data: {c['date']}\n"
            f"Timestamp: {c['ts']}\n"
            f"Link: {c['yt_link']}\n"
            f"Trecho: {c['text']}"
        )
    context_block = "\n\n---\n\n".join(fontes)
    
    system_prompt = (
        "Você é um assistente especialista em criptomoedas, blockchain e tecnologia. "
        "Seu trabalho é responder à pergunta do usuário de forma clara, direta e completa.\n\n"
        "REGRAS IMPORTANTES:\n"
        "1. Use APENAS as informações das fontes fornecidas abaixo.\n"
        "2. As transcrições contêm erros graves de ASR (ex: 'biscoito visconti' = Biscoint, 'topo do homens' = Unstoppable Domains, 'etfs/ufff' = IPFS, 'Marinete' = Mainnet, 'teste Nat' = Testnet, 'Robin' = Halving). Corrija mentalmente esses termos ao formular a resposta.\n"
        "3. Se a informação não estiver nas fontes, diga honestamente que não encontrou na base de conhecimento.\n"
        "4. Cite as fontes usando [^N] e mencione o timestamp exato.\n"
        "5. Responda em português do Brasil, com tom natural e acessível.\n\n"
    )
    
    user_prompt = f"### Pergunta do usuário\n{query}\n\n### Fontes recuperadas\n{context_block}\n\n### Resposta"

    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.25,
            max_tokens=8200
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Erro no LLM: {e}")
        return ""

# ==========================================================
# INTERFACE PRINCIPAL (CHAT)
# ==========================================================

def run_query(query, modo_varredura, n_results, similarity_threshold, use_llm, modo_digest):
    """Faz a busca (vetorial ou varredura) e, se aplicável, a síntese via LLM.
    Retorna um dict "congelado" com tudo que a UI precisa para renderizar,
    para ser guardado em session_state."""
    try:
        if modo_varredura:
            results = collection.get(
                where_document={"$contains": query.lower()},
                include=["documents", "metadatas"]
            )
            docs = results.get("documents", []) or []
            metas = results.get("metadatas", []) or []
            dists = [0.0] * len(docs)  # Zera a distância para enganar a lógica de similaridade
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
                "modo_digest": modo_digest,
                "error_message": f"❌ Erro ao consultar o banco vetorial: {e}"}

    # ── MONTAR CONTEXTO ──
    contexts = []
    for doc, meta, dist in zip(docs, metas, dists):
        # v3.2: "sim = 1 - dist" assume métrica de distância coseno (0 a 1).
        # Se a coleção foi criada sem hnsw:space="cosine" (default do Chroma
        # é L2), esse valor pode sair negativo ou > 1 — por isso o clamp.
        # Se a "aderência" mostrada na UI parecer sem sentido, verifique
        # collection.metadata na criação da coleção em rag_ingest.py.
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
        if modo_varredura:
            msg = f"Nenhuma menção exata a '{query}' encontrada no banco."
        elif not docs:
            msg = "Não encontrei nada relevante nas transcrições para essa pergunta."
        else:
            msg = "Os trechos encontrados não atingiram a aderência mínima. Tente ajustar o slider na sidebar."
        return {"query": query, "modo_varredura": modo_varredura, "contexts": [],
                "answer": "", "llm_failed": False, "llm_attempted": False,
                "modo_digest": modo_digest, "error_message": msg}

    # ── RESPOSTA DA IA ──
    answer = ""
    llm_failed = False
    llm_attempted = use_llm and bool(LLM_BACKEND) and not modo_varredura
    if llm_attempted:
        # v3.2: com Top-K até 100, mandar todos os trechos pro LLM pode
        # estourar a janela de contexto de modelos locais / sair caro na
        # OpenAI. Os contextos já vêm ordenados por similaridade, então
        # cortamos para os MAX_LLM_CONTEXTS mais relevantes só para a síntese
        # — a exibição e a exportação em Markdown continuam com todos.
        answer = synthesize(query, contexts[:MAX_LLM_CONTEXTS])
        llm_failed = not answer

    return {
        "query": query, "modo_varredura": modo_varredura, "contexts": contexts,
        "answer": answer, "llm_failed": llm_failed, "llm_attempted": llm_attempted,
        "modo_digest": modo_digest, "error_message": None,
        "llm_context_capped": llm_attempted and len(contexts) > MAX_LLM_CONTEXTS,
    }


def render_result(result):
    """Renderiza um resultado (novo ou vindo de session_state) na tela."""
    query = result["query"]
    modo_varredura = result["modo_varredura"]
    contexts = result["contexts"]
    answer = result["answer"]

    with st.chat_message("user"):
        if modo_varredura:
            st.markdown(f"🕵️ **Varredura exata por:** `{query}`")
        else:
            st.markdown(query)

    with st.chat_message("assistant"):
        if result["error_message"]:
            st.warning(result["error_message"])
            return

        if modo_varredura:
            st.info(f"🎯 **{len(contexts)} menções encontradas!** (Modo Varredura ignora o Top-K e a IA)")

        effective_digest = False
        if answer:
            if result.get("llm_context_capped"):
                st.caption(f"ℹ️ A IA usou os {MAX_LLM_CONTEXTS} trechos mais relevantes dos {len(contexts)} recuperados.")
            st.markdown(answer)

            st.markdown("**🔗 Fontes citadas:**")
            cols = st.columns(min(len(contexts), 4))
            for idx, c in enumerate(contexts):
                with cols[idx % 4]:
                    if c["yt_link"]:
                        st.link_button(
                            f"▶️ {c['title'][:25]}... ({c['ts']})",
                            c["yt_link"],
                            help=f"Aderência: {round(c['similarity'] * 100, 1)}% | {c['date']}",
                            key=f"cited_{idx}"
                        )
            effective_digest = result["modo_digest"] and not modo_varredura
        elif result["llm_failed"]:
            st.warning("A IA não conseguiu gerar uma resposta. Exibindo os trechos brutos abaixo.")
        elif result["llm_attempted"] is False and result["modo_digest"] and not modo_varredura:
            st.info("Modo Digest ativo, mas IA desabilitada. Exibindo trechos brutos.")

        # ── FONTES / TRECHOS ──
        if effective_digest:
            with st.expander(f"Ver fontes consultadas ({len(contexts)})", expanded=False):
                for i, c in enumerate(contexts, 1):
                    col1, col2 = st.columns([5, 1])
                    with col1:
                        st.markdown(
                            f"**[{i}] [{c['title']}]({c['yt_link'] or '#'})** — "
                            f"`{c['ts']}` | 📅 {c['date']} | 🎯 {round(c['similarity'] * 100, 1)}%"
                        )
                        st.caption(f"> {c['text'][:250]}{'...' if len(c['text']) > 250 else ''}")
                    with col2:
                        if c["yt_link"]:
                            st.link_button("Assistir", c["yt_link"], key=f"src_link_{i}")
                    st.divider()
        else:
            st.subheader(f"📚 Trechos recuperados ({len(contexts)})")
            for i, c in enumerate(contexts, 1):
                with st.container():
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.markdown(f"**{i}. [{c['title']}]({c['yt_link'] or '#'})**")
                        st.caption(
                            f"📅 {c['date']}  •  ⏱️ `{c['ts']}`   •  🎯 Aderência: `{round(c['similarity'] * 100, 1)}%`"
                        )
                        with st.expander("Ver trecho completo", expanded=(i == 1 or modo_varredura)):
                            st.markdown(f"> {c['text']}")
                    with col2:
                        if c["yt_link"]:
                            st.link_button("▶️ YouTube", c["yt_link"], use_container_width=True, key=f"det_link_{i}")
                    st.divider()

        # ── COMPARTILHAR / EXPORTAR ──
        md_export = build_markdown_export(query, answer, contexts, modo_varredura)
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", query).strip("_").lower()[:40] or "resultado"

        with st.expander("📤 Compartilhar resultado em Markdown", expanded=False):
            st.download_button(
                label="📥 Baixar como .md",
                data=md_export,
                file_name=f"oraculo_cripto_{slug}.md",
                mime="text/markdown",
                use_container_width=True,
                key="download_md"
            )
            st.caption("Ou copie o Markdown abaixo (ícone de cópia no canto do bloco):")
            st.code(md_export, language="markdown")


placeholder_text = "🔍 Digite uma palavra-chave (ex: livro, IPFS)..." if modo_varredura else "💬 O que você quer saber? (Ex: Como funciona um nó Lightning?)"
new_query = st.chat_input(placeholder_text)

if new_query:
    spinner_msg = f"Varrendo o banco atrás de '{new_query}'..." if modo_varredura else "Buscando nas transcrições e preparando a resposta..."
    with st.spinner(spinner_msg):
        st.session_state.last_result = run_query(
            new_query, modo_varredura, n_results, similarity_threshold, use_llm, modo_digest
        )

if "last_result" not in st.session_state or st.session_state.last_result is None:
    st.info("Digite sua pergunta ou palavra-chave na barra inferior para começar.")
    st.stop()

render_result(st.session_state.last_result)
