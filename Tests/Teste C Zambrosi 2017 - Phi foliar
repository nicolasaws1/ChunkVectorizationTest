# ============================================================
# TESTE C — CHUNK POR SEÇÃO DO ARTIGO
# 100% LOCAL — sem Qdrant, usando FAISS
# PDF: Zambrosi 2017 - Phi foliar
# ============================================================

!pip install -q sentence-transformers faiss-cpu

import faiss
import numpy as np
import json
import statistics
from pathlib import Path
from sentence_transformers import SentenceTransformer

from google.colab import drive
drive.mount('/content/drive', force_remount=False)

# ============================================================
# CONFIGURAÇÃO — reutiliza markdown do Teste A Zambrosi
# ============================================================
BASE_DIR = Path("/content/drive/MyDrive/Squad2/leo_Squad2")
PDF_NOME = "Zambrosi(2017)-Phi foliar - Janaina Lais Pacheco Lara Morandin"
MD_PATH  = BASE_DIR / "testes" / "zambrosi" / "teste_A" / "data_md" / f"{PDF_NOME}.md"

print(f"📋 TESTE C — Chunk por seção do artigo")
print(f"   PDF      : Zambrosi 2017 - Phi foliar")
print(f"   Corta em cada ## encontrado no markdown")
print(f"   banco    : FAISS local (sem Qdrant)")
print(f"   markdown : reutilizando do Teste A Zambrosi")

if not MD_PATH.exists():
    print(f"\n❌ Markdown não encontrado: {MD_PATH}")
    print(f"   Rode primeiro o teste_A_zambrosi.py!")
    raise FileNotFoundError(f"Markdown não encontrado: {MD_PATH}")

# ============================================================
# CHUNKING POR SEÇÃO
# ============================================================
def subdividir_secao(secao_texto, titulo, max_sub=800):
    linhas     = secao_texto.split("\n")
    sub_chunks = []
    current    = titulo + "\n"

    for linha in linhas:
        if linha.strip() == titulo.strip():
            continue
        if len(current) + len(linha) + 1 > max_sub:
            if current.strip():
                sub_chunks.append(current.strip())
            current = titulo + "\n" + linha + "\n"
        else:
            current += linha + "\n"

    if current.strip():
        sub_chunks.append(current.strip())

    return sub_chunks

def fazer_chunks_C(md_path):
    """TESTE C: chunk por seção usando ## do Docling"""
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.readlines()

    content = [p.strip() for p in content]
    content = [p for p in content if p.replace('-','').replace('|','').replace(' ','').strip()]

    chunks         = []
    current_chunk  = ""
    current_titulo = ""
    MAX_SECAO      = 2000

    for paragraph in content:
        if paragraph.startswith("##"):
            if current_chunk.strip():
                if len(current_chunk) > MAX_SECAO:
                    chunks.extend(subdividir_secao(current_chunk, current_titulo))
                else:
                    chunks.append(current_chunk.strip())
            current_titulo = paragraph
            current_chunk  = paragraph + "\n"
        else:
            current_chunk += paragraph + "\n"

    if current_chunk.strip():
        if len(current_chunk) > MAX_SECAO:
            chunks.extend(subdividir_secao(current_chunk, current_titulo))
        else:
            chunks.append(current_chunk.strip())

    return chunks

def avaliar_tamanho_chunks(chunks, nome):
    tamanhos = [len(c) for c in chunks]
    print(f"\n{'='*55}")
    print(f"📏 AVALIAÇÃO DE TAMANHO — {nome}")
    print(f"{'='*55}")
    print(f"   Total chunks  : {len(chunks)}")
    print(f"   Menor         : {min(tamanhos)} chars")
    print(f"   Maior         : {max(tamanhos)} chars")
    print(f"   Média         : {statistics.mean(tamanhos):.0f} chars")
    print(f"   Mediana       : {statistics.median(tamanhos):.0f} chars")
    print(f"   Desvio padrão : {statistics.stdev(tamanhos):.0f} chars")
    print(f"\n   Distribuição por faixa:")
    print(f"   < 200 chars  : {sum(1 for t in tamanhos if t < 200):>4} ← muito pequeno ⚠️")
    print(f"   200-400      : {sum(1 for t in tamanhos if 200 <= t < 400):>4}")
    print(f"   400-600      : {sum(1 for t in tamanhos if 400 <= t < 600):>4} ← ideal ✅")
    print(f"   600-800      : {sum(1 for t in tamanhos if 600 <= t < 800):>4} ← ideal ✅")
    print(f"   800-1024     : {sum(1 for t in tamanhos if 800 <= t < 1024):>4}")
    print(f"   > 1024       : {sum(1 for t in tamanhos if t > 1024):>4} ← muito grande ⚠️")
    ideal     = sum(1 for t in tamanhos if 300 <= t <= 800)
    score_tam = (ideal / len(chunks)) * 100
    print(f"\n   ⭐ Score de tamanho: {score_tam:.1f}%")
    print(f"      ({ideal} de {len(chunks)} chunks entre 300-800 chars)")
    return score_tam

# ============================================================
# EMBEDDING + FAISS LOCAL
# ============================================================
print("\n🔄 Carregando modelo de embedding (Qwen3 multilingual)...")
model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
print("✅ Modelo carregado!")

def criar_indice_faiss(chunks):
    print(f"🔄 Gerando embeddings para {len(chunks)} chunks...")
    embeddings = model.encode(chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')
    faiss.normalize_L2(embeddings)
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"✅ Índice FAISS criado com {index.ntotal} vetores!")
    return index

def buscar_faiss(pergunta, index, chunks, top_k=1):
    vetor = model.encode([pergunta])
    vetor = np.array(vetor).astype('float32')
    faiss.normalize_L2(vetor)
    scores, indices = index.search(vetor, top_k)
    return [{
        "score"   : float(scores[0][i]),
        "chunk"   : chunks[indices[0][i]],
        "chunk_id": int(indices[0][i]),
        "chars"   : len(chunks[indices[0][i]])
    } for i in range(top_k)]

# ============================================================
# EXECUÇÃO
# ============================================================
chunks    = fazer_chunks_C(str(MD_PATH))
score_tam = avaliar_tamanho_chunks(chunks, "TESTE C — Por Seção")

print(f"\n📋 SEÇÕES ENCONTRADAS:")
for i, chunk in enumerate(chunks):
    primeira_linha = chunk.split("\n")[0]
    print(f"   Chunk {i+1:02d} | {len(chunk):4d} chars | {primeira_linha[:60]}")

index_C = criar_indice_faiss(chunks)

# ============================================================
# BUSCA — perguntas em PORTUGUÊS sobre PDF em INGLÊS
# ============================================================
perguntas = [
    "Qual o efeito do fosfito foliar em citros com deficiência de fósforo?",
    "Como o fosfito afeta a anatomia foliar dos citros?",
    "Qual a diferença entre fosfato e fosfito para plantas?",
    "Como o fosfito afeta os cloroplastos das folhas?",
    "Quais são os sintomas de toxicidade do fosfito em citros?",
    "Como a deficiência de fósforo afeta o crescimento dos citros?",
    "Qual o impacto do fosfito na ultraestrutura foliar?",
    "Como o suprimento de fósforo afeta a resposta ao fosfito foliar?"
]

print(f"\n{'='*60}")
print(f"🔍 RESULTADOS DAS BUSCAS — TESTE C (Por Seção)")
print(f"   Perguntas em português | Texto em inglês")
print(f"   Modelo: Qwen3-Embedding-0.6B (multilingual)")
print(f"{'='*60}")

scores_C = []
for pergunta in perguntas:
    r = buscar_faiss(pergunta, index_C, chunks)[0]
    scores_C.append(r['score'])
    relevante = "✅" if r['score'] > 0.55 else "⚠️" if r['score'] > 0.45 else "❌"
    print(f"\n❓ {pergunta}")
    print(f"   Score  : {r['score']:.4f} {relevante}")
    print(f"   Chars  : {r['chars']}")
    print(f"   Seção  : {r['chunk'].split(chr(10))[0][:60]}")
    print(f"   Texto  : {r['chunk'][:200]}...")
    print("-"*60)

media_score = sum(scores_C)/len(scores_C)
print(f"\n📈 MÉDIA DE SCORE TESTE C : {media_score:.4f}")
print(f"   Score tamanho          : {score_tam:.1f}%")
print(f"   Scores > 0.55 (bons)   : {sum(1 for s in scores_C if s > 0.55)}/{len(scores_C)}")
print(f"   Scores 0.45-0.55 (ok)  : {sum(1 for s in scores_C if 0.45 <= s <= 0.55)}/{len(scores_C)}")
print(f"   Scores < 0.45 (ruins)  : {sum(1 for s in scores_C if s < 0.45)}/{len(scores_C)}")

resultados_C = {
    "nome"          : "C - Por Seção",
    "pdf"           : "Zambrosi 2017 - Phi foliar",
    "modelo"        : "Qwen3-Embedding-0.6B",
    "chunk_size"    : "por seção",
    "overlap"       : 0,
    "total_chunks"  : len(chunks),
    "menor_chunk"   : min(len(c) for c in chunks),
    "maior_chunk"   : max(len(c) for c in chunks),
    "media_chunk"   : sum(len(c) for c in chunks) // len(chunks),
    "score_tamanho" : score_tam,
    "scores"        : scores_C,
    "media_score"   : media_score
}

with open("/content/resultado_zambrosi_C.json", "w") as f:
    json.dump(resultados_C, f, ensure_ascii=False, indent=2)

print(f"\n✅ TESTE C CONCLUÍDO!")
print(f"   Resultados salvos em /content/resultado_zambrosi_C.json")
print(f"   Rode agora o comparativo_zambrosi.py!")
