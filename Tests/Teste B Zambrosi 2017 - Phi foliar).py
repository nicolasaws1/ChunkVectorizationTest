# ============================================================
# TESTE B — CHUNK 512 COM OVERLAP 128
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
BASE_DIR   = Path("/content/drive/MyDrive/Squad2/leo_Squad2")
PDF_NOME   = "Zambrosi(2017)-Phi foliar - Janaina Lais Pacheco Lara Morandin"
MD_PATH    = BASE_DIR / "testes" / "zambrosi" / "teste_A" / "data_md" / f"{PDF_NOME}.md"
CHUNK_SIZE = 512
OVERLAP    = 128

print(f"📋 TESTE B — Chunk menor com overlap")
print(f"   PDF        : Zambrosi 2017 - Phi foliar")
print(f"   chunk_size : {CHUNK_SIZE}")
print(f"   overlap    : {OVERLAP}")
print(f"   banco      : FAISS local (sem Qdrant)")
print(f"   markdown   : reutilizando do Teste A Zambrosi")

if not MD_PATH.exists():
    print(f"\n❌ Markdown não encontrado: {MD_PATH}")
    print(f"   Rode primeiro o teste_A_zambrosi.py!")
    raise FileNotFoundError(f"Markdown não encontrado: {MD_PATH}")

# ============================================================
# CHUNKING COM OVERLAP
# ============================================================
def fazer_chunks_B(md_path):
    """TESTE B: chunk 512 com overlap 128"""
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.readlines()

    content = [p.strip() for p in content]
    content = [p for p in content if p.replace('-','').replace('|','').replace(' ','').strip()]

    chunks        = []
    current_chunk = ""
    last_overlap  = ""

    for i, paragraph in enumerate(content):
        if i == 0:
            current_chunk += paragraph
            continue

        # Tabela — respeita limite também
        if paragraph.startswith("| ") and paragraph.endswith(" |"):
            if len(current_chunk) + len(paragraph) + 1 > CHUNK_SIZE:
                if current_chunk:
                    chunks.append(current_chunk)
                    last_overlap  = current_chunk[-OVERLAP:]
                    current_chunk = last_overlap + "\n" + paragraph
            else:
                current_chunk += "\n" + paragraph
            continue

        # Texto normal
        if len(current_chunk) + len(paragraph) + 1 > CHUNK_SIZE:
            if current_chunk:
                chunks.append(current_chunk)
                last_overlap  = current_chunk[-OVERLAP:]
                current_chunk = last_overlap + "\n" + paragraph
        else:
            current_chunk += "\n" + paragraph

    if current_chunk:
        chunks.append(current_chunk)

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
chunks    = fazer_chunks_B(str(MD_PATH))
score_tam = avaliar_tamanho_chunks(chunks, "TESTE B — 512 + Overlap 128")
index_B   = criar_indice_faiss(chunks)

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
print(f"🔍 RESULTADOS DAS BUSCAS — TESTE B (512 + Overlap 128)")
print(f"   Perguntas em português | Texto em inglês")
print(f"   Modelo: Qwen3-Embedding-0.6B (multilingual)")
print(f"{'='*60}")

scores_B = []
for pergunta in perguntas:
    r = buscar_faiss(pergunta, index_B, chunks)[0]
    scores_B.append(r['score'])
    relevante = "✅" if r['score'] > 0.55 else "⚠️" if r['score'] > 0.45 else "❌"
    print(f"\n❓ {pergunta}")
    print(f"   Score : {r['score']:.4f} {relevante}")
    print(f"   Chars : {r['chars']}")
    print(f"   Texto : {r['chunk'][:200]}...")
    print("-"*60)

media_score = sum(scores_B)/len(scores_B)
print(f"\n📈 MÉDIA DE SCORE TESTE B : {media_score:.4f}")
print(f"   Score tamanho          : {score_tam:.1f}%")
print(f"   Scores > 0.55 (bons)   : {sum(1 for s in scores_B if s > 0.55)}/{len(scores_B)}")
print(f"   Scores 0.45-0.55 (ok)  : {sum(1 for s in scores_B if 0.45 <= s <= 0.55)}/{len(scores_B)}")
print(f"   Scores < 0.45 (ruins)  : {sum(1 for s in scores_B if s < 0.45)}/{len(scores_B)}")

resultados_B = {
    "nome"          : "B - 512 + Overlap 128",
    "pdf"           : "Zambrosi 2017 - Phi foliar",
    "modelo"        : "Qwen3-Embedding-0.6B",
    "chunk_size"    : CHUNK_SIZE,
    "overlap"       : OVERLAP,
    "total_chunks"  : len(chunks),
    "menor_chunk"   : min(len(c) for c in chunks),
    "maior_chunk"   : max(len(c) for c in chunks),
    "media_chunk"   : sum(len(c) for c in chunks) // len(chunks),
    "score_tamanho" : score_tam,
    "scores"        : scores_B,
    "media_score"   : media_score
}

with open("/content/resultado_zambrosi_B.json", "w") as f:
    json.dump(resultados_B, f, ensure_ascii=False, indent=2)

print(f"\n✅ TESTE B CONCLUÍDO!")
print(f"   Resultados salvos em /content/resultado_zambrosi_B.json")
print(f"   Rode agora o teste_C_zambrosi.py!")
