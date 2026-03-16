# ============================================================
# TESTE A — BASELINE (chunk_size=1024, overlap=0)
# 100% LOCAL — sem Qdrant, usando FAISS
# PDF: Zambrosi 2017 - Phi foliar
# ============================================================

!pip install -q docling pytesseract sentence-transformers faiss-cpu
!sudo apt-get install -y -qq tesseract-ocr tesseract-ocr-por tesseract-ocr-eng

import os
import torch
import faiss
import numpy as np
import json
import statistics
from pathlib import Path
from sentence_transformers import SentenceTransformer
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import ImageRefMode, PictureItem

from google.colab import drive
drive.mount('/content/drive', force_remount=False)

# ============================================================
# CONFIGURAÇÃO DO TESSERACT
# ============================================================
candidates = [
    "/usr/share/tesseract-ocr/5/tessdata",
    "/usr/share/tesseract-ocr/4.00/tessdata",
    "/usr/share/tesseract-ocr/tessdata",
    "/usr/share/tessdata",
]

tessdata = next((p for p in candidates if os.path.isdir(p)), None)
if tessdata is None:
    tessdata = "/usr/share/tesseract-ocr/5/tessdata"
    os.makedirs(tessdata, exist_ok=True)

os.environ["TESSDATA_PREFIX"] = tessdata
print(f"TESSDATA_PREFIX → {tessdata}")

needed  = ["osd.traineddata", "por.traineddata", "eng.traineddata"]
missing = [f for f in needed if not os.path.isfile(os.path.join(tessdata, f))]

if missing:
    print(f"⚠️  Instalando arquivos faltando: {missing}")
    os.system("apt-get install -y -qq tesseract-ocr-por tesseract-ocr-eng 2>&1")
    tessdata = next((p for p in candidates if os.path.isdir(p)), tessdata)
    os.environ["TESSDATA_PREFIX"] = tessdata
    print("✅ Tesseract configurado!")
else:
    print("✅ Tesseract OK!")

# ============================================================
# CONFIGURAÇÃO — PDF Zambrosi 2017
# ============================================================
PDF_TESTE  = "/content/drive/MyDrive/Squad2/leo_Squad2/data/Zambrosi(2017)-Phi foliar - Janaina Lais Pacheco Lara Morandin.pdf"
BASE_DIR   = Path("/content/drive/MyDrive/Squad2/leo_Squad2")
MD_DIR     = BASE_DIR / "testes" / "zambrosi" / "teste_A" / "md_images"
DATA_DIR   = BASE_DIR / "testes" / "zambrosi" / "teste_A" / "data_md"
CHUNK_SIZE = 1024
OVERLAP    = 0

for d in [MD_DIR, DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f"\n📋 TESTE A — Baseline")
print(f"   PDF        : Zambrosi 2017 - Phi foliar")
print(f"   chunk_size : {CHUNK_SIZE}")
print(f"   overlap    : {OVERLAP}")
print(f"   banco      : FAISS local (sem Qdrant)")

# ============================================================
# FUNÇÕES
# ============================================================
def clean_text(text):
    text = text.replace("glyph<c=3,font=/CIDFont+F5>", " ")
    text = text.replace("glyph<c=3,font=/CIDFont+F8>", " ")
    text = text.replace("&gt;", "").replace("&lt;", "")
    return text

def pdf_to_markdown(file_path):
    base_stem = Path(file_path).stem
    md_path   = DATA_DIR / f"{base_stem}.md"

    if md_path.exists():
        print(f"ℹ️  Markdown já existe, reutilizando!")
        return str(md_path)

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True
    pipeline_options.ocr_options = TesseractCliOcrOptions(lang=["eng"], force_full_page_ocr=True)
    pipeline_options.generate_picture_images = True

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )

    print(f"🔄 Convertendo PDF...")
    result = converter.convert(file_path).document

    for text in getattr(result, "texts", []):
        text.orig = clean_text(getattr(text, "orig", ""))
    for table in getattr(result, "tables", []):
        for cell in getattr(table.data, "table_cells", []):
            cell.text = clean_text(getattr(cell, "text", ""))

    picture_counter = 0
    for element, _ in result.iterate_items():
        if isinstance(element, PictureItem):
            picture_counter += 1
            img_path = MD_DIR / f"{base_stem}-picture-{picture_counter}.png"
            with img_path.open("wb") as fp:
                element.get_image(result).save(fp, "PNG")

    print(f"   {picture_counter} imagens salvas em {MD_DIR}")

    result_md = result.export_to_markdown()
    with md_path.open("w", encoding="utf-8") as f:
        f.write(result_md)

    print(f"✅ Markdown salvo!")
    return str(md_path)

def fazer_chunks_A(md_path):
    """TESTE A: chunk fixo 1024 sem overlap"""
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.readlines()

    content = [p.strip() for p in content]
    content = [p for p in content if p.replace('-','').replace('|','').replace(' ','').strip()]

    chunks        = []
    current_chunk = ""

    for i, paragraph in enumerate(content):
        if i == 0:
            current_chunk += paragraph
            continue

        if paragraph.startswith("| ") and paragraph.endswith(" |"):
            current_chunk += "\n" + paragraph
            continue

        if len(current_chunk) + len(paragraph) + 1 > CHUNK_SIZE:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = paragraph
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
md_path      = pdf_to_markdown(PDF_TESTE)
chunks       = fazer_chunks_A(md_path)
score_tam    = avaliar_tamanho_chunks(chunks, "TESTE A — Baseline 1024")
index_A      = criar_indice_faiss(chunks)

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
print(f"🔍 RESULTADOS DAS BUSCAS — TESTE A (Baseline 1024)")
print(f"   Perguntas em português | Texto em inglês")
print(f"   Modelo: Qwen3-Embedding-0.6B (multilingual)")
print(f"{'='*60}")

scores_A = []
for pergunta in perguntas:
    r = buscar_faiss(pergunta, index_A, chunks)[0]
    scores_A.append(r['score'])
    relevante = "✅" if r['score'] > 0.55 else "⚠️" if r['score'] > 0.45 else "❌"
    print(f"\n❓ {pergunta}")
    print(f"   Score : {r['score']:.4f} {relevante}")
    print(f"   Chars : {r['chars']}")
    print(f"   Texto : {r['chunk'][:200]}...")
    print("-"*60)

media_score = sum(scores_A)/len(scores_A)
print(f"\n📈 MÉDIA DE SCORE TESTE A : {media_score:.4f}")
print(f"   Score tamanho          : {score_tam:.1f}%")
print(f"   Scores > 0.55 (bons)   : {sum(1 for s in scores_A if s > 0.55)}/{len(scores_A)}")
print(f"   Scores 0.45-0.55 (ok)  : {sum(1 for s in scores_A if 0.45 <= s <= 0.55)}/{len(scores_A)}")
print(f"   Scores < 0.45 (ruins)  : {sum(1 for s in scores_A if s < 0.45)}/{len(scores_A)}")

# Salva resultados
resultados_A = {
    "nome"          : "A - Baseline 1024",
    "pdf"           : "Zambrosi 2017 - Phi foliar",
    "modelo"        : "Qwen3-Embedding-0.6B",
    "chunk_size"    : CHUNK_SIZE,
    "overlap"       : OVERLAP,
    "total_chunks"  : len(chunks),
    "menor_chunk"   : min(len(c) for c in chunks),
    "maior_chunk"   : max(len(c) for c in chunks),
    "media_chunk"   : sum(len(c) for c in chunks) // len(chunks),
    "score_tamanho" : score_tam,
    "scores"        : scores_A,
    "media_score"   : media_score
}

with open("/content/resultado_zambrosi_A.json", "w") as f:
    json.dump(resultados_A, f, ensure_ascii=False, indent=2)

print(f"\n✅ TESTE A CONCLUÍDO!")
print(f"   Resultados salvos em /content/resultado_zambrosi_A.json")
print(f"   Rode agora o teste B com o mesmo PDF!")
