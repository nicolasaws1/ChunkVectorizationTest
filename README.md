# Experimentos de Chunking para PDFs Científicos Agrícolas

Experimentos para avaliar estratégias de chunking em PDFs científicos agrícolas para sistemas RAG. Três abordagens foram testadas e comparadas usando FAISS para busca vetorial local e Qwen3-Embedding-0.6B para embeddings semânticos.

---

## O que é Chunking?

Chunking é o processo de dividir um texto longo em pedaços menores (chunks) para vetorização. O tamanho e a forma como esses pedaços são criados impacta diretamente a qualidade da busca semântica em sistemas RAG.

---

## PDFs Utilizados nos Testes

### Alva et al. (2005) — PDF Principal
> ALVA, A. K. et al. **Nitrogen and Irrigation Management Practices to Improve Nitrogen Uptake Efficiency and Minimize Leaching Losses**. Journal of Crop Improvement, v. 15, n. 2, p. 369-420, 2005.

**Por que escolhemos esse PDF:**
Este artigo foi selecionado como PDF principal dos experimentos por ser um documento **longo (52 páginas)** com **arquitetura simples** — texto corrido, poucas tabelas e seções bem definidas. O objetivo era entender como cada estratégia de chunking se comporta em um documento extenso antes de avançar para PDFs com estruturas mais complexas. Por ser um artigo de revisão sobre nitrogênio e irrigação em citros, também é diretamente relevante para o domínio agrícola do projeto.

**Características do documento:**
- 52 páginas
- Texto predominantemente corrido
- Poucas tabelas simples
- Seções bem definidas (Introdução, Métodos, Resultados, Conclusão, Referências)
- Idioma: inglês
- Área: manejo de nitrogênio e irrigação em citros

---

### Zambrosi et al. (2017) — PDF Complementar
> ZAMBROSI, F. C. B. et al. **Anatomical and ultrastructural damage to citrus leaves from phosphite spray depends on phosphorus supply to roots**. Plant and Soil, 2017.

**Por que escolhemos esse PDF:**
Utilizado como validação dos resultados obtidos com o PDF do Alva (2005). É um artigo **mais curto (13 páginas)** e com maior densidade de imagens e figuras microscópicas, permitindo observar o comportamento do chunking em um documento com características diferentes.

**Características do documento:**
- 13 páginas
- Muitas imagens e figuras microscópicas
- Tabelas com dados experimentais
- Seções bem definidas
- Idioma: inglês
- Área: fisiologia foliar de citros e fosfito

---

### Mattos Jr. et al. (2010) — PDF em Português
> MATTOS JR., D.; QUAGGIO, J. A.; BOARETTO, R. M. **Uso de elicitores para defesa em plantas cítricas**. Citrus Research & Technology, v. 31, n. 1, p. 65-74, 2010.

**Por que escolhemos esse PDF:**
Este artigo foi selecionado por ser um documento em **português**, permitindo avaliar o comportamento das estratégias de chunking quando a pergunta e o texto estão no mesmo idioma. É uma revisão de **10 páginas** com seções bem definidas e semanticamente ricas sobre elicitores e resistência de citros ao HLB, sendo diretamente relevante para o domínio agrícola do projeto.

**Características do documento:**
- 10 páginas
- Artigo de revisão com seções bem definidas
- Tabelas com dados sobre resistência de plantas
- Uma figura esquemática
- Idioma: português (com resumo em inglês)
- Área: elicitores e resistência sistêmica adquirida em citros

---

## Experimentos

### Teste A — Tamanho Fixo (1024 chars, sem overlap)
Divide o texto em blocos de até 1024 caracteres sem sobreposição entre chunks. É a estratégia mais simples e foi usada como baseline de comparação.

- ✅ Simples de implementar
- ❌ Chunks grandes demais para o modelo de embedding
- ❌ Perde informações nas fronteiras entre chunks
- ❌ Tabelas sem controle de tamanho

---

### Teste B — Tamanho Fixo com Overlap (512 chars + 128 overlap)
Divide em blocos de 512 caracteres repetindo os últimos 128 caracteres do bloco anterior no início do próximo. Garante que nenhuma informação fique perdida na fronteira entre dois chunks.

- ✅ Tamanho ideal para o Qwen3-Embedding (512 tokens)
- ✅ Overlap preserva contexto entre chunks
- ✅ ~70% dos chunks no tamanho ideal (300-800 chars)
- ✅ Melhor resultado em PDFs com seções fragmentadas

---

### Teste C — Por Seção do Artigo
Corta nos títulos `##` gerados pelo Docling, criando um chunk por seção do artigo científico (Abstract, Introdução, Métodos, Resultados, Discussão). Seções muito grandes são subdivididas mantendo o título como contexto.

- ✅ Preserva a estrutura do artigo científico
- ✅ Melhor resultado em PDFs com seções bem definidas
- ⚠️ Chunks de referências bibliográficas podem prejudicar a busca
- ⚠️ Chunks muito pequenos em seções curtas

---

## Resultados

### Zambrosi 2017 — PDF curto em inglês com seções fragmentadas
Perguntas em português sobre texto em inglês (busca cross-lingual).

| Configuração | Score Busca | Score Tamanho | Score Final |
|---|---|---|---|
| **B — 512 + Overlap 128** ✅ | **0.5815** | **70.6%** | **0.6315** |
| C — Por Seção | 0.5782 | 52.9% | 0.5583 |
| A — Baseline 1024 | 0.5708 | 21.4% | 0.4282 |

---

### Mattos 2010 — PDF curto em português com seções bem definidas
Perguntas em português sobre texto em português (busca no mesmo idioma).

| Configuração | Score Busca | Score Tamanho | Score Final |
|---|---|---|---|
| **C — Por Seção** ✅ | **0.6045** | **70.3%** | **0.6440** |
| B — 512 + Overlap 128 | 0.5936 | 70.2% | 0.6370 |
| A — Baseline 1024 | 0.5875 | 28.3% | 0.4655 |

> Score Final = Score Busca (60%) + Score Tamanho (40%)

---

## Conclusões

Os experimentos revelaram duas descobertas importantes:

**1. A melhor estratégia depende da estrutura do PDF**

| Tipo de PDF | Melhor estratégia | Motivo |
|---|---|---|
| Seções bem definidas e semânticas | C — Por Seção | Cada seção trata de um único assunto |
| Seções fragmentadas ou curtas | B — 512 + Overlap | Overlap preserva contexto entre chunks |

**2. Mesmo idioma melhora os scores de busca**

Os scores do Mattos 2010 (português → português) foram consistentemente maiores que os do Zambrosi 2017 (português → inglês), confirmando que o Qwen3-Embedding performa melhor quando pergunta e texto estão no mesmo idioma, mesmo sendo um modelo multilingual.

```
Score médio Zambrosi (PT → EN): ~0.57
Score médio Mattos   (PT → PT): ~0.60
```

**3. O Teste A (Baseline) foi consistentemente o pior** nos dois PDFs por gerar chunks muito grandes que excedem o limite do modelo de embedding (512 tokens).

---

## Tecnologias

| Componente | Tecnologia |
|---|---|
| Extração de PDF | Docling + Tesseract OCR |
| Embedding | Qwen3-Embedding-0.6B (119 idiomas) |
| Busca vetorial local | FAISS |
| Ambiente | Google Colab |

---

## Estrutura

```
├── README.md
├── requirements.txt
├── .gitignore
├── teste_A_baseline.py     ← chunk 1024 sem overlap
├── teste_B_overlap.py      ← chunk 512 + overlap 128
├── teste_C_secao.py        ← chunk por seção do artigo
└── comparativo.py          ← ranking final dos 3 testes
```

---

## Como Rodar

Os testes foram desenvolvidos para rodar no **Google Colab** com acesso ao Google Drive.

### Instalação

```bash
pip install docling pytesseract sentence-transformers faiss-cpu
apt-get install tesseract-ocr tesseract-ocr-por tesseract-ocr-eng
```

### Ordem de execução

```
1. teste_A_baseline.py  → gera resultado_teste_A.json
2. teste_B_overlap.py   → gera resultado_teste_B.json
3. teste_C_secao.py     → gera resultado_teste_C.json
4. comparativo.py       → ranking final
```

### Configuração

Ajuste no início de cada arquivo:

```python
PDF_TESTE = "/content/drive/MyDrive/seu_arquivo.pdf"
BASE_DIR  = Path("/content/drive/MyDrive/sua_pasta")
OCR_LANG  = ["eng"]  # ou ["por"] para PDFs em português

# Ajuste também as perguntas de teste para o seu PDF
perguntas = [
    "Sua pergunta 1?",
    "Sua pergunta 2?",
]
```

---

## Próximos Passos

| Etapa | PDF | Idioma | Status |
|---|---|---|---|
| ✅ Concluído | Alva 2005 — longo, arquitetura simples | Inglês | Validação inicial |
| ✅ Concluído | Zambrosi 2017 — curto, muitas imagens | Inglês | Validação cross-lingual |
| ✅ Concluído | Mattos 2010 — curto, seções bem definidas | Português | Validação mesmo idioma |
| 🔄 Próximo | PDF médio (~20-30 páginas) com muitas tabelas | A definir | Avaliar chunking tabular |
| 🔄 Futuro | PDF longo em português | Português | Validar OCR PT em doc grande |

O objetivo final é encontrar uma estratégia de chunking robusta que funcione bem para diferentes tipos de documentos científicos agrícolas antes de aplicar em produção com mais de 1000 PDFs.

---

## Observações

- Os PDFs científicos **não estão incluídos** por questões de direitos autorais
- Os testes rodam **100% localmente** no Colab sem necessidade de banco vetorial externo
- O modelo Qwen3-Embedding-0.6B suporta **busca cross-lingual** — perguntas em português encontram textos em inglês, mas scores são maiores quando ambos estão no mesmo idioma
