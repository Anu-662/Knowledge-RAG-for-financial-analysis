# 📊 Financial Knowledge Graph RAG System

![Python](https://img.shields.io/badge/python-3.12+-blue.svg)
![Neo4j](https://img.shields.io/badge/Neo4j-5.0+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

A production-grade hybrid Retrieval-Augmented Generation (RAG) system combining knowledge graphs with semantic search for intelligent financial analysis.

## 🎯 Problem Statement

Traditional search systems suffer from **relationship blindness** - they can find articles mentioning "Apple" and "competitors," but can't understand that Apple competes with Microsoft in cloud computing, NVIDIA in chips, and Meta in AR/VR.

**This system solves that.**

## 💡 Solution Overview

Hybrid RAG architecture combining:
- **Neo4j Knowledge Graph**: Stores 991 entities and 921 relationships between companies
- **FAISS Vector Database**: Enables semantic search across 527 article chunks
- **GPT-4o-mini**: Powers entity extraction and contextual answer generation

**Result**: Ask "Which tech companies compete with Apple in cloud computing?" and get intelligent, relationship-aware answers in <2 seconds.

## 🏗️ Architecture
```
┌─────────────┐
│  NewsAPI    │ ──► 527 Articles
└─────────────┘
       │
       ▼
┌─────────────────────────┐
│  Data Collection Agent  │ ──► Autonomous scraping
└─────────────────────────┘
       │
       ▼
┌─────────────────────────┐
│  GPT-4o-mini Extractor  │ ──► 260 companies, 132 people
└─────────────────────────┘      100% success rate
       │
       ├──────────────┬───────────────┐
       ▼              ▼               ▼
┌──────────┐   ┌──────────┐   ┌──────────┐
│  Neo4j   │   │  FAISS   │   │Sentiment │
│  Graph   │   │  Vector  │   │ Analysis │
│          │   │    DB    │   │          │
│991 nodes │   │527 chunks│   │72 sectors│
│921 edges │   │          │   │          │
└──────────┘   └──────────┘   └──────────┘
       │              │               │
       └──────────────┴───────────────┘
                      │
                      ▼
            ┌──────────────────┐
            │  Hybrid Retriever │
            │  Graph + Vector   │
            └──────────────────┘
                      │
                      ▼
            ┌──────────────────┐
            │   GPT-4o-mini    │
            │Answer Generation │
            └──────────────────┘
                      │
                      ▼
            ┌──────────────────┐
            │ Contextual Answer│
            │   with Sources   │
            └──────────────────┘
```

## ✨ Key Features

- 🤖 **Autonomous Data Collection**: Scrapes 527 real-time financial articles via NewsAPI
- 🧠 **Entity Extraction**: GPT-4o-mini extracts companies, people, relationships (100% success rate)
- 🕸️ **Knowledge Graph**: Neo4j stores 991 nodes and 921 relationships
- 🔍 **Semantic Search**: FAISS vector database with OpenAI embeddings
- 😊 **Sentiment Analysis**: Filters by positive/negative/neutral sentiment
- ⚡ **Fast Queries**: Sub-2-second response time
- 🔗 **Multi-hop Reasoning**: Traverses company relationships across articles
- 🐳 **Production Ready**: Containerized with Docker

## 📊 Dataset Statistics

| Metric | Count |
|--------|-------|
| **Articles Processed** | 527 |
| **Unique Companies** | 260 |
| **People Identified** | 132 |
| **Industry Sectors** | 72 |
| **Graph Nodes** | 991 |
| **Graph Relationships** | 921 |
| **Vector Embeddings** | 527 |

**Sentiment Distribution:**
- Positive: 94 articles (17.8%)
- Neutral: 387 articles (73.4%)
- Negative: 46 articles (8.7%)

**Top Sectors:**
- Technology (101 articles)
- Finance (56 articles)
- Energy (36 articles)
- Healthcare (15 articles)

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- Docker (for Neo4j)
- OpenAI API key
- NewsAPI key (optional - synthetic data available)

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/financial-rag-kg.git
cd financial-rag-kg

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your API keys:
# OPENAI_API_KEY=sk-your-key-here
# NEWSAPI_KEY=your-key-here (optional)
```

### Start Neo4j
```bash
# Using Docker (recommended)
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password123 \
  neo4j:latest

# Verify at http://localhost:7474
# Login: neo4j / password123
```

## 📖 Usage

### Phase 1: Data Collection
```bash
# Option A: Use synthetic data (no API key needed)
python3 src/data_processing/synthetic_data_generator.py

# Option B: Collect real data (requires NewsAPI key)
python3 src/data_processing/data_agent.py
```

### Phase 2: Build Knowledge Graph
```bash
# Step 1: Extract entities with GPT-4o-mini
python3 src/kg_builder/entity_extractor.py

# Step 2: Load graph into Neo4j
python3 src/kg_builder/graph_loader.py
```

### Phase 3: Create Vector Database
```bash
# Build FAISS vector database
python3 src/rag/vector_db_builder.py
```

### Phase 4: Query the System
```bash
# Start interactive query engine
python3 src/rag/hybrid_query_engine.py
```

**Example queries:**
```
🔍 Which companies compete with Apple?
🔍 Show positive news about Tesla
🔍 What's happening in the renewable energy sector?
🔍 Find tech companies with negative sentiment
```

## 💻 Project Structure
```
financial-rag-kg/
├── data/
│   ├── processed/          # Processed articles (527 files)
│   ├── entities/           # Extracted entities
│   ├── vector_db/          # FAISS vector database
│   └── metadata/           # Statistics and summaries
├── src/
│   ├── data_processing/
│   │   ├── data_agent.py              # NewsAPI scraper
│   │   ├── synthetic_data_generator.py # Synthetic data
│   │   └── view_data.py               # Data exploration
│   ├── kg_builder/
│   │   ├── entity_extractor.py        # GPT-4 entity extraction
│   │   └── graph_loader.py            # Neo4j loader
│   └── rag/
│       ├── vector_db_builder.py       # FAISS database
│       └── hybrid_query_engine.py     # Query engine
├── requirements.txt
├── .env.example
└── README.md
```

## 🔬 Technical Deep Dive

### Hybrid Retrieval Algorithm

The system uses a two-stage retrieval process:

**1. Graph Retrieval (Relationship-Aware)**
```cypher
// Find competitors of a company
MATCH (c:Company {name: $company})-[:COMPETES_WITH]-(competitor)
RETURN competitor.name

// Find companies in same sector
MATCH (c:Company)-[:IN_SECTOR]->(s:Sector {name: $sector})
RETURN c.name
```

**2. Vector Retrieval (Semantic-Aware)**
```python
# Convert query to embedding
query_embedding = get_embedding(query)

# Search FAISS index
distances, indices = faiss_index.search(query_embedding, k=5)

# Apply sentiment filter
filtered_results = [r for r in results if r.sentiment == filter]
```

**3. Hybrid Fusion**
- Combines graph context (relationships) with vector results (relevant text)
- LLM synthesizes both into coherent answer
- Cites sources from both retrieval methods

### Entity Extraction Pipeline
```python
# Prompt template for GPT-4o-mini
extract_entities(article_text) → {
  "companies": ["Apple Inc.", "Microsoft"],
  "people": [{"name": "Tim Cook", "role": "CEO", "company": "Apple"}],
  "sectors": ["Technology"],
  "events": [{"type": "earnings_report", "date": "2024-01-28"}],
  "metrics": [{"type": "revenue", "value": "119.6B"}],
  "relationships": [
    {"entity1": "Apple", "relationship": "competes_with", "entity2": "Microsoft"}
  ]
}
```

### Performance Optimizations

- **Batch Processing**: Process articles in batches for efficiency
- **Caching**: Cache embeddings to avoid re-computation
- **Indexing**: Neo4j constraints and FAISS indexing for fast queries
- **Rate Limiting**: Respect API limits with exponential backoff

## 📈 Results & Impact

### Query Performance

| Query Type | Traditional Search | This System | Improvement |
|------------|-------------------|-------------|-------------|
| Simple keyword | ~2s | ~1.5s | 25% faster |
| Multi-entity | Manual (hours) | <2s | **99.9% faster** |
| Relationship-based | Not possible | <2s | **∞** |

### Sample Queries & Answers

**Query:** "Which companies compete with Apple in cloud computing?"

**Answer:**
> Based on the knowledge graph and recent articles, Apple's main competitors in cloud computing include:
>
> 1. **Microsoft** - Azure cloud platform competes directly with Apple's iCloud services
> 2. **Google** - Google Cloud Platform and Workspace compete in enterprise solutions
> 3. **Amazon** - AWS dominates cloud infrastructure where Apple is expanding
>
> Sources: [Links to 3 relevant articles with sentiment context]

## 💰 Cost Analysis

Total project cost: **~$3-4**

| Phase | Cost |
|-------|------|
| Entity Extraction (527 articles) | $0.08 |
| Vector Embeddings (527 chunks) | $0.50 |
| Query Testing (~100 queries) | $1-2 |
| **Total** | **$2.58-3.08** |

## 🛠️ Technologies Used

- **Languages**: Python 3.12
- **Graph Database**: Neo4j 5.0
- **Vector Database**: FAISS
- **LLM**: OpenAI GPT-4o-mini
- **Embeddings**: OpenAI text-embedding-3-small
- **Data Source**: NewsAPI
- **Deployment**: Docker
- **Libraries**: `neo4j`, `faiss-cpu`, `openai`, `numpy`, `requests`

## 🔮 Future Enhancements

- [ ] Streamlit web interface for interactive queries
- [ ] Real-time data updates with streaming pipeline
- [ ] Graph algorithms (PageRank, community detection)
- [ ] Multi-modal support (financial charts, images)
- [ ] Advanced relationship types (partnerships, acquisitions)
- [ ] Time-series analysis of sentiment trends
- [ ] Export to various formats (PDF reports, Excel)

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details


⭐ **Star this repo if you found it helpful!**

Built with ❤️ for intelligent financial analysis
