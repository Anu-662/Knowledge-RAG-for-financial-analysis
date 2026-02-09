"""
Hybrid Query Engine - Phase 3 Final Step
Combines Neo4j graph queries with vector search for intelligent retrieval
"""

import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import faiss
from openai import OpenAI
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()


class HybridQueryEngine:
    """Intelligent query engine combining graph and vector search"""
    
    def __init__(self, data_dir: str = None):
        # Auto-detect project root
        if data_dir is None:
            script_dir = Path(__file__).parent
            if script_dir.name == "rag":
                project_root = script_dir.parent.parent
            else:
                project_root = Path.cwd()
            data_dir = project_root / "data"
        else:
            data_dir = Path(data_dir)
        
        self.data_dir = data_dir
        self.vector_db_dir = data_dir / "vector_db"
        
        # OpenAI setup
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found")
        
        self.client = OpenAI(api_key=api_key)
        
        # Load FAISS index
        print(" Loading vector database...")
        index_file = self.vector_db_dir / "faiss_index.bin"
        metadata_file = self.vector_db_dir / "metadata.pkl"
        
        if not index_file.exists():
            raise FileNotFoundError(f"FAISS index not found. Run vector_db_builder.py first!")
        
        self.index = faiss.read_index(str(index_file))
        
        with open(metadata_file, 'rb') as f:
            data = pickle.load(f)
            self.metadata = data["metadata"]
            self.documents = data["documents"]
        
        print(f"Loaded {len(self.documents)} chunks")
        
        # Neo4j setup
        neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        neo4j_user = os.getenv("NEO4J_USERNAME", "neo4j")
        neo4j_pass = os.getenv("NEO4J_PASSWORD", "password123")
        
        print(f"Connecting to Neo4j...")
        self.neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pass))
        print("Connected to Neo4j")
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for query"""
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return np.array(response.data[0].embedding).astype('float32')
    
    def vector_search(self, query: str, top_k: int = 5, 
                     sentiment_filter: str = None) -> List[Dict]:
        """Search using vector similarity"""
        
        # Get query embedding
        query_embedding = self.get_embedding(query).reshape(1, -1)
        
        # Search FAISS
        distances, indices = self.index.search(query_embedding, top_k * 3)  # Get more for filtering
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= len(self.metadata):
                continue
            
            meta = self.metadata[idx]
            
            # Apply sentiment filter
            if sentiment_filter and meta["sentiment_label"] != sentiment_filter:
                continue
            
            results.append({
                "text": self.documents[idx],
                "score": float(1 / (1 + dist)),  # Convert distance to similarity
                "metadata": meta
            })
            
            if len(results) >= top_k:
                break
        
        return results
    
    def graph_search(self, query: str) -> Dict:
        """Search Neo4j graph for entity relationships"""
        
        # Extract potential company names (simple approach)
        # In production, use NER
        companies = self._extract_companies(query)
        
        graph_context = {
            "companies_mentioned": [],
            "relationships": [],
            "sectors": []
        }
        
        with self.neo4j_driver.session() as session:
            # Find companies in graph
            for company in companies:
                result = session.run("""
                    MATCH (c:Company)
                    WHERE toLower(c.name) CONTAINS toLower($company)
                    RETURN c.name as name, c.sector as sector
                    LIMIT 5
                """, company=company)
                
                for record in result:
                    graph_context["companies_mentioned"].append({
                        "name": record["name"],
                        "sector": record["sector"]
                    })
            
            # Get relationships for found companies
            if graph_context["companies_mentioned"]:
                company_name = graph_context["companies_mentioned"][0]["name"]
                
                # Find competitors
                result = session.run("""
                    MATCH (c:Company {name: $name})-[:COMPETES_WITH]-(competitor)
                    RETURN competitor.name as competitor
                    LIMIT 5
                """, name=company_name)
                
                for record in result:
                    graph_context["relationships"].append({
                        "type": "competitor",
                        "entity": record["competitor"]
                    })
                
                # Find co-mentioned companies
                result = session.run("""
                    MATCH (c:Company {name: $name})-[r:MENTIONED_WITH]-(other)
                    RETURN other.name as other, r.co_mentions as count
                    ORDER BY count DESC
                    LIMIT 5
                """, name=company_name)
                
                for record in result:
                    graph_context["relationships"].append({
                        "type": "mentioned_with",
                        "entity": record["other"],
                        "count": record["count"]
                    })
                
                # Get sector info
                result = session.run("""
                    MATCH (c:Company {name: $name})-[:IN_SECTOR]->(s:Sector)
                    RETURN s.name as sector
                """, name=company_name)
                
                for record in result:
                    graph_context["sectors"].append(record["sector"])
        
        return graph_context
    
    def _extract_companies(self, query: str) -> List[str]:
        """Simple company extraction from query"""
        # Common company names (expand this list)
        known_companies = [
            "Apple", "Microsoft", "Google", "Amazon", "Tesla", "Meta",
            "NVIDIA", "Intel", "AMD", "IBM", "Oracle", "Salesforce",
            "JPMorgan", "Goldman Sachs", "Morgan Stanley", "Citigroup",
            "Pfizer", "Moderna", "Johnson & Johnson",
            "ExxonMobil", "Chevron", "Walmart", "Target", "Nike"
        ]
        
        query_lower = query.lower()
        found = []
        
        for company in known_companies:
            if company.lower() in query_lower:
                found.append(company)
        
        return found
    
    def hybrid_query(self, query: str, top_k: int = 5, 
                    sentiment_filter: str = None) -> Dict:
        """Combine graph and vector search"""
        
        print(f"\n🔍 Query: {query}")
        print("="*60)
        
        # 1. Vector search
        print("📊 Searching vector database...")
        vector_results = self.vector_search(query, top_k, sentiment_filter)
        print(f"✓ Found {len(vector_results)} relevant chunks")
        
        # 2. Graph search
        print("🕸️  Searching knowledge graph...")
        graph_context = self.graph_search(query)
        print(f"✓ Found {len(graph_context['companies_mentioned'])} companies, "
              f"{len(graph_context['relationships'])} relationships")
        
        # 3. Combine results
        combined = {
            "query": query,
            "vector_results": vector_results,
            "graph_context": graph_context,
            "top_chunks": [r["text"] for r in vector_results[:3]]
        }
        
        return combined
    
    def generate_answer(self, query: str, top_k: int = 5,
                       sentiment_filter: str = None) -> str:
        """Generate final answer using LLM"""
        
        # Get hybrid results
        results = self.hybrid_query(query, top_k, sentiment_filter)
        
        # Build context
        context_parts = []
        
        # Add graph context
        if results["graph_context"]["companies_mentioned"]:
            context_parts.append("## Companies Mentioned:")
            for comp in results["graph_context"]["companies_mentioned"]:
                context_parts.append(f"- {comp['name']} (Sector: {comp['sector']})")
        
        if results["graph_context"]["relationships"]:
            context_parts.append("\n## Related Companies:")
            for rel in results["graph_context"]["relationships"]:
                context_parts.append(f"- {rel['entity']} ({rel['type']})")
        
        # Add vector results
        context_parts.append("\n## Relevant Article Excerpts:")
        for i, chunk in enumerate(results["top_chunks"], 1):
            meta = results["vector_results"][i-1]["metadata"]
            context_parts.append(f"\n[{i}] {meta['article_title']}")
            context_parts.append(f"Source: {meta['source']} | Sentiment: {meta['sentiment_label']}")
            context_parts.append(f"{chunk}\n")
        
        context = "\n".join(context_parts)
        
        # Generate answer
        print("\n🤖 Generating answer with GPT-4o-mini...")
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a financial analyst assistant. Answer questions based on the provided context from news articles and company relationships. Be concise and cite sources."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
                }
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        answer = response.choices[0].message.content
        
        return answer, results
    
    def close(self):
        """Clean up connections"""
        self.neo4j_driver.close()


def main():
    """Interactive query interface"""
    
    print("HYBRID KNOWLEDGE GRAPH RAG SYSTEM")
    print("="*60)
    
    engine = HybridQueryEngine()
    
    print("\n Example queries:")
    print("  - Which companies compete with Apple?")
    print("  - Show positive news about Tesla")
    print("  - What's happening in the technology sector?")
    print("  - Tell me about renewable energy companies")
    
    try:
        while True:
            print("\n" + "="*60)
            query = input("\n🔍 Enter your question (or 'quit' to exit): ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query:
                continue
            
            # Check for sentiment keywords
            sentiment_filter = None
            if "positive" in query.lower() or "good news" in query.lower():
                sentiment_filter = "positive"
            elif "negative" in query.lower() or "concerns" in query.lower():
                sentiment_filter = "negative"
            
            # Generate answer
            answer, results = engine.generate_answer(query, sentiment_filter=sentiment_filter)
            
            print("\n📝 ANSWER:")
            print("="*60)
            print(answer)
            print("="*60)
            
            # Show sources
            print("\nSources:")
            for i, result in enumerate(results["vector_results"][:3], 1):
                meta = result["metadata"]
                print(f"{i}. {meta['article_title']} ({meta['source']})")
    
    finally:
        engine.close()
        print("\n Goodbye!")


if __name__ == "__main__":
    main()
