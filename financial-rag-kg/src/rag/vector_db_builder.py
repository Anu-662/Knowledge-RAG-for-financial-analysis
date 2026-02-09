"""
Vector Database Builder using FAISS - Phase 3 Step 1
Creates embeddings from article chunks for semantic search
Compatible with Python 3.14
"""

import os
import json
import pickle
from pathlib import Path
from typing import List, Dict
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import time

# Install: pip install faiss-cpu numpy
try:
    import faiss
except ImportError:
    print("❌ FAISS not installed. Run: pip install faiss-cpu")
    exit(1)

load_dotenv()


class VectorDatabaseBuilder:
    """Build vector database from article chunks using FAISS"""
    
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
        self.processed_dir = data_dir / "processed"
        self.vector_db_dir = data_dir / "vector_db"
        self.vector_db_dir.mkdir(exist_ok=True)
        
        # OpenAI setup
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file")
        
        self.client = OpenAI(api_key=api_key)
        
        # Embedding dimension for text-embedding-3-small
        self.embedding_dim = 1536
        
        # Stats
        self.total_cost = 0.0
        self.total_tokens = 0
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding from OpenAI"""
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            
            # Track costs ($0.02/1M tokens)
            tokens = response.usage.total_tokens
            self.total_tokens += tokens
            self.total_cost += (tokens / 1_000_000) * 0.02
            
            return response.data[0].embedding
            
        except Exception as e:
            print(f"  ❌ Error getting embedding: {str(e)[:100]}")
            return None
    
    def create_vector_database(self):
        """Create FAISS vector database from all article chunks"""
        
        print("🔮 VECTOR DATABASE BUILDER - PHASE 3 (FAISS)")
        print("="*60)
        
        # Load all articles
        article_files = sorted(list(self.processed_dir.glob("*.json")))
        total = len(article_files)
        
        print(f"📊 Processing {total} articles...")
        print(f"💰 Estimated cost: ${(total * 0.001):.3f}\n")
        
        # Storage for embeddings and metadata
        embeddings_list = []
        metadata_list = []
        documents_list = []
        
        chunk_id = 0
        processed_articles = 0
        
        for i, article_file in enumerate(article_files, 1):
            try:
                with open(article_file) as f:
                    article = json.load(f)
                
                # Get chunks
                chunks = article.get("chunks", [])
                
                if not chunks:
                    continue
                
                # Process each chunk
                for chunk_idx, chunk_text in enumerate(chunks):
                    if len(chunk_text) < 50:
                        continue
                    
                    # Get embedding
                    embedding = self.get_embedding(chunk_text)
                    
                    if embedding is None:
                        continue
                    
                    # Store
                    embeddings_list.append(embedding)
                    documents_list.append(chunk_text)
                    metadata_list.append({
                        "chunk_id": chunk_id,
                        "article_id": article["id"],
                        "article_title": article["title"],
                        "source": article["source"],
                        "sector": article.get("sector", "General"),
                        "sentiment_label": article["sentiment"]["label"],
                        "sentiment_polarity": article["sentiment"]["polarity"],
                        "published_at": article.get("published_at", ""),
                        "chunk_index": chunk_idx,
                        "url": article.get("url", "")
                    })
                    
                    chunk_id += 1
                    
                    # Rate limiting
                    time.sleep(0.05)
                
                processed_articles += 1
                
                if processed_articles % 50 == 0:
                    print(f"  ✓ Processed {processed_articles}/{total} articles | "
                          f"Chunks: {chunk_id} | Cost: ${self.total_cost:.4f}")
                
            except Exception as e:
                print(f"  ❌ Error processing {article_file.name}: {str(e)[:100]}")
                continue
        
        # Create FAISS index
        print(f"\n🔧 Building FAISS index...")
        
        embeddings_array = np.array(embeddings_list).astype('float32')
        
        # Use IndexFlatL2 for exact search (good for <1M vectors)
        index = faiss.IndexFlatL2(self.embedding_dim)
        index.add(embeddings_array)
        
        # Save FAISS index
        index_file = self.vector_db_dir / "faiss_index.bin"
        faiss.write_index(index, str(index_file))
        
        # Save metadata and documents
        metadata_file = self.vector_db_dir / "metadata.pkl"
        with open(metadata_file, 'wb') as f:
            pickle.dump({
                "metadata": metadata_list,
                "documents": documents_list
            }, f)
        
        # Final summary
        print(f"\n{'='*60}")
        print(f"✅ VECTOR DATABASE CREATED!")
        print(f"{'='*60}")
        print(f"📊 Articles processed: {processed_articles}/{total}")
        print(f"📝 Total chunks embedded: {chunk_id}")
        print(f"💰 Total cost: ${self.total_cost:.4f}")
        print(f"🔢 Total tokens: {self.total_tokens:,}")
        print(f"📁 Database location: {self.vector_db_dir}")
        print(f"📦 FAISS index: {index_file}")
        print(f"📋 Metadata: {metadata_file}")
        
        # Save stats
        stats = {
            "total_articles": processed_articles,
            "total_chunks": chunk_id,
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost, 4),
            "embedding_model": "text-embedding-3-small",
            "embedding_dimension": self.embedding_dim,
            "index_type": "FAISS IndexFlatL2"
        }
        
        stats_file = self.vector_db_dir / "embedding_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\n✅ Ready for hybrid retrieval!")
        
        return index, metadata_list, documents_list


if __name__ == "__main__":
    builder = VectorDatabaseBuilder()
    builder.create_vector_database()
