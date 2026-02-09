"""
Autonomous Financial Data Collection Agent
Scrapes, cleans, chunks, and extracts metadata from financial articles
"""

import os
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any
import requests
from pathlib import Path
import hashlib
from dotenv import load_dotenv
load_dotenv()





class FinancialDataAgent:
    """Agent that autonomously collects and processes financial articles"""
    
    def __init__(self, output_dir: str = "data"):
        self.output_dir = Path(output_dir)
        self.raw_dir = self.output_dir / "raw"
        self.processed_dir = self.output_dir / "processed"
        self.metadata_dir = self.output_dir / "metadata"
        
        # Create directories
        for dir_path in [self.raw_dir, self.processed_dir, self.metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # NewsAPI configuration (you'll add your key)
        self.newsapi_key = os.getenv("NEWSAPI_KEY", "YOUR_KEY_HERE")
        self.newsapi_base = "https://newsapi.org/v2"
        
        # Financial keywords for targeted scraping
        self.financial_keywords = [
            "stock market", "earnings report", "Federal Reserve",
            "inflation", "interest rates", "merger acquisition",
            "IPO", "cryptocurrency", "S&P 500", "tech stocks",
            "banking", "semiconductor", "renewable energy"
        ]
        
        # Sector mapping
        self.sector_keywords = {
            "Technology": ["tech", "semiconductor", "AI", "software", "cloud"],
            "Finance": ["bank", "fintech", "insurance", "investment"],
            "Energy": ["oil", "renewable", "energy", "solar", "battery"],
            "Healthcare": ["pharma", "biotech", "healthcare", "medical"],
            "Consumer": ["retail", "consumer", "e-commerce"]
        }
        
    def scrape_newsapi(self, keyword: str, days_back: int = 30, max_articles: int = 100) -> List[Dict]:
        """Scrape articles from NewsAPI"""
        articles = []
        
        # Calculate date range
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days_back)
        
        url = f"{self.newsapi_base}/everything"
        params = {
            "q": keyword,
            "from": from_date.strftime("%Y-%m-%d"),
            "to": to_date.strftime("%Y-%m-%d"),
            "language": "en",
            "sortBy": "relevancy",
            "pageSize": min(max_articles, 100),
            "apiKey": self.newsapi_key
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get("status") == "ok":
                articles.extend(data.get("articles", []))
                print(f"✓ Scraped {len(data.get('articles', []))} articles for '{keyword}'")
            else:
                print(f"✗ NewsAPI error: {data.get('message', 'Unknown error')}")
                
        except Exception as e:
            print(f"✗ Error scraping '{keyword}': {str(e)}")
        
        time.sleep(1)  # Rate limiting
        return articles
    
    def scrape_all_keywords(self, max_per_keyword: int = 50) -> List[Dict]:
        """Scrape articles for all financial keywords"""
        all_articles = []
        
        print(f"\n🤖 Starting autonomous data collection...")
        print(f"📊 Target: {len(self.financial_keywords)} keywords, ~{len(self.financial_keywords) * max_per_keyword} articles\n")
        
        for keyword in self.financial_keywords:
            articles = self.scrape_newsapi(keyword, max_articles=max_per_keyword)
            all_articles.extend(articles)
        
        # Deduplicate by URL
        unique_articles = {art.get("url"): art for art in all_articles if art.get("url")}
        unique_articles = list(unique_articles.values())
        
        print(f"\n✓ Collected {len(unique_articles)} unique articles")
        return unique_articles
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = " ".join(text.split())
        
        # Remove common boilerplate
        boilerplate_phrases = [
            "Click here to subscribe",
            "Sign up for our newsletter",
            "Read more:",
            "Advertisement"
        ]
        for phrase in boilerplate_phrases:
            text = text.replace(phrase, "")
        
        return text.strip()
    
    def chunk_text(self, text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks (by tokens approximation)"""
        # Rough approximation: 1 token ≈ 4 characters
        char_chunk_size = chunk_size * 4
        char_overlap = overlap * 4
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + char_chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for period followed by space
                period_pos = text.rfind(". ", start, end)
                if period_pos > start:
                    end = period_pos + 1
            
            chunk = text[start:end].strip()
            if len(chunk) > 100:  # Only keep substantial chunks
                chunks.append(chunk)
            
            start = end - char_overlap
        
        return chunks
    
    def extract_sector(self, text: str, title: str) -> str:
        """Classify article sector based on keywords"""
        combined_text = (title + " " + text).lower()
        
        sector_scores = {}
        for sector, keywords in self.sector_keywords.items():
            score = sum(1 for kw in keywords if kw in combined_text)
            if score > 0:
                sector_scores[sector] = score
        
        if sector_scores:
            return max(sector_scores, key=sector_scores.get)
        return "General"
    
    def extract_sentiment(self, text: str) -> Dict[str, float]:
        """Extract sentiment using simple keyword-based approach"""
        # Positive and negative financial keywords
        positive_words = ["gain", "profit", "growth", "surge", "rally", "outperform", 
                         "beat", "strong", "bullish", "recovery"]
        negative_words = ["loss", "decline", "crash", "drop", "weak", "bearish",
                         "miss", "downturn", "recession", "volatility"]
        
        text_lower = text.lower()
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        total = pos_count + neg_count
        if total == 0:
            return {"polarity": 0.0, "label": "neutral"}
        
        polarity = (pos_count - neg_count) / total
        
        if polarity > 0.2:
            label = "positive"
        elif polarity < -0.2:
            label = "negative"
        else:
            label = "neutral"
        
        return {"polarity": round(polarity, 3), "label": label}
    
    def process_article(self, article: Dict) -> Dict[str, Any]:
        """Process single article: clean, chunk, extract metadata"""
        # Extract basic info
        title = article.get("title", "")
        description = article.get("description", "")
        content = article.get("content", "")
        url = article.get("url", "")
        published_at = article.get("publishedAt", "")
        source_name = article.get("source", {}).get("name", "Unknown")
        
        # Combine text
        full_text = f"{title}. {description}. {content}"
        cleaned_text = self.clean_text(full_text)
        
        # Skip if too short
        if len(cleaned_text) < 200:
            return None
        
        # Chunk the text
        chunks = self.chunk_text(cleaned_text)
        
        # Extract metadata
        sector = self.extract_sector(cleaned_text, title)
        sentiment = self.extract_sentiment(cleaned_text)
        
        # Create unique ID
        article_id = hashlib.md5(url.encode()).hexdigest()[:12]
        
        return {
            "id": article_id,
            "title": title,
            "url": url,
            "source": source_name,
            "published_at": published_at,
            "sector": sector,
            "sentiment": sentiment,
            "full_text": cleaned_text,
            "chunks": chunks,
            "num_chunks": len(chunks),
            "processed_at": datetime.now().isoformat()
        }
    
    def save_processed_data(self, processed_articles: List[Dict]):
        """Save processed articles and metadata"""
        print(f"\n💾 Saving {len(processed_articles)} processed articles...")
        
        # Save individual processed files
        for article in processed_articles:
            article_id = article["id"]
            
            # Save full processed article
            article_file = self.processed_dir / f"{article_id}.json"
            with open(article_file, "w") as f:
                json.dump(article, f, indent=2)
        
        # Save master metadata file
        metadata = []
        for article in processed_articles:
            metadata.append({
                "id": article["id"],
                "title": article["title"],
                "source": article["source"],
                "published_at": article["published_at"],
                "sector": article["sector"],
                "sentiment": article["sentiment"],
                "num_chunks": article["num_chunks"],
                "url": article["url"]
            })
        
        metadata_file = self.metadata_dir / f"metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Save summary statistics
        self.save_statistics(processed_articles)
        
        print(f"✓ Saved to {self.processed_dir}")
        print(f"✓ Metadata saved to {metadata_file}")
    
    def save_statistics(self, processed_articles: List[Dict]):
        """Generate and save collection statistics"""
        total_chunks = sum(art["num_chunks"] for art in processed_articles)
        
        sector_counts = {}
        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
        
        for article in processed_articles:
            sector = article["sector"]
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
            
            sentiment_label = article["sentiment"]["label"]
            sentiment_counts[sentiment_label] += 1
        
        stats = {
            "total_articles": len(processed_articles),
            "total_chunks": total_chunks,
            "avg_chunks_per_article": round(total_chunks / len(processed_articles), 2),
            "sectors": sector_counts,
            "sentiment_distribution": sentiment_counts,
            "collection_date": datetime.now().isoformat()
        }
        
        stats_file = self.metadata_dir / "collection_stats.json"
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)
        
        # Print summary
        print(f"\n📈 COLLECTION SUMMARY")
        print(f"{'='*50}")
        print(f"Total Articles: {stats['total_articles']}")
        print(f"Total Chunks: {stats['total_chunks']}")
        print(f"Avg Chunks/Article: {stats['avg_chunks_per_article']}")
        print(f"\nSector Distribution:")
        for sector, count in sorted(sector_counts.items(), key=lambda x: -x[1]):
            print(f"  {sector}: {count}")
        print(f"\nSentiment Distribution:")
        for sentiment, count in sentiment_counts.items():
            print(f"  {sentiment}: {count}")
    
    def run(self, max_per_keyword: int = 50) -> List[Dict]:
        """Main autonomous agent execution"""
        print(" Financial Data Collection Agent Starting...")
        print("="*60)
        
        # Step 1: Scrape articles
        raw_articles = self.scrape_all_keywords(max_per_keyword)
        
        if not raw_articles:
            print("⚠️  No articles collected. Check your NewsAPI key.")
            return []
        
        # Step 2: Process articles
        print(f"\n🔧 Processing {len(raw_articles)} articles...")
        processed_articles = []
        
        for i, article in enumerate(raw_articles, 1):
            if i % 50 == 0:
                print(f"  Processed {i}/{len(raw_articles)}...")
            
            processed = self.process_article(article)
            if processed:
                processed_articles.append(processed)
        
        print(f"✓ Successfully processed {len(processed_articles)} articles")
        
        # Step 3: Save everything
        self.save_processed_data(processed_articles)
        
        print(f"\nAgent completed successfully!")
        print(f"Data saved to: {self.output_dir.absolute()}")
        
        return processed_articles


if __name__ == "__main__":
    # Create and run the agent
    agent = FinancialDataAgent(output_dir="data")
    
    # For demo without API key, we'll use a smaller test
    # In production, set your NewsAPI key as environment variable
    
    processed_data = agent.run(max_per_keyword=50)
    
    print(f"\n Next Steps:")
    print(f"1. Get NewsAPI key from https://newsapi.org")
    print(f"2. Set environment variable: export NEWSAPI_KEY='your_key'")
    print(f"3. Run agent to collect 500+ articles")
    print(f"4. Use processed data for Knowledge Graph construction")
