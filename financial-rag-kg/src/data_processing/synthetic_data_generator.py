"""
Synthetic Financial Data Generator
Creates realistic financial articles for testing/demo purposes
"""

import json
import random
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict


class SyntheticFinancialDataGenerator:
    """Generate realistic synthetic financial articles"""
    
    def __init__(self, output_dir: str = "data"):
        self.output_dir = Path(output_dir)
        self.raw_dir = self.output_dir / "raw"
        self.processed_dir = self.output_dir / "processed"
        self.metadata_dir = self.output_dir / "metadata"
        
        for dir_path in [self.raw_dir, self.processed_dir, self.metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Templates for realistic articles
        self.companies = {
            "Technology": ["Apple", "Microsoft", "Google", "Amazon", "Meta", "Tesla", "NVIDIA", "Intel", "AMD", "Salesforce"],
            "Finance": ["JPMorgan Chase", "Goldman Sachs", "Morgan Stanley", "Bank of America", "Wells Fargo", "Citigroup"],
            "Energy": ["ExxonMobil", "Chevron", "Tesla Energy", "NextEra Energy", "Duke Energy", "Enphase"],
            "Healthcare": ["Johnson & Johnson", "Pfizer", "UnitedHealth", "CVS Health", "Moderna", "AbbVie"],
            "Consumer": ["Walmart", "Target", "Costco", "Nike", "Coca-Cola", "Procter & Gamble"]
        }
        
        self.article_templates = {
            "earnings": [
                "{company} reported {sentiment} quarterly earnings, with revenue {direction} to ${revenue}B, {comparison} analyst expectations.",
                "{company} shares {movement} {percent}% after posting earnings of ${eps} per share, {sentiment_word} the ${consensus} consensus.",
                "In its latest earnings call, {company} announced {metric} growth of {percent}%, signaling {outlook} outlook for the sector."
            ],
            "market": [
                "{company} stock {movement} {percent}% today amid {event}, with analysts {action} their price targets.",
                "Investors {reaction} to {company}'s announcement of {news}, causing shares to {movement} in {session} trading.",
                "{company} market cap {direction} ${value}B following {catalyst}, making it {ranking} among {sector} peers."
            ],
            "strategy": [
                "{company} unveiled plans to {initiative}, expecting to {outcome} by {timeframe}.",
                "CEO of {company} announced strategic shift toward {focus}, aiming to capture {target} market share.",
                "{company} is investing ${investment}B in {area}, positioning itself for {advantage} competitive advantage."
            ],
            "regulation": [
                "Federal regulators {action} {company}'s {transaction}, citing {reason} concerns.",
                "{company} faces scrutiny over {issue}, with lawmakers calling for {response}.",
                "New {regulation} could impact {company}'s operations, potentially {effect} quarterly guidance."
            ]
        }
        
        self.sentiment_variations = {
            "positive": {
                "sentiment": "strong", "direction": "climbing", "comparison": "beating",
                "sentiment_word": "surpassing", "movement": "surged", "reaction": "rallied behind",
                "outlook": "optimistic", "action": "raising"
            },
            "negative": {
                "sentiment": "disappointing", "direction": "falling", "comparison": "missing",
                "sentiment_word": "missing", "movement": "plunged", "reaction": "sold off",
                "outlook": "cautious", "action": "cutting"
            },
            "neutral": {
                "sentiment": "mixed", "direction": "flat", "comparison": "meeting",
                "sentiment_word": "matching", "movement": "fluctuated", "reaction": "reacted calmly to",
                "outlook": "stable", "action": "maintaining"
            }
        }
        
        self.sources = [
            "Bloomberg", "Reuters", "Wall Street Journal", "CNBC", 
            "Financial Times", "MarketWatch", "Yahoo Finance", "Seeking Alpha"
        ]
    
    def generate_article(self, sector: str, article_type: str, sentiment: str) -> Dict:
        """Generate a single synthetic article"""
        company = random.choice(self.companies[sector])
        template = random.choice(self.article_templates[article_type])
        sent_vars = self.sentiment_variations[sentiment]
        
        # Generate realistic values
        revenue = round(random.uniform(10, 150), 1)
        eps = round(random.uniform(0.5, 5.0), 2)
        percent = round(random.uniform(2, 15), 1)
        investment = round(random.uniform(1, 50), 1)
        
        # Fill template - create context dict
        context = {
            "company": company,
            "sector": sector,
            "revenue": revenue,
            "eps": eps,
            "percent": percent,
            "investment": investment,
            "consensus": round(eps - random.uniform(0.1, 0.5), 2),
            "value": random.randint(50, 500),
            "event": "market volatility",
            "news": "strategic partnership",
            "session": "after-hours",
            "catalyst": "positive analyst reports",
            "ranking": "leading",
            "initiative": "expand into new markets",
            "outcome": "increase market share",
            "timeframe": "Q4 2025",
            "focus": "AI and cloud services",
            "target": "20%",
            "area": "renewable energy",
            "advantage": "first-mover",
            "transaction": "merger proposal",
            "reason": "antitrust",
            "issue": "data privacy practices",
            "response": "increased transparency",
            "regulation": "SEC ruling",
            "effect": "affecting",
            "metric": "revenue"
        }
        # Merge sentiment variables
        context.update(sent_vars)
        content = template.format(**context)
        
        # Generate related content paragraphs
        paragraphs = [
            content,
            f"Analysts at major investment firms have {sent_vars['action']} their outlook on {company}, "
            f"citing {random.choice(['strong fundamentals', 'market dynamics', 'competitive positioning', 'innovation pipeline'])}.",
            f"The {sector.lower()} sector has seen {random.choice(['increased volatility', 'strong momentum', 'mixed signals', 'consolidation'])} "
            f"in recent weeks, with {company} {random.choice(['leading the pack', 'facing headwinds', 'maintaining steady growth'])}.",
            f"Looking ahead, industry experts expect {company} to {random.choice(['continue expanding', 'face challenges', 'maintain trajectory'])} "
            f"as {random.choice(['economic conditions', 'regulatory landscape', 'consumer demand', 'technological shifts'])} evolve."
        ]
        
        full_text = " ".join(paragraphs)
        
        # Generate metadata
        pub_date = datetime.now() - timedelta(days=random.randint(0, 60))
        source = random.choice(self.sources)
        article_id = hashlib.md5(f"{company}{pub_date}{article_type}".encode()).hexdigest()[:12]
        
        return {
            "id": article_id,
            "title": f"{company} {article_type.title()}: {content.split(',')[0]}",
            "url": f"https://example.com/article/{article_id}",
            "source": source,
            "published_at": pub_date.isoformat(),
            "sector": sector,
            "sentiment": {"polarity": self._get_polarity(sentiment), "label": sentiment},
            "full_text": full_text,
            "article_type": article_type,
            "company": company
        }
    
    def _get_polarity(self, sentiment: str) -> float:
        """Convert sentiment label to polarity score"""
        polarity_map = {"positive": 0.6, "neutral": 0.0, "negative": -0.6}
        base = polarity_map[sentiment]
        return round(base + random.uniform(-0.2, 0.2), 3)
    
    def chunk_text(self, text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
        """Chunk text by approximate tokens"""
        char_chunk_size = chunk_size * 4
        char_overlap = overlap * 4
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + char_chunk_size
            if end < len(text):
                period_pos = text.rfind(". ", start, end)
                if period_pos > start:
                    end = period_pos + 1
            
            chunk = text[start:end].strip()
            if len(chunk) > 100:
                chunks.append(chunk)
            start = end - char_overlap
        
        return chunks if chunks else [text]
    
    def generate_dataset(self, num_articles: int = 500) -> List[Dict]:
        """Generate full synthetic dataset"""
        print(f"🤖 Generating {num_articles} synthetic financial articles...")
        print("="*60)
        
        articles = []
        
        # Distribution strategy
        article_types = ["earnings", "market", "strategy", "regulation"]
        sentiments = ["positive", "neutral", "negative"]
        
        for i in range(num_articles):
            if (i + 1) % 50 == 0:
                print(f"  Generated {i + 1}/{num_articles}...")
            
            # Random selection with realistic distribution
            sector = random.choice(list(self.companies.keys()))
            article_type = random.choice(article_types)
            
            # Bias toward neutral/positive (more realistic)
            sentiment = random.choices(
                sentiments, 
                weights=[0.35, 0.45, 0.20]  # positive, neutral, negative
            )[0]
            
            article = self.generate_article(sector, article_type, sentiment)
            
            # Add chunks
            article["chunks"] = self.chunk_text(article["full_text"])
            article["num_chunks"] = len(article["chunks"])
            
            articles.append(article)
        
        print(f"✓ Generated {len(articles)} articles\n")
        return articles
    
    def save_dataset(self, articles: List[Dict]):
        """Save generated dataset"""
        print(f"💾 Saving {len(articles)} articles...")
        
        # Save individual articles
        for article in articles:
            file_path = self.processed_dir / f"{article['id']}.json"
            with open(file_path, 'w') as f:
                json.dump(article, f, indent=2)
        
        # Save metadata
        metadata = [{
            "id": a["id"], "title": a["title"], "source": a["source"],
            "published_at": a["published_at"], "sector": a["sector"],
            "sentiment": a["sentiment"], "num_chunks": a["num_chunks"],
            "company": a["company"], "article_type": a["article_type"]
        } for a in articles]
        
        meta_file = self.metadata_dir / f"metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(meta_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Generate statistics
        total_chunks = sum(a['num_chunks'] for a in articles)
        sector_counts = {}
        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
        type_counts = {}
        company_counts = {}
        
        for a in articles:
            sector_counts[a['sector']] = sector_counts.get(a['sector'], 0) + 1
            sentiment_counts[a['sentiment']['label']] += 1
            type_counts[a['article_type']] = type_counts.get(a['article_type'], 0) + 1
            company_counts[a['company']] = company_counts.get(a['company'], 0) + 1
        
        stats = {
            "total_articles": len(articles),
            "total_chunks": total_chunks,
            "avg_chunks_per_article": round(total_chunks / len(articles), 2),
            "sectors": sector_counts,
            "sentiment_distribution": sentiment_counts,
            "article_types": type_counts,
            "top_10_companies": dict(sorted(company_counts.items(), key=lambda x: -x[1])[:10]),
            "collection_date": datetime.now().isoformat()
        }
        
        stats_file = self.metadata_dir / "collection_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Print summary
        print(f"\n📈 DATASET SUMMARY")
        print("="*60)
        print(f"Total Articles: {stats['total_articles']}")
        print(f"Total Chunks: {stats['total_chunks']}")
        print(f"Avg Chunks/Article: {stats['avg_chunks_per_article']}")
        
        print(f"\n📊 Sector Distribution:")
        for sector, count in sorted(sector_counts.items(), key=lambda x: -x[1]):
            pct = (count / len(articles)) * 100
            print(f"  {sector:12} {count:3} ({pct:.1f}%)")
        
        print(f"\n😊 Sentiment Distribution:")
        for sentiment, count in sentiment_counts.items():
            pct = (count / len(articles)) * 100
            print(f"  {sentiment:10} {count:3} ({pct:.1f}%)")
        
        print(f"\n📝 Article Types:")
        for atype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            pct = (count / len(articles)) * 100
            print(f"  {atype:12} {count:3} ({pct:.1f}%)")
        
        print(f"\n🏢 Top 10 Companies Mentioned:")
        for company, count in list(stats['top_10_companies'].items())[:10]:
            print(f"  {company:20} {count:3} articles")
        
        print(f"\n✅ Data saved to: {self.output_dir.absolute()}")
        print(f"📁 Files:")
        print(f"  - Individual articles: {self.processed_dir}/")
        print(f"  - Metadata: {meta_file}")
        print(f"  - Statistics: {stats_file}")
    
    def run(self, num_articles: int = 500):
        """Generate and save synthetic dataset"""
        print("🚀 Synthetic Financial Data Generator")
        print("="*60)
        print(f"Target: {num_articles} articles across 5 sectors")
        print(f"Coverage: Earnings, Market, Strategy, Regulation\n")
        
        articles = self.generate_dataset(num_articles)
        self.save_dataset(articles)
        
        print("\n🎯 Next Steps:")
        print("1. Review generated articles in data/processed/")
        print("2. Check statistics in data/metadata/collection_stats.json")
        print("3. Ready for Phase 2: Knowledge Graph Construction!")
        
        return articles


if __name__ == "__main__":
    generator = SyntheticFinancialDataGenerator(output_dir="data")
    generator.run(num_articles=500)
