"""
Data Viewer - Explore generated financial articles
"""

import json
from pathlib import Path
from collections import Counter

def load_stats():
    """Load and display statistics"""
    stats_file = Path("data/metadata/collection_stats.json")
    
    with open(stats_file) as f:
        stats = json.load(f)
    
    print("📊 DATASET OVERVIEW")
    print("="*60)
    print(f"Total Articles: {stats['total_articles']}")
    print(f"Total Chunks: {stats['total_chunks']}")
    print(f"Average Chunks per Article: {stats['avg_chunks_per_article']}")
    print(f"\nSector Breakdown:")
    for sector, count in stats['sectors'].items():
        print(f"  {sector:12} {count:3}")
    
    print(f"\nSentiment Distribution:")
    for sent, count in stats['sentiment_distribution'].items():
        print(f"  {sent:10} {count:3}")
    
    return stats

def view_sample_articles(n=5):
    """View sample articles"""
    processed_dir = Path("data/processed")
    files = list(processed_dir.glob("*.json"))[:n]
    
    print(f"\n📰 SAMPLE ARTICLES (showing {n})")
    print("="*60)
    
    for i, file_path in enumerate(files, 1):
        with open(file_path) as f:
            article = json.load(f)
        
        print(f"\n[{i}] {article['title']}")
        print(f"    Source: {article['source']} | Sector: {article['sector']}")
        print(f"    Sentiment: {article['sentiment']['label']} ({article['sentiment']['polarity']})")
        print(f"    Company: {article['company']} | Type: {article['article_type']}")
        print(f"    Text: {article['full_text'][:200]}...")

def analyze_entities():
    """Extract entities for KG preview"""
    processed_dir = Path("data/processed")
    
    companies = Counter()
    sectors = Counter()
    sentiments = Counter()
    article_types = Counter()
    
    for file_path in processed_dir.glob("*.json"):
        with open(file_path) as f:
            article = json.load(f)
            companies[article['company']] += 1
            sectors[article['sector']] += 1
            sentiments[article['sentiment']['label']] += 1
            article_types[article['article_type']] += 1
    
    print(f"\n🔍 ENTITY ANALYSIS (for Knowledge Graph)")
    print("="*60)
    print(f"\nUnique Companies: {len(companies)}")
    print("Top 10:")
    for company, count in companies.most_common(10):
        print(f"  {company:20} {count:3} mentions")
    
    print(f"\nSectors: {len(sectors)}")
    for sector, count in sectors.most_common():
        print(f"  {sector:12} {count:3}")
    
    # Preview potential relationships
    print(f"\n🔗 POTENTIAL RELATIONSHIPS")
    print("="*60)
    print(f"Company -[IN_SECTOR]-> Sector: {sum(companies.values())} relationships")
    print(f"Article -[MENTIONS]-> Company: {sum(companies.values())} relationships")
    print(f"Article -[HAS_SENTIMENT]-> Sentiment: {sum(sentiments.values())} relationships")
    print(f"Article -[TYPE]-> ArticleType: {sum(article_types.values())} relationships")
    
    total_relationships = sum(companies.values()) * 3  # Rough estimate
    print(f"\nEstimated Total Relationships: ~{total_relationships}")
    
    return {
        "companies": companies,
        "sectors": sectors,
        "sentiments": sentiments,
        "article_types": article_types
    }

def main():
    print("\n🚀 FINANCIAL DATA VIEWER")
    print("="*60)
    
    # Load stats
    stats = load_stats()
    
    # View samples
    view_sample_articles(5)
    
    # Analyze entities
    entities = analyze_entities()
    
    print("\n✅ Phase 1 Complete!")
    print("\n🎯 Ready for Phase 2: Knowledge Graph Construction")
    print("   Expected entities: 40+ companies, 5 sectors")
    print(f"   Expected relationships: ~{stats['total_articles'] * 3}+")

if __name__ == "__main__":
    main()
