"""
Neo4j Graph Loader - Phase 2 Step 2
Loads extracted entities into Neo4j knowledge graph
"""

import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()


class Neo4jGraphLoader:
    """Load entities into Neo4j knowledge graph"""
    
    def __init__(self, uri: str = None, username: str = None, password: str = None):
        # Neo4j connection
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.username = username or os.getenv("NEO4J_USERNAME", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "password123")
        
        print(f"🔌 Connecting to Neo4j at {self.uri}...")
        
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password)
            )
            # Test connection
            self.driver.verify_connectivity()
            print("✅ Connected to Neo4j!")
        except Exception as e:
            print(f"❌ Failed to connect to Neo4j: {e}")
            print("\n💡 Make sure Neo4j is running:")
            print("   - Open Neo4j Desktop and start database")
            print("   - OR run: docker start neo4j")
            print(f"   - Check connection at: http://localhost:7474")
            raise
        
        # Data directories
        script_dir = Path(__file__).parent
        if script_dir.name == "kg_builder":
            project_root = script_dir.parent.parent
        else:
            project_root = Path.cwd()
        
        self.data_dir = project_root / "data"
        self.entities_dir = self.data_dir / "entities"
        self.processed_dir = self.data_dir / "processed"
        
        # Stats
        self.stats = {
            "nodes_created": 0,
            "relationships_created": 0,
            "companies": 0,
            "people": 0,
            "articles": 0,
            "sectors": 0
        }
    
    def clear_database(self):
        """Clear all nodes and relationships (for fresh start)"""
        print("\n🗑️  Clearing existing graph...")
        
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        
        print("✅ Database cleared!")
    
    def create_constraints(self):
        """Create uniqueness constraints and indexes"""
        print("\n🔧 Creating constraints and indexes...")
        
        constraints = [
            "CREATE CONSTRAINT company_name IF NOT EXISTS FOR (c:Company) REQUIRE c.name IS UNIQUE",
            "CREATE CONSTRAINT article_id IF NOT EXISTS FOR (a:Article) REQUIRE a.id IS UNIQUE",
            "CREATE CONSTRAINT person_name IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS UNIQUE",
            "CREATE CONSTRAINT sector_name IF NOT EXISTS FOR (s:Sector) REQUIRE s.name IS UNIQUE",
            "CREATE INDEX article_date IF NOT EXISTS FOR (a:Article) ON (a.published_at)",
            "CREATE INDEX company_sector IF NOT EXISTS FOR (c:Company) ON (c.sector)",
        ]
        
        with self.driver.session() as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    # Constraint might already exist
                    pass
        
        print("✅ Constraints and indexes created!")
    
    def load_companies(self, entities: Dict):
        """Load company nodes"""
        print("\n🏢 Loading companies...")
        
        companies = entities.get("unique_companies", [])
        
        with self.driver.session() as session:
            for company in companies:
                name = company.get("name", "").strip()
                if not name:
                    continue
                
                session.run("""
                    MERGE (c:Company {name: $name})
                    SET c.ticker = $ticker,
                        c.role = $role,
                        c.created_at = datetime()
                """, 
                    name=name,
                    ticker=company.get("ticker", ""),
                    role=company.get("role", "")
                )
                
                self.stats["companies"] += 1
        
        print(f"✅ Loaded {self.stats['companies']} companies")
    
    def load_sectors(self, entities: Dict):
        """Load sector nodes"""
        print("\n📊 Loading sectors...")
        
        sectors = entities.get("sectors", [])
        
        with self.driver.session() as session:
            for sector in sectors:
                if not sector:
                    continue
                
                session.run("""
                    MERGE (s:Sector {name: $name})
                    SET s.created_at = datetime()
                """, name=sector)
                
                self.stats["sectors"] += 1
        
        print(f"✅ Loaded {self.stats['sectors']} sectors")
    
    def load_articles(self):
        """Load article nodes with metadata"""
        print("\n📰 Loading articles...")
        
        article_files = list(self.processed_dir.glob("*.json"))
        
        with self.driver.session() as session:
            for article_file in article_files:
                with open(article_file) as f:
                    article = json.load(f)
                
                session.run("""
                    MERGE (a:Article {id: $id})
                    SET a.title = $title,
                        a.url = $url,
                        a.source = $source,
                        a.published_at = $published_at,
                        a.sector = $sector,
                        a.sentiment_label = $sentiment_label,
                        a.sentiment_polarity = $sentiment_polarity,
                        a.created_at = datetime()
                """,
                    id=article["id"],
                    title=article["title"],
                    url=article["url"],
                    source=article["source"],
                    published_at=article["published_at"],
                    sector=article.get("sector", "General"),
                    sentiment_label=article["sentiment"]["label"],
                    sentiment_polarity=article["sentiment"]["polarity"]
                )
                
                self.stats["articles"] += 1
                
                if self.stats["articles"] % 50 == 0:
                    print(f"  Loaded {self.stats['articles']} articles...")
        
        print(f"✅ Loaded {self.stats['articles']} articles")
    
    def load_people(self):
        """Load people from entity files"""
        print("\n👤 Loading people...")
        
        entity_files = list(self.entities_dir.glob("*_entities.json"))
        people_set = set()
        
        with self.driver.session() as session:
            for entity_file in entity_files:
                with open(entity_file) as f:
                    entities = json.load(f)
                
                for person in entities.get("people", []):
                    name = person.get("name", "").strip()
                    if not name or name in people_set:
                        continue
                    
                    people_set.add(name)
                    
                    session.run("""
                        MERGE (p:Person {name: $name})
                        SET p.role = $role,
                            p.company = $company,
                            p.created_at = datetime()
                    """,
                        name=name,
                        role=person.get("role", ""),
                        company=person.get("company", "")
                    )
                    
                    self.stats["people"] += 1
        
        print(f"✅ Loaded {self.stats['people']} people")
    
    def create_relationships(self):
        """Create all relationships between nodes"""
        print("\n🔗 Creating relationships...")
        
        entity_files = list(self.entities_dir.glob("*_entities.json"))
        
        with self.driver.session() as session:
            for i, entity_file in enumerate(entity_files, 1):
                with open(entity_file) as f:
                    entities = json.load(f)
                
                article_id = entities.get("article_id")
                
                # Article MENTIONS Company
                for company in entities.get("companies", []):
                    company_name = company.get("name", "").strip()
                    if not company_name:
                        continue
                    
                    session.run("""
                        MATCH (a:Article {id: $article_id})
                        MATCH (c:Company {name: $company_name})
                        MERGE (a)-[r:MENTIONS]->(c)
                        SET r.role = $role
                    """,
                        article_id=article_id,
                        company_name=company_name,
                        role=company.get("role", "")
                    )
                    
                    self.stats["relationships_created"] += 1
                
                # Company IN_SECTOR Sector
                for company in entities.get("companies", []):
                    company_name = company.get("name", "").strip()
                    if not company_name:
                        continue
                    
                    for sector in entities.get("sectors", []):
                        if not sector:
                            continue
                        
                        session.run("""
                            MATCH (c:Company {name: $company_name})
                            MATCH (s:Sector {name: $sector})
                            MERGE (c)-[:IN_SECTOR]->(s)
                        """,
                            company_name=company_name,
                            sector=sector
                        )
                        
                        self.stats["relationships_created"] += 1
                
                # Person WORKS_AT Company
                for person in entities.get("people", []):
                    if not person or not isinstance(person, dict):
                        continue
                    person_name = (person.get("name") or "").strip()
                    company_name = (person.get("company") or "").strip()
                    
                    if person_name and company_name:
                        session.run("""
                            MATCH (p:Person {name: $person_name})
                            MATCH (c:Company {name: $company_name})
                            MERGE (p)-[r:WORKS_AT]->(c)
                            SET r.role = $role
                        """,
                            person_name=person_name,
                            company_name=company_name,
                            role=person.get("role", "")
                        )
                        
                        self.stats["relationships_created"] += 1
                
                # Company COMPETES_WITH Company (from relationships)
                for rel in entities.get("relationships", []):
                    if rel.get("relationship") == "competes_with":
                        entity1 = rel.get("entity1", "").strip()
                        entity2 = rel.get("entity2", "").strip()
                        
                        if entity1 and entity2:
                            session.run("""
                                MATCH (c1:Company {name: $entity1})
                                MATCH (c2:Company {name: $entity2})
                                MERGE (c1)-[:COMPETES_WITH]-(c2)
                            """,
                                entity1=entity1,
                                entity2=entity2
                            )
                            
                            self.stats["relationships_created"] += 1
                
                if i % 50 == 0:
                    print(f"  Processed {i}/{len(entity_files)} entity files...")
        
        print(f"✅ Created {self.stats['relationships_created']} relationships")
    
    def create_co_occurrence_relationships(self):
        """Create MENTIONED_WITH relationships for companies mentioned in same article"""
        print("\n🔗 Creating co-occurrence relationships...")
        
        with self.driver.session() as session:
            result = session.run("""
                MATCH (a:Article)-[:MENTIONS]->(c1:Company)
                MATCH (a)-[:MENTIONS]->(c2:Company)
                WHERE c1 <> c2
                WITH c1, c2, count(a) as co_mentions
                WHERE co_mentions >= 2
                MERGE (c1)-[r:MENTIONED_WITH]-(c2)
                SET r.co_mentions = co_mentions
                RETURN count(r) as relationships_created
            """)
            
            count = result.single()["relationships_created"]
            print(f"✅ Created {count} co-occurrence relationships")
    
    def compute_graph_stats(self):
        """Compute and display graph statistics"""
        print("\n📊 Computing graph statistics...")
        
        with self.driver.session() as session:
            # Node counts
            result = session.run("""
                MATCH (n)
                RETURN labels(n)[0] as label, count(*) as count
                ORDER BY count DESC
            """)
            
            print("\n📈 NODE COUNTS:")
            for record in result:
                print(f"  {record['label']:15} {record['count']:5}")
            
            # Relationship counts
            result = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) as type, count(*) as count
                ORDER BY count DESC
            """)
            
            print("\n🔗 RELATIONSHIP COUNTS:")
            for record in result:
                print(f"  {record['type']:20} {record['count']:5}")
            
            # Most mentioned companies
            result = session.run("""
                MATCH (a:Article)-[:MENTIONS]->(c:Company)
                WITH c, count(a) as mentions
                ORDER BY mentions DESC
                LIMIT 10
                RETURN c.name as company, mentions
            """)
            
            print("\n🏆 TOP 10 MOST MENTIONED COMPANIES:")
            for i, record in enumerate(result, 1):
                print(f"  {i:2}. {record['company']:30} ({record['mentions']} mentions)")
    
    def run(self, clear_existing: bool = True):
        """Run the complete graph loading process"""
        print("🚀 NEO4J KNOWLEDGE GRAPH LOADER")
        print("="*60)
        
        start_time = datetime.now()
        
        # Load consolidated entities
        print("\n📥 Loading consolidated entities...")
        consolidated_file = self.entities_dir / "consolidated_entities.json"
        
        if not consolidated_file.exists():
            print(f"❌ Consolidated entities file not found: {consolidated_file}")
            return
        
        with open(consolidated_file) as f:
            entities = json.load(f)
        
        print(f"✅ Loaded entities from {len(entities.get('unique_companies', []))} companies")
        
        # Clear database if requested
        if clear_existing:
            self.clear_database()
        
        # Create constraints
        self.create_constraints()
        
        # Load nodes
        self.load_sectors(entities)
        self.load_companies(entities)
        self.load_articles()
        self.load_people()
        
        # Create relationships
        self.create_relationships()
        self.create_co_occurrence_relationships()
        
        # Compute stats
        self.compute_graph_stats()
        
        # Final summary
        duration = (datetime.now() - start_time).total_seconds()
        
        print("\n" + "="*60)
        print("✅ GRAPH LOADING COMPLETE!")
        print("="*60)
        print(f"⏱️  Duration: {duration:.1f} seconds")
        print(f"📊 Nodes created: {sum([self.stats['companies'], self.stats['articles'], self.stats['people'], self.stats['sectors']])}")
        print(f"🔗 Relationships created: {self.stats['relationships_created']}")
        print("\n🌐 View your graph:")
        print("   1. Open: http://localhost:7474")
        print("   2. Run query: MATCH (n) RETURN n LIMIT 100")
        print("\n💡 Example queries to try:")
        print("   - Find Apple's competitors:")
        print("     MATCH (c:Company {name: 'Apple Inc.'})-[:COMPETES_WITH]-(competitor)")
        print("     RETURN competitor.name")
        print("\n   - Find positive articles about tech companies:")
        print("     MATCH (a:Article)-[:MENTIONS]->(c:Company)-[:IN_SECTOR]->(s:Sector {name: 'Technology'})")
        print("     WHERE a.sentiment_label = 'positive'")
        print("     RETURN a.title, c.name")
    
    def close(self):
        """Close Neo4j connection"""
        self.driver.close()


if __name__ == "__main__":
    import sys
    
    # Check for --keep-data flag
    clear_existing = "--keep-data" not in sys.argv
    
    loader = Neo4jGraphLoader()
    
    try:
        loader.run(clear_existing=clear_existing)
    finally:
        loader.close()