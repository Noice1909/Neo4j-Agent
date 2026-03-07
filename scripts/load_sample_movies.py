"""
Load sample movie data into Neo4j for testing.

Creates actors, directors, movies, and their relationships.
Includes Tom Hanks, Christopher Nolan, and other well-known entities.
"""
import asyncio
from neo4j import AsyncGraphDatabase
from src.core.config import get_settings

# Sample data
SAMPLE_DATA = """
// Create Actors
CREATE (tom:Actor {name: 'Tom Hanks', born: 1956})
CREATE (keanu:Actor {name: 'Keanu Reeves', born: 1964})
CREATE (carrie:Actor {name: 'Carrie-Anne Moss', born: 1967})
CREATE (matt:Actor {name: 'Matthew McConaughey', born: 1969})
CREATE (anne:Actor {name: 'Anne Hathaway', born: 1982})

// Create Directors
CREATE (nolan:Director {name: 'Christopher Nolan', born: 1970})
CREATE (wachowski1:Director {name: 'Lana Wachowski', born: 1965})
CREATE (wachowski2:Director {name: 'Lilly Wachowski', born: 1967})
CREATE (zemeckis:Director {name: 'Robert Zemeckis', born: 1951})

// Create Movies
CREATE (forrest:Movie {title: 'Forrest Gump', released: 1994, tagline: 'Life is like a box of chocolates'})
CREATE (matrix:Movie {title: 'The Matrix', released: 1999, tagline: 'Welcome to the Real World'})
CREATE (interstellar:Movie {title: 'Interstellar', released: 2014, tagline: 'Mankind was born on Earth. It was never meant to die here.'})
CREATE (inception:Movie {title: 'Inception', released: 2010, tagline: 'Your mind is the scene of the crime'})
CREATE (dunkirk:Movie {title: 'Dunkirk', released: 2017, tagline: 'When 400,000 men couldn't get home, home came for them'})
CREATE (castaway:Movie {title: 'Cast Away', released: 2000, tagline: 'At the edge of the world, his journey begins'})
CREATE (terminal:Movie {title: 'The Terminal', released: 2004, tagline: 'Life is waiting'})
CREATE (saving:Movie {title: 'Saving Private Ryan', released: 1998, tagline: 'The mission is a man'})

// Create relationships - Actors to Movies
CREATE (tom)-[:ACTED_IN {roles: ['Forrest Gump']}]->(forrest)
CREATE (tom)-[:ACTED_IN {roles: ['Chuck Noland']}]->(castaway)
CREATE (tom)-[:ACTED_IN {roles: ['Viktor Navorski']}]->(terminal)
CREATE (tom)-[:ACTED_IN {roles: ['Captain Miller']}]->(saving)
CREATE (keanu)-[:ACTED_IN {roles: ['Neo']}]->(matrix)
CREATE (carrie)-[:ACTED_IN {roles: ['Trinity']}]->(matrix)
CREATE (matt)-[:ACTED_IN {roles: ['Cooper']}]->(interstellar)
CREATE (anne)-[:ACTED_IN {roles: ['Brand']}]->(interstellar)

// Create relationships - Directors to Movies
CREATE (zemeckis)-[:DIRECTED]->(forrest)
CREATE (zemeckis)-[:DIRECTED]->(castaway)
CREATE (wachowski1)-[:DIRECTED]->(matrix)
CREATE (wachowski2)-[:DIRECTED]->(matrix)
CREATE (nolan)-[:DIRECTED]->(interstellar)
CREATE (nolan)-[:DIRECTED]->(inception)
CREATE (nolan)-[:DIRECTED]->(dunkirk)

// Create Concept metadata nodes for NLP
CREATE (c1:Concept {
    name: 'Actor',
    description: 'A person who performs in movies or films',
    nlp_terms: 'actors,performer,cast,star,celebrity',
    sample_values: 'Tom Hanks,Keanu Reeves,Anne Hathaway'
})
CREATE (c2:Concept {
    name: 'Director',
    description: 'A person who directs movies or films',
    nlp_terms: 'directors,filmmaker,director',
    sample_values: 'Christopher Nolan,Robert Zemeckis'
})
CREATE (c3:Concept {
    name: 'Movie',
    description: 'A motion picture or film',
    nlp_terms: 'movies,films,motion pictures,cinema',
    sample_values: 'Forrest Gump,The Matrix,Interstellar'
})

RETURN 'Sample data loaded successfully' AS status
"""


async def load_sample_data():
    """Load sample movie data into Neo4j."""
    settings = get_settings()

    print("=" * 60)
    print("LOADING SAMPLE MOVIE DATA")
    print("=" * 60)

    driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password)
    )

    try:
        async with driver.session(database=settings.neo4j_database) as session:
            # Clear existing data (optional - comment out if you want to keep existing data)
            print("\n1. Clearing existing data...")
            await session.run("MATCH (n) DETACH DELETE n")
            print("   ✓ Existing data cleared")

            # Load sample data
            print("\n2. Loading sample data...")
            result = await session.run(SAMPLE_DATA)
            summary = await result.consume()
            print(f"   ✓ Created {summary.counters.nodes_created} nodes")
            print(f"   ✓ Created {summary.counters.relationships_created} relationships")
            print(f"   ✓ Set {summary.counters.properties_set} properties")

            # Create full-text index for Concept nodes
            print("\n3. Creating full-text index...")
            try:
                await session.run("""
                    CREATE FULLTEXT INDEX concept_name_description_ft IF NOT EXISTS
                    FOR (n:Concept)
                    ON EACH [n.name, n.description]
                """)
                print("   ✓ Full-text index created")
            except Exception as e:
                if "already exists" in str(e).lower():
                    print("   ✓ Full-text index already exists")
                else:
                    raise

            # Verify data
            print("\n4. Verifying data...")
            result = await session.run("MATCH (a:Actor) RETURN count(a) as count")
            record = await result.single()
            print(f"   ✓ Actors: {record['count']}")

            result = await session.run("MATCH (d:Director) RETURN count(d) as count")
            record = await result.single()
            print(f"   ✓ Directors: {record['count']}")

            result = await session.run("MATCH (m:Movie) RETURN count(m) as count")
            record = await result.single()
            print(f"   ✓ Movies: {record['count']}")

            result = await session.run("MATCH (c:Concept) RETURN count(c) as count")
            record = await result.single()
            print(f"   ✓ Concept nodes: {record['count']}")

            # Show Tom Hanks movies
            print("\n5. Tom Hanks movies:")
            result = await session.run("""
                MATCH (a:Actor {name: 'Tom Hanks'})-[:ACTED_IN]->(m:Movie)
                RETURN m.title as title, m.released as year
                ORDER BY year
            """)
            async for record in result:
                print(f"   - {record['title']} ({record['year']})")

    finally:
        await driver.close()

    print("\n" + "=" * 60)
    print("✅ SAMPLE DATA LOADED SUCCESSFULLY!")
    print("=" * 60)
    print("\nYou can now test with queries like:")
    print("  - 'Tell me about Tom Hanks movies'")
    print("  - 'How many movies are there?'")
    print("  - 'Which director has directed the most movies?'")
    print("  - 'List top 5 actors by number of movies'")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(load_sample_data())
