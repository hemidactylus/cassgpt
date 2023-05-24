from cassandra.cluster import Cluster

def getSession(**kwargs):
    cluster = Cluster(**kwargs)
    session = cluster.connect()
    return session

def createKeyspace(session, keyspace):
    session.execute(
        f"""
        CREATE KEYSPACE IF NOT EXISTS {keyspace}
        WITH REPLICATION = {{ 'class': 'SimpleStrategy', 'replication_factor': 1 }}
        """
    )
