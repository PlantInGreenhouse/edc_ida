from neo4j import GraphDatabase
from pyvis.network import Network

# Neo4j 연결 정보
NEO4J_URI = "neo4j+s://ae297466.databases.neo4j.io"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "VrgRq63ubZ-RTdMVzoPm73GAue3Tk84w3g2K-uPEmUg"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# 트리플 데이터 정의
triplets = [
    # 공황장애 관련
    ['공황장애', '유발', '유전적_요인'],
    ['공황장애', '유발', '스트레스'],
    ['공황장애', '유발', '노르에피네프린_불균형'],
    ['공황장애', '유발', '과호흡'],
    ['공황장애', '유발', '산-염기_불균형'],
    ['공황장애', '유발', '카페인'],
    ['공황장애', '유발', '이산화탄소'],
    ['공황장애', '치료', '약물치료'],

    # 노인 및 노인 우울증 관련
    ['노인', '유발', '삶의_질_저하'],
    ['만성_통증', '영향', '삶의_질_감소'],
    ['노인', '유발', '우울감'],
    ['노인', '유발', '상실감'],
    ['노인', '영향', '경제적_어려움'],
    ['노인', '영향', '사회적_고립'],
    ['노인_우울증', '치료', '삶의_질_향상'],
    ['노인_우울증', '치료', '경제적_부담_감소'],
    ['노인_우울증', '유발', '유전적_요인'],
    ['노인_우울증', '유발', '신경생물학적_요인'],
    ['노인_우울증', '유발', '환경적_요인'],
    ['노인_우울증', '연관', '인지기능_저하'],
    ['노인_우울증', '연관', '실행기능_장애'],
    ['노인_우울증', '연관', '편도체_기능_이상'],
    ['노인_우울증', '연관', '해마_구조_이상'],
    ['노인_우울증', '유발', '파킨슨병'],
    ['노인_우울증', '유발', '치매'],
    ['노인_우울증', '유발', '심혈관계_질환'],
    ['노인_우울증', '유발', '암']
]

# Neo4j에 트리플 업로드
def insert_triplets(tx, triplets):
    for subj, rel, obj in triplets:
        query = f"""
        MERGE (s:Entity {{name: '{subj}'}})
        MERGE (o:Entity {{name: '{obj}'}})
        MERGE (s)-[:{rel}]->(o)
        """
        tx.run(query)

with driver.session(database="neo4j") as session:
    session.execute_write(insert_triplets, triplets)

# pyvis를 사용한 시각화
def visualize_knowledge_graph(triplets, output_html="graph.html"):
    net = Network(height="750px", width="100%", directed=True)
    nodes = set()

    for subj, rel, obj in triplets:
        if subj not in nodes:
            net.add_node(subj, label=subj)
            nodes.add(subj)
        if obj not in nodes:
            net.add_node(obj, label=obj)
            nodes.add(obj)
        net.add_edge(subj, obj, label=rel)

    net.show(output_html)

# 시각화 실행
visualize_knowledge_graph(triplets)