from text2network.functions.file_helpers import check_create_folder
from text2network.utils.logging_helpers import setup_logger
from text2network.classes.neo4jnw import neo4j_network


def test_setup():
    # Set a configuration path
    configuration_path = '/Test_Data/config/config.ini'

    # Load Configuration file
    import configparser

    config = configparser.ConfigParser()
    print(check_create_folder(configuration_path))
    config.read(check_create_folder(configuration_path))
    # Setup logging
    setup_logger(config['Paths']['log'], config['General']['logging_level'], "Test")

    # First, create an empty network
    semantic_network = neo4j_network(config)

    tokens = ["t_manager", "t_leader", "t_boss", "t_company", "t_team", "t_employee"]
    token_ids = [1, 2, 3, 4, 5, 6]

    semantic_network.setup_neo_db(tokens,token_ids)
    semantic_network.db.reset_dictionary()


    token_ids=semantic_network.ensure_ids(tokens)


    # sentence 1: leader (boss,manager) employee (company,team)
    tie_dict = {'weight': 0.5, 'run_index': 1, 'seq_id': 1, 'pos': 1, 'p1': 'p1'}
    ctie_dict = {'weight': 1.0, 'run_index': 1, 'seq_id': 1, 'pos': 1, 'p1': 'p1'}
    ties=[(semantic_network.ensure_ids('t_leader'),semantic_network.ensure_ids('t_manager'),1000,tie_dict),
          (semantic_network.ensure_ids('t_leader'),semantic_network.ensure_ids('t_boss'),1000,tie_dict)]
    cties=[(semantic_network.ensure_ids('t_leader'),semantic_network.ensure_ids('t_employee'),1000,ctie_dict)]
    semantic_network.db.insert_edges_context(semantic_network.ensure_ids('t_leader'), ties, cties)

    tie_dict = {'weight': 0.5, 'run_index': 1, 'seq_id': 1, 'pos': 2, 'p1': 'p1'}
    ctie_dict = {'weight': 1.0, 'run_index': 1, 'seq_id': 1, 'pos': 2, 'p1': 'p1'}
    ties=[(semantic_network.ensure_ids('t_employee'),semantic_network.ensure_ids('t_company'),1000,tie_dict),
          (semantic_network.ensure_ids('t_employee'),semantic_network.ensure_ids('t_team'),1000,tie_dict)]
    cties=[(semantic_network.ensure_ids('t_employee'),semantic_network.ensure_ids('t_leader'),1000,ctie_dict)]
    semantic_network.db.insert_edges_context(semantic_network.ensure_ids('t_employee'), ties, cties)

    # sentence 2: leader (manager) company (team)
    tie_dict = {'weight': 1, 'run_index': 2, 'seq_id': 2, 'pos': 1, 'p1': 'p1'}
    ctie_dict = {'weight': 1.0, 'run_index': 2, 'seq_id': 2, 'pos': 1, 'p1': 'p1'}
    ties=[(semantic_network.ensure_ids('t_leader'),semantic_network.ensure_ids('t_manager'),1000,tie_dict)]
    cties=[(semantic_network.ensure_ids('t_leader'),semantic_network.ensure_ids('t_company'),1000,ctie_dict)]
    semantic_network.db.insert_edges_context(semantic_network.ensure_ids('t_leader'), ties, cties)

    tie_dict = {'weight': 1, 'run_index': 2, 'seq_id': 2, 'pos': 2, 'p1': 'p1'}
    ctie_dict = {'weight': 1.0, 'run_index': 2, 'seq_id': 2, 'pos': 2, 'p1': 'p1'}
    ties=[(semantic_network.ensure_ids('t_company'),semantic_network.ensure_ids('t_team'),1000,tie_dict)]
    cties=[(semantic_network.ensure_ids('t_company'),semantic_network.ensure_ids('t_leader'),1000,ctie_dict)]
    semantic_network.db.insert_edges_context(semantic_network.ensure_ids('t_company'), ties, cties)

    # sentence 3: manager (leader) company (team)
    tie_dict = {'weight': 1, 'run_index': 3, 'seq_id': 3, 'pos': 1, 'p1': 'p1'}
    ctie_dict = {'weight': 1.0, 'run_index': 3, 'seq_id': 3, 'pos': 1, 'p1': 'p1'}
    ties=[(semantic_network.ensure_ids('t_manager'),semantic_network.ensure_ids('t_leader'),1000,tie_dict)]
    cties=[(semantic_network.ensure_ids('t_manager'),semantic_network.ensure_ids('t_company'),1000,ctie_dict)]
    semantic_network.db.insert_edges_context(semantic_network.ensure_ids('t_leader'), ties, cties)

    tie_dict = {'weight': 1, 'run_index': 3, 'seq_id': 3, 'pos': 2, 'p1': 'p1'}
    ctie_dict = {'weight': 1.0, 'run_index': 3, 'seq_id': 3, 'pos': 2, 'p1': 'p1'}
    ties=[(semantic_network.ensure_ids('t_company'),semantic_network.ensure_ids('t_team'),1000,tie_dict)]
    cties=[(semantic_network.ensure_ids('t_company'),semantic_network.ensure_ids('t_manager'),1000,ctie_dict)]
    semantic_network.db.insert_edges_context(semantic_network.ensure_ids('t_company'), ties, cties)

    # sentence 4: boss (leader) company (employee)
    tie_dict = {'weight': 1, 'run_index': 4, 'seq_id': 4, 'pos': 1, 'p1': 'p1'}
    ctie_dict = {'weight': 1.0, 'run_index': 4, 'seq_id': 4, 'pos': 1, 'p1': 'p1'}
    ties=[(semantic_network.ensure_ids('t_boss'),semantic_network.ensure_ids('t_leader'),1000,tie_dict)]
    cties=[(semantic_network.ensure_ids('t_boss'),semantic_network.ensure_ids('t_company'),1000,ctie_dict)]
    semantic_network.db.insert_edges_context(semantic_network.ensure_ids('t_boss'), ties, cties)

    tie_dict = {'weight': 1, 'run_index':4, 'seq_id': 4, 'pos': 2, 'p1': 'p1'}
    ctie_dict = {'weight': 1.0, 'run_index': 4, 'seq_id': 4, 'pos': 2, 'p1': 'p1'}
    ties=[(semantic_network.ensure_ids('t_company'),semantic_network.ensure_ids('t_employee'),1000,tie_dict)]
    cties=[(semantic_network.ensure_ids('t_company'),semantic_network.ensure_ids('t_boss'),1000,ctie_dict)]
    semantic_network.db.insert_edges_context(semantic_network.ensure_ids('t_company'), ties, cties)

    semantic_network.db.write_queue()

    return semantic_network, config

def test_cleanup(semantic_network):

    print("Clearing Database")
    tokens = ["t_manager", "t_leader", "t_boss", "t_company", "t_team", "t_employee"]

    for token in tokens:
        qry = "".join(["Match (r:word {token:'",token,"'})-[:onto]->(t) DETACH DELETE t"])
        semantic_network.db.add_query(qry, run=True)
        qry = "".join(["Match (r:word {token:'", token, "'})<-[:conto]->(t) DETACH DELETE t"])
        semantic_network.db.add_query(qry, run=True)
        qry = "".join(["Match (r:word {token:'", token, "'}) DETACH DELETE r"])
        semantic_network.db.add_query(qry, run=True)
