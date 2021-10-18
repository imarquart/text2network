
def test_setup():

    config = test_config()

    # First, create an empty network
    neo4j_interface = Neo4j_Insertion_Interface(config)

    tokens,token_ids = get_token_list()
    neo4j_interface.setup_neo_db(tokens,token_ids)
    neo4j_interface.db.reset_dictionary()




    token_ids=semantic_network.ensure_ids(tokens)


    # sentence 1: leader (boss,manager) employee (company,team)
    tie_dict = {'weight': 0.5, 'run_index': 1, 'seq_id': 1, 'pos': 1, 'p1': 'p1'}
    ties=[(semantic_network.ensure_ids('t_leader'),semantic_network.ensure_ids('t_manager'),1000,tie_dict),
          (semantic_network.ensure_ids('t_leader'),semantic_network.ensure_ids('t_boss'),1000,tie_dict)]
    semantic_network.db.insert_edges_context(semantic_network.ensure_ids('t_leader'), ties)

    tie_dict = {'weight': 0.5, 'run_index': 1, 'seq_id': 1, 'pos': 2, 'p1': 'p1'}
    ties=[(semantic_network.ensure_ids('t_employee'),semantic_network.ensure_ids('t_company'),1000,tie_dict),
          (semantic_network.ensure_ids('t_employee'),semantic_network.ensure_ids('t_team'),1000,tie_dict)]
    semantic_network.db.insert_edges_context(semantic_network.ensure_ids('t_employee'), ties)

    # sentence 2: leader (manager) company (team)
    tie_dict = {'weight': 1, 'run_index': 2, 'seq_id': 2, 'pos': 1, 'p1': 'p1'}
    ties=[(semantic_network.ensure_ids('t_leader'),semantic_network.ensure_ids('t_manager'),1000,tie_dict)]
    semantic_network.db.insert_edges_context(semantic_network.ensure_ids('t_leader'), ties)

    tie_dict = {'weight': 1, 'run_index': 2, 'seq_id': 2, 'pos': 2, 'p1': 'p1'}
    ties=[(semantic_network.ensure_ids('t_company'),semantic_network.ensure_ids('t_team'),1000,tie_dict)]
    semantic_network.db.insert_edges_context(semantic_network.ensure_ids('t_company'), ties)

    # sentence 3: manager (leader) company (team)
    tie_dict = {'weight': 1, 'run_index': 3, 'seq_id': 3, 'pos': 1, 'p1': 'p1'}
    ties=[(semantic_network.ensure_ids('t_manager'),semantic_network.ensure_ids('t_leader'),1000,tie_dict)]
    semantic_network.db.insert_edges_context(semantic_network.ensure_ids('t_leader'), ties)

    tie_dict = {'weight': 1, 'run_index': 3, 'seq_id': 3, 'pos': 2, 'p1': 'p1'}
    ties=[(semantic_network.ensure_ids('t_company'),semantic_network.ensure_ids('t_team'),1000,tie_dict)]
    semantic_network.db.insert_edges_context(semantic_network.ensure_ids('t_company'), ties)

    # sentence 4: boss (leader) company (employee)
    tie_dict = {'weight': 1, 'run_index': 4, 'seq_id': 4, 'pos': 1, 'p1': 'p1'}
    ties=[(semantic_network.ensure_ids('t_boss'),semantic_network.ensure_ids('t_leader'),1000,tie_dict)]
    semantic_network.db.insert_edges_context(semantic_network.ensure_ids('t_boss'), ties)

    tie_dict = {'weight': 1, 'run_index':4, 'seq_id': 4, 'pos': 2, 'p1': 'p1'}
    ties=[(semantic_network.ensure_ids('t_company'),semantic_network.ensure_ids('t_employee'),1000,tie_dict)]
    semantic_network.db.insert_edges_context(semantic_network.ensure_ids('t_company'), ties)

    semantic_network.db.write_queue()

    return semantic_network, config
