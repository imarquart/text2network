from src.functions.file_helpers import check_create_folder
from src.utils.logging_helpers import setup_logger
import logging
import time

# Example file: Preprocessing, Training an Network creation

# Set a configuration path
configuration_path='/example/config/config.ini'
# Load Configuration file
import configparser
config = configparser.ConfigParser()
config.read(check_create_folder(configuration_path))
# Setup logging
setup_logger(config['Paths']['log'],config['General']['logging_level'] )


# Preprocessing
from src.classes.nw_preprocessor import nw_preprocessor
# Set up preprocessor
#preprocessor = nw_preprocessor(config)
# Option 1: Process several folders separated by time variable
#preprocessor.preprocess_folders(overwrite=True,excludelist=['checked', 'Error'])
# Option 2: Process one folder
#preprocessor.preprocess_files(overwrite=True,excludelist=['checked', 'Error'])

# Training
from src.classes.bert_trainer import bert_trainer
# Create a trainer for BERT
#trainer=bert_trainer(config)
# Train according to hierarchy defined in configuration
#trainer.train_berts()


# Network creation

# First, create an empty network
from src.classes.neo4jnw import neo4j_network
semantic_network = neo4j_network(config)

# Next, create a processor
from src.classes.nw_processor import nw_processor
#processor = nw_processor(semantic_network,config)
# Run the processor on our data
#processor.run_all_queries(delete_incomplete=True, delete_all=True)

print("------------------------")
logging.info("------------------------")

# Analysis examples
# Setup network on processed database
start_time = time.time()
semantic_network = neo4j_network(config)
print("http Setup  %s seconds" % (time.time() - start_time))
logging.info("http Setup  %s seconds" % (time.time() - start_time))
print("------------------------")
logging.info("------------------------")
start_time = time.time()
# Query one token directly from the database
print(semantic_network['president'])
print("http query one token %s seconds" % (time.time() - start_time))
logging.info("http query one token %s seconds" % (time.time() - start_time))
print("------------------------")
logging.info("------------------------")
del semantic_network
start_time = time.time()
semantic_network = neo4j_network(config, connection_type="bolt")
print("bolt Setup  %s seconds" % (time.time() - start_time))
logging.info("bolt Setup  %s seconds" % (time.time() - start_time))
print("------------------------")
logging.info("------------------------")

start_time = time.time()
# Query one token directly from the database
print(semantic_network['president'])
print("bolt query one token %s seconds" % (time.time() - start_time))
logging.info("bolt query one token %s seconds" % (time.time() - start_time))
print("------------------------")
logging.info("------------------------")
del semantic_network
semantic_network = neo4j_network(config)
print(semantic_network.pd_format(semantic_network.proximities(['president'],years=2009)))
print("http proximity %s seconds" % (time.time() - start_time))
logging.info("http proximity %s seconds" % (time.time() - start_time))
print("------------------------")
logging.info("------------------------")
del semantic_network
semantic_network = neo4j_network(config, connection_type="bolt")
print(semantic_network.pd_format(semantic_network.proximities(['president'],years=2009)))
print("bolt proximity %s seconds" % (time.time() - start_time))
logging.info("bolt proximity %s seconds" % (time.time() - start_time))
print("------------------------")
logging.info("------------------------")
del semantic_network
semantic_network = neo4j_network(config)
print(semantic_network.pd_format(semantic_network.proximities(['president','leader','ceo','obama'],years=2009)))
print("http proximities %s seconds" % (time.time() - start_time))
logging.info("http proximities %s seconds" % (time.time() - start_time))
print("------------------------")
logging.info("------------------------")
del semantic_network
semantic_network = neo4j_network(config, connection_type="bolt")
print(semantic_network.pd_format(semantic_network.proximities(['president','leader','ceo','obama'],years=2009)))
print("bolt proximities %s seconds" % (time.time() - start_time))
logging.info("bolt proximities %s seconds" % (time.time() - start_time))
print("------------------------")
logging.info("------------------------")
del semantic_network
semantic_network = neo4j_network(config)
# Condition the network
start_time = time.time()
semantic_network.condition(tokens=["president"],years=2009, depth=1,weight_cutoff=0.1 )
print("http 1 step ego %s seconds" % (time.time() - start_time))
logging.info("http 1 step ego %s seconds" % (time.time() - start_time))
print("length network %i" % (len(semantic_network)))
logging.info("length network %i" % (len(semantic_network)))
print("------------------------")
logging.info("------------------------")
del semantic_network
semantic_network = neo4j_network(config, connection_type="bolt")
# Condition the network
start_time = time.time()
semantic_network.condition(tokens=["president"],years=2009, depth=1,weight_cutoff=0.1 )
print("bolt 1 step ego %s seconds" % (time.time() - start_time))
logging.info("bolt 1 step ego %s seconds" % (time.time() - start_time))
print("length network %i" % (len(semantic_network)))
logging.info("length network %i" % (len(semantic_network)))
print("------------------------")
logging.info("------------------------")
del semantic_network
semantic_network = neo4j_network(config)
# Condition the network
start_time = time.time()
semantic_network.condition(tokens=["president"],years=2009, depth=2,weight_cutoff=0.1 )
print("http 2 step ego %s seconds" % (time.time() - start_time))
logging.info("http 2 step ego %s seconds" % (time.time() - start_time))
print("length network %i" % (len(semantic_network)))
logging.info("length network %i" % (len(semantic_network)))
print("------------------------")
logging.info("------------------------")
del semantic_network
semantic_network = neo4j_network(config, connection_type="bolt")
# Condition the network
start_time = time.time()
semantic_network.condition(tokens=["president"],years=2009, depth=2,weight_cutoff=0.1)
print("bolt 2 step ego %s seconds" % (time.time() - start_time))
logging.info("bolt 2 step ego %s seconds" % (time.time() - start_time))
print("length network %i" % (len(semantic_network)))
logging.info("length network %i" % (len(semantic_network)))
print("------------------------")
logging.info("------------------------")
del semantic_network
semantic_network = neo4j_network(config)
# Condition the network
start_time = time.time()
semantic_network.condition(tokens=["president"],years=2009, depth=3,weight_cutoff=0.1 )
print("http 3 step ego %s seconds" % (time.time() - start_time))
logging.info("http 3 step ego %s seconds" % (time.time() - start_time))
print("length network %i" % (len(semantic_network)))
logging.info("length network %i" % (len(semantic_network)))
print("------------------------")
logging.info("------------------------")
del semantic_network
semantic_network = neo4j_network(config, connection_type="bolt")
# Condition the network
start_time = time.time()
semantic_network.condition(tokens=["president"],years=2009, depth=3,weight_cutoff=0.1 )
print("bolt 3 step ego %s seconds" % (time.time() - start_time))
logging.info("bolt 3 step ego %s seconds" % (time.time() - start_time))
print("length network %i" % (len(semantic_network)))
logging.info("length network %i" % (len(semantic_network)))
print("------------------------")
logging.info("------------------------")
del semantic_network
semantic_network = neo4j_network(config)
# Condition the network
start_time = time.time()
semantic_network.condition(years=2009, batchsize=1000,weight_cutoff=0.3)
print("http one year condition, 1000, 0.3 cutoff %s seconds" % (time.time() - start_time))
logging.info("http one year condition,  1000, 0.3 cutoff %s seconds" % (time.time() - start_time))
print("length network %i" % (len(semantic_network)))
logging.info("length network %i" % (len(semantic_network)))
print("------------------------")
logging.info("------------------------")
del semantic_network
semantic_network = neo4j_network(config, connection_type="bolt")
# Condition the network
start_time = time.time()
semantic_network.condition(years=2009, batchsize=1000,weight_cutoff=0.3)
print("bolt one year condition, 1000, 0.3 cutoff %s seconds" % (time.time() - start_time))
logging.info("bolt one year condition,  1000, 0.3 cutoff %s seconds" % (time.time() - start_time))
print("length network %i" % (len(semantic_network)))
logging.info("length network %i" % (len(semantic_network)))
print("------------------------")
logging.info("------------------------")

del semantic_network
semantic_network = neo4j_network(config)
# Condition the network
start_time = time.time()
semantic_network.condition(years=2009, batchsize=100,weight_cutoff=0.1)
print("http one year condition, 100b, 0.1 cutoff %s seconds" % (time.time() - start_time))
logging.info("http one year condition, 100b, 0.1 cutoff %s seconds" % (time.time() - start_time))
print("length network %i" % (len(semantic_network)))
logging.info("length network %i" % (len(semantic_network)))
print("------------------------")
logging.info("------------------------")
del semantic_network
semantic_network = neo4j_network(config, connection_type="bolt")
# Condition the network
start_time = time.time()
semantic_network.condition(years=2009, batchsize=100,weight_cutoff=0.1)
print("bolt one year condition, 100b, 0.1 cutoff %s seconds" % (time.time() - start_time))
logging.info("bolt one year condition, 100b, 0.1 cutoff %s seconds" % (time.time() - start_time))
print("length network %i" % (len(semantic_network)))
logging.info("length network %i" % (len(semantic_network)))
print("------------------------")
logging.info("------------------------")
del semantic_network
semantic_network = neo4j_network(config)
# Condition the network
start_time = time.time()
semantic_network.condition(years=2009, batchsize=100,weight_cutoff=0.01)
print("http one year condition, 100b, 0.01 cutoff %s seconds" % (time.time() - start_time))
logging.info("http one year condition, 100b, 0.01 cutoff %s seconds" % (time.time() - start_time))
print("length network %i" % (len(semantic_network)))
logging.info("length network %i" % (len(semantic_network)))
print("------------------------")
logging.info("------------------------")
del semantic_network
semantic_network = neo4j_network(config, connection_type="bolt")
# Condition the network
start_time = time.time()
semantic_network.condition(years=2009, batchsize=100,weight_cutoff=0.01)
print("bolt one year condition, 100b, 0.01 cutoff %s seconds" % (time.time() - start_time))
logging.info("bolt one year condition, 100b, 0.01 cutoff %s seconds" % (time.time() - start_time))
print("length network %i" % (len(semantic_network)))
logging.info("length network %i" % (len(semantic_network)))
print("------------------------")
logging.info("------------------------")
del semantic_network
semantic_network = neo4j_network(config)
# Condition the network
start_time = time.time()
semantic_network.condition(years=2009, batchsize=10000,weight_cutoff=0.01)
print("http one year condition, 10000, 0.001 cutoff %s seconds" % (time.time() - start_time))
logging.info("http one year condition, 10000, 0.001 cutoff %s seconds" % (time.time() - start_time))
print("length network %i" % (len(semantic_network)))
logging.info("length network %i" % (len(semantic_network)))
print("------------------------")
logging.info("------------------------")
del semantic_network
semantic_network = neo4j_network(config, connection_type="bolt")
# Condition the network
start_time = time.time()
semantic_network.condition(years=2009, batchsize=100,weight_cutoff=0.011)
print("bolt one year condition, 10000, 0.01 cutoff %s seconds" % (time.time() - start_time))
logging.info("bolt one year condition, 10000, 0.01 cutoff %s seconds" % (time.time() - start_time))
print("length network %i" % (len(semantic_network)))
logging.info("length network %i" % (len(semantic_network)))
print("------------------------")
logging.info("------------------------")
# Above steps in one operation
print(semantic_network.pd_format(semantic_network.proximities(['president'],years=[2009])))

# Switch to normed ties
semantic_network.set_norm_ties()
print(semantic_network.pd_format(semantic_network.proximities(['president'],years=[2009])))
# Switch back
semantic_network.set_norm_ties()


# Switch to backout measure
semantic_network.condition(years=2009)
semantic_network.to_backout()
print(semantic_network.pd_format(semantic_network.proximities(['president'])))
# Decondition to reverse
semantic_network.decondition()


# Get a list of centralities
semantic_network.condition(years=[2009])
print(semantic_network.pd_format(semantic_network.centralities()))


# Get a list of centralities but use a cutoff
semantic_network.condition(years=[2009],weight_cutoff=0.01)
print(semantic_network.pd_format(semantic_network.centralities()))

# Get centralities in a 2-step ego network
semantic_network.condition(years=[2009])
print(semantic_network.pd_format(semantic_network.centralities(ego_nw_tokens="president", depth=2)))


# Get a list of yearly centralities
from src.functions.measures import yearly_centralities
years=semantic_network.get_times_list()
print(yearly_centralities(semantic_network, years))


# Hierarchically cluster
levels=2
semantic_network.condition(years=[2009])
clusters=semantic_network.cluster(levels=levels)

for cl in clusters:
    print("Name: {}, Level: {}, Parent: {}, Nodes: {}".format(cl['name'],cl['level'],cl['parent'],cl['graph'].nodes))


# Cluster, but also provide node-level measures
from src.functions.node_measures import proximity, centrality
semantic_network.condition(years=[2009])
clusters=semantic_network.cluster(levels=1, to_measure=[proximity,centrality])
print(semantic_network.pd_format(clusters[0]['measures']))