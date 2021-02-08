from src.functions.file_helpers import check_create_folder
from src.utils.logging_helpers import setup_logger
import logging
import time

# Set a configuration path
configuration_path='/example/config/config.ini'
# Load Configuration file
import configparser
config = configparser.ConfigParser()
config.read(check_create_folder(configuration_path))
# Setup logging
setup_logger(config['Paths']['log'],config['General']['logging_level'] )



# First, create an empty network
from src.classes.neo4jnw import neo4j_network
semantic_network = neo4j_network(config)


print("------------------------BENCHMARK------------------------")
logging.info("------------------------BENCHMARK------------------------")


context_words=["manager","leader","ceo"]
del semantic_network
semantic_network = neo4j_network(config, connection_type="bolt")
# Condition the network
start_time = time.time()
semantic_network.condition(years=2009, context=context_words, weight_cutoff=0.1)
print("bolt one year condition, 3 context words, 0.1 cutoff %s seconds" % (time.time() - start_time))
logging.info("bolt one year condition, 3 context words, 0.1 cutoff %s seconds" % (time.time() - start_time))
print("nodes in network %i" % (len(semantic_network)))
print("ties in network %i" % (semantic_network.graph.number_of_edges()))
logging.info("nodes in network %i" % (len(semantic_network)))
logging.info("ties in network %i" % (semantic_network.graph.number_of_edges()))
print("------------------------------------------------")
logging.info("------------------------------------------------")

del semantic_network
semantic_network = neo4j_network(config, connection_type="bolt")
# Condition the network
start_time = time.time()
semantic_network.condition(years=2009, context=context_words)
print("bolt one year condition, 3 context words, NO cutoff %s seconds" % (time.time() - start_time))
logging.info("bolt one year condition, 3 context words, NO cutoff %s seconds" % (time.time() - start_time))
print("nodes in network %i" % (len(semantic_network)))
print("ties in network %i" % (semantic_network.graph.number_of_edges()))
logging.info("nodes in network %i" % (len(semantic_network)))
logging.info("ties in network %i" % (semantic_network.graph.number_of_edges()))
print("------------------------------------------------")
logging.info("------------------------------------------------")

del semantic_network
semantic_network = neo4j_network(config, connection_type="bolt")
# Condition the network
start_time = time.time()
semantic_network.condition(context=context_words, weight_cutoff=0.1)
print("bolt ALL year condition, 3 context words, 0.1 cutoff %s seconds" % (time.time() - start_time))
logging.info("bolt ALL year condition, 3 context words, 0.1 cutoff %s seconds" % (time.time() - start_time))
print("nodes in network %i" % (len(semantic_network)))
print("ties in network %i" % (semantic_network.graph.number_of_edges()))
logging.info("nodes in network %i" % (len(semantic_network)))
logging.info("ties in network %i" % (semantic_network.graph.number_of_edges()))
print("------------------------------------------------")
logging.info("------------------------------------------------")

del semantic_network
semantic_network = neo4j_network(config, connection_type="bolt")
# Condition the network
start_time = time.time()
semantic_network.condition(context=context_words)
print("bolt ALL year condition, 3 context words, NO cutoff %s seconds" % (time.time() - start_time))
logging.info("bolt ALL year condition, 3 context words, NO cutoff %s seconds" % (time.time() - start_time))
print("nodes in network %i" % (len(semantic_network)))
print("ties in network %i" % (semantic_network.graph.number_of_edges()))
logging.info("nodes in network %i" % (len(semantic_network)))
logging.info("ties in network %i" % (semantic_network.graph.number_of_edges()))
print("------------------------------------------------")
logging.info("------------------------------------------------")

del semantic_network
semantic_network = neo4j_network(config, connection_type="bolt")
# Condition the network
start_time = time.time()
semantic_network.condition()
print("bolt ALL years condition, 100000, no cutoff %s seconds" % (time.time() - start_time))
logging.info("bolt ALL years condition, 100000, no cutoff %s seconds" % (time.time() - start_time))
print("nodes in network %i" % (len(semantic_network)))
print("ties in network %i" % (semantic_network.graph.number_of_edges()))
logging.info("nodes in network %i" % (len(semantic_network)))
logging.info("ties in network %i" % (semantic_network.graph.number_of_edges()))
print("------------------------------------------------")
logging.info("------------------------------------------------")


# Analysis examples
# Setup network on processed database
start_time = time.time()
semantic_network = neo4j_network(config)
print("http Setup  %s seconds" % (time.time() - start_time))
logging.info("http Setup  %s seconds" % (time.time() - start_time))
print("------------------------------------------------")
logging.info("------------------------------------------------")
start_time = time.time()
# Query one token directly from the database
print(semantic_network['president'])
print("http query one token %s seconds" % (time.time() - start_time))
logging.info("http query one token %s seconds" % (time.time() - start_time))
print("------------------------------------------------")
logging.info("------------------------------------------------")
del semantic_network
start_time = time.time()
semantic_network = neo4j_network(config, connection_type="bolt")
print("bolt Setup  %s seconds" % (time.time() - start_time))
logging.info("bolt Setup  %s seconds" % (time.time() - start_time))
print("------------------------------------------------")
logging.info("------------------------------------------------")

start_time = time.time()
# Query one token directly from the database
print(semantic_network['president'])
print("bolt query one token %s seconds" % (time.time() - start_time))
logging.info("bolt query one token %s seconds" % (time.time() - start_time))
print("------------------------------------------------")
logging.info("------------------------------------------------")
del semantic_network
semantic_network = neo4j_network(config)
print(semantic_network.pd_format(semantic_network.proximities(['president'],years=2000)))
print("http proximity %s seconds" % (time.time() - start_time))
logging.info("http proximity %s seconds" % (time.time() - start_time))
print("------------------------------------------------")
logging.info("------------------------------------------------")
del semantic_network
semantic_network = neo4j_network(config, connection_type="bolt")
print(semantic_network.pd_format(semantic_network.proximities(['president'],years=2000)))
print("bolt proximity %s seconds" % (time.time() - start_time))
logging.info("bolt proximity %s seconds" % (time.time() - start_time))
print("------------------------------------------------")
logging.info("------------------------------------------------")
del semantic_network
semantic_network = neo4j_network(config)
print(semantic_network.pd_format(semantic_network.proximities(['president','leader','ceo','obama'],years=2009)))
print("http proximities %s seconds" % (time.time() - start_time))
logging.info("http proximities %s seconds" % (time.time() - start_time))
print("------------------------------------------------")
logging.info("------------------------------------------------")
del semantic_network
semantic_network = neo4j_network(config, connection_type="bolt")
print(semantic_network.pd_format(semantic_network.proximities(['president','leader','ceo','obama'],years=2009)))
print("bolt proximities %s seconds" % (time.time() - start_time))
logging.info("bolt proximities %s seconds" % (time.time() - start_time))
print("------------------------------------------------")
logging.info("------------------------------------------------")
del semantic_network
semantic_network = neo4j_network(config)
# Condition the network
start_time = time.time()
semantic_network.condition(tokens=["leader"],years=1995, depth=1,weight_cutoff=0.1 )
print("http 1 step ego %s seconds" % (time.time() - start_time))
logging.info("http 1 step ego, cutoff 0.1, %s seconds" % (time.time() - start_time))
print("nodes in network %i" % (len(semantic_network)))
print("ties in network %i" % (semantic_network.graph.number_of_edges()))
logging.info("nodes in network %i" % (len(semantic_network)))
logging.info("ties in network %i" % (semantic_network.graph.number_of_edges()))


del semantic_network
semantic_network = neo4j_network(config, connection_type="bolt")
# Condition the network
start_time = time.time()
semantic_network.condition(tokens=["leader"],years=1995, depth=1,weight_cutoff=0.1 )
print("bolt 1 step ego, cutoff 0.1, %s seconds" % (time.time() - start_time))
logging.info("bolt 1 step ego, cutoff 0.1, %s seconds" % (time.time() - start_time))
print("nodes in network %i" % (len(semantic_network)))
print("ties in network %i" % (semantic_network.graph.number_of_edges()))
logging.info("nodes in network %i" % (len(semantic_network)))
logging.info("ties in network %i" % (semantic_network.graph.number_of_edges()))
print("------------------------------------------------")
logging.info("------------------------------------------------")
del semantic_network
semantic_network = neo4j_network(config)
# Condition the network
start_time = time.time()
semantic_network.condition(tokens=["president"],years=2009, depth=2,weight_cutoff=0.1 )
print("http 2 step ego, cutoff 0.1,  %s seconds" % (time.time() - start_time))
logging.info("http 2 step ego, cutoff 0.1,  %s seconds" % (time.time() - start_time))
print("nodes in network %i" % (len(semantic_network)))
print("ties in network %i" % (semantic_network.graph.number_of_edges()))
logging.info("nodes in network %i" % (len(semantic_network)))
logging.info("ties in network %i" % (semantic_network.graph.number_of_edges()))

del semantic_network
semantic_network = neo4j_network(config, connection_type="bolt")
# Condition the network
start_time = time.time()
semantic_network.condition(tokens=["president"],years=2009, depth=2,weight_cutoff=0.1)
print("bolt 2 step ego, cutoff 0.1,  %s seconds" % (time.time() - start_time))
logging.info("bolt 2 step ego, cutoff 0.1,  %s seconds" % (time.time() - start_time))
print("nodes in network %i" % (len(semantic_network)))
print("ties in network %i" % (semantic_network.graph.number_of_edges()))
logging.info("nodes in network %i" % (len(semantic_network)))
logging.info("ties in network %i" % (semantic_network.graph.number_of_edges()))
print("------------------------------------------------")
logging.info("------------------------------------------------")
del semantic_network
semantic_network = neo4j_network(config, connection_type="bolt")
# Condition the network
start_time = time.time()
semantic_network.condition(tokens=["manager"],years=1999, depth=3,weight_cutoff=0.1 )
print("bolt 3 step ego, cutoff 0.1,  %s seconds" % (time.time() - start_time))
logging.info("bolt 3 step ego, cutoff 0.1,  %s seconds" % (time.time() - start_time))
print("nodes in network %i" % (len(semantic_network)))
print("ties in network %i" % (semantic_network.graph.number_of_edges()))
logging.info("nodes in network %i" % (len(semantic_network)))
logging.info("ties in network %i" % (semantic_network.graph.number_of_edges()))

del semantic_network
semantic_network = neo4j_network(config)
# Condition the network
start_time = time.time()
semantic_network.condition(tokens=["manager"],years=1999, depth=3,weight_cutoff=0.1 )
print("http 3 step ego, cutoff 0.1,  %s seconds" % (time.time() - start_time))
logging.info("http 3 step ego, cutoff 0.1,  %s seconds" % (time.time() - start_time))
print("nodes in network %i" % (len(semantic_network)))
print("ties in network %i" % (semantic_network.graph.number_of_edges()))
logging.info("nodes in network %i" % (len(semantic_network)))
logging.info("ties in network %i" % (semantic_network.graph.number_of_edges()))
print("------------------------------------------------")
logging.info("------------------------------------------------")
del semantic_network



print("------------------------------------------------")
logging.info("------------------------------------------------")
semantic_network = neo4j_network(config)
# Condition the network
start_time = time.time()
semantic_network.condition(tokens=["leader"],years=2006, depth=3,weight_cutoff=0.01 )
print("http 3 step ego, cutoff 0.01,  %s seconds" % (time.time() - start_time))
logging.info("http 3 step ego, cutoff 0.01,  %s seconds" % (time.time() - start_time))
print("nodes in network %i" % (len(semantic_network)))
print("ties in network %i" % (semantic_network.graph.number_of_edges()))
logging.info("nodes in network %i" % (len(semantic_network)))
logging.info("ties in network %i" % (semantic_network.graph.number_of_edges()))

del semantic_network
semantic_network = neo4j_network(config, connection_type="bolt")
# Condition the network
start_time = time.time()
semantic_network.condition(tokens=["leader"],years=2006, depth=3,weight_cutoff=0.01 )
print("bolt 3 step ego, cutoff 0.01,  %s seconds" % (time.time() - start_time))
logging.info("bolt 3 step ego, cutoff0.01,  %s seconds" % (time.time() - start_time))
print("nodes in network %i" % (len(semantic_network)))
print("ties in network %i" % (semantic_network.graph.number_of_edges()))
logging.info("nodes in network %i" % (len(semantic_network)))
logging.info("ties in network %i" % (semantic_network.graph.number_of_edges()))
print("------------------------------------------------")
logging.info("------------------------------------------------")
del semantic_network



semantic_network = neo4j_network(config)
# Condition the network
start_time = time.time()
semantic_network.condition(years=2009, batchsize=1000)
print("http one year condition, 1000, no cutoff %s seconds" % (time.time() - start_time))
logging.info("http one year condition,  1000, no cutoff %s seconds" % (time.time() - start_time))
print("nodes in network %i" % (len(semantic_network)))
print("ties in network %i" % (semantic_network.graph.number_of_edges()))
logging.info("nodes in network %i" % (len(semantic_network)))
logging.info("ties in network %i" % (semantic_network.graph.number_of_edges()))
print("------------------------------------------------")
logging.info("------------------------------------------------")
del semantic_network
semantic_network = neo4j_network(config, connection_type="bolt")
# Condition the network
start_time = time.time()
semantic_network.condition(years=2009, batchsize=1000)
print("bolt one year condition, 1000, no cutoff %s seconds" % (time.time() - start_time))
logging.info("bolt one year condition,  1000, no cutoff %s seconds" % (time.time() - start_time))
print("nodes in network %i" % (len(semantic_network)))
print("ties in network %i" % (semantic_network.graph.number_of_edges()))
logging.info("nodes in network %i" % (len(semantic_network)))
logging.info("ties in network %i" % (semantic_network.graph.number_of_edges()))
print("------------------------------------------------")
logging.info("------------------------------------------------")


# Condition the network
del semantic_network
semantic_network = neo4j_network(config)
start_time = time.time()
semantic_network.condition(years=2009, batchsize=100000)
print("http one year condition, 100000, no cutoff %s seconds" % (time.time() - start_time))
logging.info("http one year condition,  100000, no cutoff %s seconds" % (time.time() - start_time))
print("nodes in network %i" % (len(semantic_network)))
print("ties in network %i" % (semantic_network.graph.number_of_edges()))
logging.info("nodes in network %i" % (len(semantic_network)))
logging.info("ties in network %i" % (semantic_network.graph.number_of_edges()))
print("------------------------------------------------")
logging.info("------------------------------------------------")
del semantic_network
semantic_network = neo4j_network(config, connection_type="bolt")
# Condition the network
start_time = time.time()
semantic_network.condition(years=2009, batchsize=100000)
print("bolt one year condition, 100000, no cutoff %s seconds" % (time.time() - start_time))
logging.info("bolt one year condition,  100000, no cutoff %s seconds" % (time.time() - start_time))
print("nodes in network %i" % (len(semantic_network)))
print("ties in network %i" % (semantic_network.graph.number_of_edges()))
logging.info("nodes in network %i" % (len(semantic_network)))
logging.info("ties in network %i" % (semantic_network.graph.number_of_edges()))
print("------------------------------------------------")
logging.info("------------------------------------------------")

# Condition the network
del semantic_network
semantic_network = neo4j_network(config, connection_type="bolt")
# Condition the network
start_time = time.time()
semantic_network.condition(years=2009, batchsize=100,weight_cutoff=0.1)
print("bolt 1 year condition, batch 100, 0.1 cutoff %s seconds" % (time.time() - start_time))
logging.info("bolt 1 year condition, batch 100, 0.1 cutoff %s seconds" % (time.time() - start_time))
print("nodes in network %i" % (len(semantic_network)))
print("ties in network %i" % (semantic_network.graph.number_of_edges()))
logging.info("nodes in network %i" % (len(semantic_network)))
logging.info("ties in network %i" % (semantic_network.graph.number_of_edges()))
del semantic_network
semantic_network = neo4j_network(config)
start_time = time.time()
semantic_network.condition(years=2009, batchsize=100,weight_cutoff=0.1)
print("http 1 year condition, batch 100, 0.1 cutoff %s seconds" % (time.time() - start_time))
logging.info("http 1 year condition, batch 100, 0.1 cutoff %s seconds" % (time.time() - start_time))
print("nodes in network %i" % (len(semantic_network)))
print("ties in network %i" % (semantic_network.graph.number_of_edges()))
logging.info("nodes in network %i" % (len(semantic_network)))
logging.info("ties in network %i" % (semantic_network.graph.number_of_edges()))
print("------------------------------------------------")
logging.info("------------------------------------------------")


# Condition the network
del semantic_network
semantic_network = neo4j_network(config)
start_time = time.time()
semantic_network.condition(years=2011, batchsize=10,weight_cutoff=0.1)
print("http 1 year condition, batch 10, 0.1 cutoff %s seconds" % (time.time() - start_time))
logging.info("http 1 year condition, batch 10, 0.1 cutoff %s seconds" % (time.time() - start_time))
print("nodes in network %i" % (len(semantic_network)))
print("ties in network %i" % (semantic_network.graph.number_of_edges()))
logging.info("nodes in network %i" % (len(semantic_network)))
logging.info("ties in network %i" % (semantic_network.graph.number_of_edges()))
del semantic_network
semantic_network = neo4j_network(config, connection_type="bolt")
# Condition the network
start_time = time.time()
semantic_network.condition(years=2011, batchsize=10,weight_cutoff=0.1)
print("bolt 1 year condition, batch 10, 0.1 cutoff %s seconds" % (time.time() - start_time))
logging.info("bolt 1 year condition, batch 10, 0.1 cutoff %s seconds" % (time.time() - start_time))
print("nodes in network %i" % (len(semantic_network)))
print("ties in network %i" % (semantic_network.graph.number_of_edges()))
logging.info("nodes in network %i" % (len(semantic_network)))
logging.info("ties in network %i" % (semantic_network.graph.number_of_edges()))
print("------------------------------------------------")
logging.info("------------------------------------------------")


# Condition the network
del semantic_network
semantic_network = neo4j_network(config, connection_type="bolt")
# Condition the network
start_time = time.time()
semantic_network.condition(years=2012, batchsize=1000,weight_cutoff=0.1)
print("bolt 1 year condition, batch 1000, 0.1 cutoff %s seconds" % (time.time() - start_time))
logging.info("bolt 1 year condition, batch 1000, 0.1 cutoff %s seconds" % (time.time() - start_time))
print("nodes in network %i" % (len(semantic_network)))
print("ties in network %i" % (semantic_network.graph.number_of_edges()))
logging.info("nodes in network %i" % (len(semantic_network)))
logging.info("ties in network %i" % (semantic_network.graph.number_of_edges()))
del semantic_network
semantic_network = neo4j_network(config)
start_time = time.time()
semantic_network.condition(years=2012, batchsize=1000,weight_cutoff=0.1)
print("http 1 year condition, batch 1000, 0.1 cutoff %s seconds" % (time.time() - start_time))
logging.info("http 1 year condition, batch 1000, 0.1 cutoff %s seconds" % (time.time() - start_time))
print("nodes in network %i" % (len(semantic_network)))
print("ties in network %i" % (semantic_network.graph.number_of_edges()))
logging.info("nodes in network %i" % (len(semantic_network)))
logging.info("ties in network %i" % (semantic_network.graph.number_of_edges()))
print("------------------------------------------------")
logging.info("------------------------------------------------")


# Condition the network
del semantic_network
semantic_network = neo4j_network(config)
start_time = time.time()
semantic_network.condition(years=2013, batchsize=10000,weight_cutoff=0.1)
print("http 1 year condition, batch 10000, 0.1 cutoff %s seconds" % (time.time() - start_time))
logging.info("http 1 year condition, batch 10000, 0.1 cutoff %s seconds" % (time.time() - start_time))
print("nodes in network %i" % (len(semantic_network)))
print("ties in network %i" % (semantic_network.graph.number_of_edges()))
logging.info("nodes in network %i" % (len(semantic_network)))
logging.info("ties in network %i" % (semantic_network.graph.number_of_edges()))
del semantic_network
semantic_network = neo4j_network(config, connection_type="bolt")
# Condition the network
start_time = time.time()
semantic_network.condition(years=2013, batchsize=10000,weight_cutoff=0.1)
print("bolt 1 year condition, batch 10000, 0.1 cutoff %s seconds" % (time.time() - start_time))
logging.info("bolt 1 year condition, batch 10000, 0.1 cutoff %s seconds" % (time.time() - start_time))
print("nodes in network %i" % (len(semantic_network)))
print("ties in network %i" % (semantic_network.graph.number_of_edges()))
logging.info("nodes in network %i" % (len(semantic_network)))
logging.info("ties in network %i" % (semantic_network.graph.number_of_edges()))
print("------------------------------------------------")
logging.info("------------------------------------------------")


# Condition the network
del semantic_network
semantic_network = neo4j_network(config)
start_time = time.time()
semantic_network.condition(years=2014, batchsize=100000,weight_cutoff=0.1)
print("http 1 year condition, batch 100000, 0.1 cutoff %s seconds" % (time.time() - start_time))
logging.info("http 1 year condition, batch 100000, 0.1 cutoff %s seconds" % (time.time() - start_time))
print("nodes in network %i" % (len(semantic_network)))
print("ties in network %i" % (semantic_network.graph.number_of_edges()))
logging.info("nodes in network %i" % (len(semantic_network)))
logging.info("ties in network %i" % (semantic_network.graph.number_of_edges()))
del semantic_network
semantic_network = neo4j_network(config, connection_type="bolt")
# Condition the network
start_time = time.time()
semantic_network.condition(years=2014, batchsize=100000,weight_cutoff=0.1)
print("bolt 1 year condition, batch 100000, 0.1 cutoff %s seconds" % (time.time() - start_time))
logging.info("bolt 1 year condition, batch 100000, 0.1 cutoff %s seconds" % (time.time() - start_time))
print("nodes in network %i" % (len(semantic_network)))
print("ties in network %i" % (semantic_network.graph.number_of_edges()))
logging.info("nodes in network %i" % (len(semantic_network)))
logging.info("ties in network %i" % (semantic_network.graph.number_of_edges()))
print("------------------------------------------------")
logging.info("------------------------------------------------")




# Condition the network
del semantic_network
semantic_network = neo4j_network(config)
start_time = time.time()
semantic_network.condition(years=2009, batchsize=100,weight_cutoff=0.01)
print("http 1 year condition, batch 100, 0.01 cutoff %s seconds" % (time.time() - start_time))
logging.info("http 1 year condition, batch 100, 0.01 cutoff %s seconds" % (time.time() - start_time))
print("nodes in network %i" % (len(semantic_network)))
print("ties in network %i" % (semantic_network.graph.number_of_edges()))
logging.info("nodes in network %i" % (len(semantic_network)))
logging.info("ties in network %i" % (semantic_network.graph.number_of_edges()))
del semantic_network
semantic_network = neo4j_network(config, connection_type="bolt")
# Condition the network
start_time = time.time()
semantic_network.condition(years=2009, batchsize=100,weight_cutoff=0.01)
print("bolt 1 year condition, batch 100, 0.01 cutoff %s seconds" % (time.time() - start_time))
logging.info("bolt 1 year condition, batch 100, 0.01 cutoff %s seconds" % (time.time() - start_time))
print("nodes in network %i" % (len(semantic_network)))
print("ties in network %i" % (semantic_network.graph.number_of_edges()))
logging.info("nodes in network %i" % (len(semantic_network)))
logging.info("ties in network %i" % (semantic_network.graph.number_of_edges()))
print("------------------------------------------------")
logging.info("------------------------------------------------")


# Condition the network
del semantic_network
semantic_network = neo4j_network(config)
start_time = time.time()
semantic_network.condition(years=2010, batchsize=100,weight_cutoff=0.5)
print("http 1 year condition, batch 100, 0.5 cutoff %s seconds" % (time.time() - start_time))
logging.info("http 1 year condition, batch 100, 0.5 cutoff %s seconds" % (time.time() - start_time))
print("nodes in network %i" % (len(semantic_network)))
print("ties in network %i" % (semantic_network.graph.number_of_edges()))
logging.info("nodes in network %i" % (len(semantic_network)))
logging.info("ties in network %i" % (semantic_network.graph.number_of_edges()))
del semantic_network
semantic_network = neo4j_network(config, connection_type="bolt")
# Condition the network
start_time = time.time()
semantic_network.condition(years=2010, batchsize=100,weight_cutoff=0.5)
print("bolt 1 year condition, batch 100, 0.5 cutoff %s seconds" % (time.time() - start_time))
logging.info("bolt 1 year condition, batch 100, 0.5 cutoff %s seconds" % (time.time() - start_time))
print("nodes in network %i" % (len(semantic_network)))
print("ties in network %i" % (semantic_network.graph.number_of_edges()))
logging.info("nodes in network %i" % (len(semantic_network)))
logging.info("ties in network %i" % (semantic_network.graph.number_of_edges()))
print("------------------------------------------------")
logging.info("------------------------------------------------")


# Condition the network
del semantic_network
semantic_network = neo4j_network(config, connection_type="bolt")
# Condition the network
start_time = time.time()
semantic_network.condition(batchsize=100000,weight_cutoff=0.1)
print("bolt ALL year condition, batch 100000, 0.1 cutoff %s seconds" % (time.time() - start_time))
logging.info("bolt ALL year condition, batch 100000, 0.1 cutoff %s seconds" % (time.time() - start_time))
print("nodes in network %i" % (len(semantic_network)))
print("ties in network %i" % (semantic_network.graph.number_of_edges()))
logging.info("nodes in network %i" % (len(semantic_network)))
logging.info("ties in network %i" % (semantic_network.graph.number_of_edges()))
del semantic_network
semantic_network = neo4j_network(config)
start_time = time.time()
semantic_network.condition(batchsize=100000,weight_cutoff=0.1)
print("http ALL year condition, batch 100000, 0.1 cutoff %s seconds" % (time.time() - start_time))
logging.info("http ALL year condition, batch 100000, 0.1 cutoff %s seconds" % (time.time() - start_time))
print("nodes in network %i" % (len(semantic_network)))
print("ties in network %i" % (semantic_network.graph.number_of_edges()))
logging.info("nodes in network %i" % (len(semantic_network)))
logging.info("ties in network %i" % (semantic_network.graph.number_of_edges()))
print("------------------------------------------------")
logging.info("------------------------------------------------")


# Condition the network
del semantic_network
semantic_network = neo4j_network(config)
semantic_network.set_norm_ties()
start_time = time.time()
semantic_network.condition(years=2009, batchsize=100000,weight_cutoff=0.1)
print("http NORMED 1 year condition, batch 100000, 0.1 cutoff %s seconds" % (time.time() - start_time))
logging.info("http NORMED 1 year condition, batch 100000, 0.1 cutoff %s seconds" % (time.time() - start_time))
print("nodes in network %i" % (len(semantic_network)))
print("ties in network %i" % (semantic_network.graph.number_of_edges()))
logging.info("nodes in network %i" % (len(semantic_network)))
logging.info("ties in network %i" % (semantic_network.graph.number_of_edges()))
del semantic_network
semantic_network = neo4j_network(config, connection_type="bolt")
semantic_network.set_norm_ties()
# Condition the network
start_time = time.time()
semantic_network.condition(years=2009, batchsize=100000,weight_cutoff=0.1)
print("bolt NORMED 1 year condition, batch 100000, 0.1 cutoff %s seconds" % (time.time() - start_time))
logging.info("bolt NORMED 1 year condition, batch 100000, 0.1 cutoff %s seconds" % (time.time() - start_time))
print("nodes in network %i" % (len(semantic_network)))
print("ties in network %i" % (semantic_network.graph.number_of_edges()))
logging.info("nodes in network %i" % (len(semantic_network)))
logging.info("ties in network %i" % (semantic_network.graph.number_of_edges()))
print("------------------------------------------------")
logging.info("------------------------------------------------")

# Switch to backout measure
del semantic_network
semantic_network = neo4j_network(config)
semantic_network.condition(years=2010, batchsize=100000,weight_cutoff=0.1)
start_time = time.time()
semantic_network.to_backout()
print("http to backout %s seconds" % (time.time() - start_time))
logging.info("http to backout %s seconds" % (time.time() - start_time))
# Switch to backout measure
del semantic_network
semantic_network = neo4j_network(config, connection_type="bolt")
semantic_network.condition(years=2010, batchsize=100000,weight_cutoff=0.1)
start_time = time.time()
semantic_network.to_backout()
print("bolt to backout %s seconds" % (time.time() - start_time))
logging.info("bolt to backout %s seconds" % (time.time() - start_time))
print("------------------------------------------------")
logging.info("------------------------------------------------")


# Get a list of centralities
del semantic_network
semantic_network = neo4j_network(config)
start_time = time.time()
print(semantic_network.pd_format(semantic_network.centralities(weight_cutoff=0.1)))
print("http all years, centralities, cutoff 0.1, %s seconds" % (time.time() - start_time))
logging.info("http all years, centralities, cutoff 0.1, %s seconds" % (time.time() - start_time))
# Switch to backout measure
del semantic_network
semantic_network = neo4j_network(config, connection_type="bolt")
start_time = time.time()
print(semantic_network.pd_format(semantic_network.centralities(weight_cutoff=0.1)))
print("bolt all years, centralities, cutoff 0.1, %s seconds" % (time.time() - start_time))
logging.info("bolt all years, centralities, cutoff 0.1, %s seconds" % (time.time() - start_time))
print("------------------------------------------------")
logging.info("------------------------------------------------")

# Hierarchically cluster
del semantic_network
semantic_network = neo4j_network(config)
levels=2
semantic_network.condition(years=2009, weight_cutoff=0.1)
start_time = time.time()
clusters=semantic_network.cluster(levels=levels)
for cl in clusters:
    print("Name: {}, Level: {}, Parent: {}, Nodes: {}".format(cl['name'],cl['level'],cl['parent'],cl['graph'].nodes))
print("http one year 2-lvl clustering, cutoff 0.1, %s seconds" % (time.time() - start_time))
logging.info("http one year 2-lvl clustering, cutoff 0.1, %s seconds" % (time.time() - start_time))
del semantic_network
semantic_network = neo4j_network(config, connection_type="bolt")
levels=2
semantic_network.condition(years=2009, weight_cutoff=0.1)
start_time = time.time()
clusters=semantic_network.cluster(levels=levels)
for cl in clusters:
    if len(cl['graph'].nodes)>2:
        print("Name: {}, Level: {}, Parent: {}, Nodes: {}".format(cl['name'],cl['level'],cl['parent'],cl['graph'].nodes))
print("bolt one year 2-lvl clustering, cutoff 0.1, %s seconds" % (time.time() - start_time))
logging.info("bolt one year 2-lvl clustering, cutoff 0.1, %s seconds" % (time.time() - start_time))
print("------------------------------------------------")
logging.info("------------------------------------------------")



# Cluster, but also provide node-level measures
from src.functions.node_measures import proximity, centrality
del semantic_network
semantic_network = neo4j_network(config)
start_time = time.time()
semantic_network.condition(years=2009, weight_cutoff=0.1)
clusters=semantic_network.cluster(levels=2, to_measure=[proximity,centrality])
print(semantic_network.pd_format(clusters[0]['measures']))
print("http one year 2-lvl clustering+measures, cutoff 0.1, %s seconds" % (time.time() - start_time))
logging.info("http one year 2-lvl clustering+measures, cutoff 0.1, %s seconds" % (time.time() - start_time))
# Cluster, but also provide node-level measures
del semantic_network
semantic_network = neo4j_network(config, connection_type="bolt")
start_time = time.time()
semantic_network.condition(years=2009, weight_cutoff=0.1)
clusters=semantic_network.cluster(levels=2, to_measure=[proximity,centrality])
print(semantic_network.pd_format(clusters[0]['measures']))
print("bolt one year 2-lvl clustering+measures, cutoff 0.1, %s seconds" % (time.time() - start_time))
logging.info("bolt one year 2-lvl clustering+measures, cutoff 0.1, %s seconds" % (time.time() - start_time))