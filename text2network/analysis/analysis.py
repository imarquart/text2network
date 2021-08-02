from src.functions.file_helpers import check_create_folder
from src.utils.logging_helpers import setup_logger
import logging
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Set a configuration path
configuration_path='/config/config.ini'
# Load Configuration file
import configparser
config = configparser.ConfigParser()
print(check_create_folder(configuration_path))
config.read(check_create_folder(configuration_path))
# Setup logging
setup_logger(config['Paths']['log'],config['General']['logging_level'],"analysis")



# First, create an empty network
from src.classes.neo4jnw import neo4j_network
semantic_network = neo4j_network(config)

#years=np.array(semantic_network.get_times_list())
#years=-np.sort(-years)
logging.info("------------------------------------------------")
years=range(1980,2020)
focal_words=["leader","manager"]
focal_words2=["leader"]
alter_subset=["boss","coach","consultant","expert","mentor","superior"]
proxlist=[]
centlist=[]
for year in years:
    year1=int(year)-1
    year2=int(year)
    year3=int(year)+1
    if year==1980:
        yvec=[year2,year3]
    elif year==2020:
        yvec=[year1,year2]
    else:
        yvec=[year1,year2,year3]

    yvec=year2
    semantic_network.condition(years=yvec,weight_cutoff=0)
    #semantic_network.to_backout()
    logging.info("Years {}".format(yvec))
    asdf=semantic_network.pd_format(semantic_network.proximities(focal_tokens=focal_words2, alter_subset=alter_subset))[0]
    asdf["normed"]=asdf.leader/np.sum(asdf.leader)
    proxlist.append(asdf.leader)
    logging.info(asdf)
    #logging.info(semantic_network.pd_format(semantic_network.proximities(focal_tokens=focal_words,alter_subset=alter_subset)))
    cents=semantic_network.pd_format(semantic_network.centralities(focal_tokens=focal_words))
    logging.info(cents)
    centlist.append(cents)

    logging.info("------------------------------------------------")


centlist = [x[0] for x in centlist]
centlist1 = [x.normedPageRank for x in centlist]
centlist_df= [x[['leader','manager']] for x in centlist1]
centlist_df = pd.concat(centlist_df, axis=1)
centlist_df.columns=years

plt.plot(centlist_df.T)
plt.show()

proxlist_df = [x.reindex(alter_subset) for x in proxlist]
proxlist_df = pd.concat(proxlist_df, axis=1)
proxlist_df.columns=years
proxlist_df=proxlist_df.T
proxlist_df=proxlist_df.fillna(0)
proxlist_perc=proxlist_df.div(proxlist_df.sum(axis=1), axis=0)
proxlist_perc=proxlist_perc.fillna(0)


proxlist_df.plot.area()
plt.legend(loc="lower center",ncol=6,bbox_to_anchor=(0.5, 1.05))

plt.plot(proxlist_perc)
plt.show()
