from itertools import product

from src.functions.file_helpers import check_create_folder
from src.measures.measures import average_cluster_proximities, extract_all_clusters
from src.utils.logging_helpers import setup_logger
import logging
import numpy as np
from src.functions.graph_clustering import consensus_louvain, louvain_cluster
from src.classes.neo4jnw import neo4j_network

# Set a configuration path
configuration_path = '/config/config.ini'
# Settings
years = list(range(1980, 2021))
focal_token = "founder"
alter_subset = ["boss"]
alter_subset = ["ceo", "kid", "manager", "head", "sadhu", "boss", "collector", "arbitrator", "offender", "partner",
                "person", "catcher", "player", "founder", "musician", "volunteer", "golfer", "commander", "employee",
                "speaker", "coach", "candidate", "champion", "expert", "negotiator", "owner", "chief", "entrepreneur",
                "successor", "star", "salesperson", "teacher", "alpha", "cop", "performer", "editor", "agent",
                "supervisor", "chef", "builder", "consultant", "listener", "assistant", "veteran", "journalist",
                "physicist", "chair", "reformer", "facilitator", "ally", "buddy", "colleague", "enthusiast",
                "proponent", "artist", "composer", "achiever", "citizen", "researcher", "hero", "minister", "designer",
                "protagonist", "writer", "scientist", "fool", "mayor", "senator", "admiral", "statesman", "co",
                "lawyer", "middle", "prosecutor", "businessman", "billionaire", "actor", "baseman", "politician",
                "novice", "secretary", "driver", "jerk", "rebel", "lieutenant", "victim", "sergeant", "inventor",
                "front", "helm"]
alter_subset3 = ["chieftain", "enemy", "congressman", "ombudsman", "believer", "deputy", "guest", "magistrate", "heir",
                 "wizard", "hostess", "protaga", "athlete", "supervisor", "head", "emeritus", "critic", "thief", "man",
                 "golfer", "policeman", "trainer", "visitor", "specialist", "trainee", "helper", "adjunct", "prey",
                 "scholar", "dreamer", "titan", "partner", "resident", "preacher", "boxer", "successor", "reformer",
                 "prosecutor", "warlord", "rocker", "peasant", "chairs", "champ", "coward", "salesperson",
                 "comptroller",
                 "sponsor", "builder", "former", "cornerback", "colonel", "bully", "negotiator", "nun", "rebel",
                 "reporter", "hitter", "technician", "rear", "top", "warriors", "savior", "magician", "representative",
                 "president", "hillbilly", "practitioner", "sheriff", "biker", "teenager", "patriarch", "front",
                 "creature", "creator", "superstar", "archbishop", "monkey", "selector", "alpha", "player", "superman",
                 "collaborator", "villain", "bystander", "director", "bearer", "advisor", "coordinator", "entrepreneur",
                 "legislator", "keeper", "composer", "linguist", "spy", "predecessor", "priests", "recruiter",
                 "offender", "co", "newcomer", "auditor", "missionary", "researcher", "slave", "outsider",
                 "sociologist",
                 "pessimist", "publisher", "salesman", "mentor", "racer", "heads", "spectator", "guardian", "aide",
                 "ops", "sidekick", "teammate", "dean", "sergeant", "organizer", "instrumentalist", "contestant",
                 "expert", "novice", "presidency", "warrior", "valet", "geek", "adversary", "intern", "victim", "sage",
                 "liaison", "chairperson", "middle", "analyst", "minister", "chief", "crusader", "person", "bargainer",
                 "commodore", "donor", "cop", "star", "forecaster", "psychologist", "calf", "poet", "administrator",
                 "friend", "skipper", "operator", "vp", "biographer", "consultant", "counsels", "businessman", "buddy",
                 "thinker", "moderator", "kid", "dictator", "celebrity", "seeker", "benefactor", "hacker", "citizen",
                 "shortstop", "founding", "volunteer", "subordinate", "attorney", "skier", "theologian", "shooter",
                 "coach", "bulldozer", "boy", "martyr", "counselor", "skeptic", "architect", "foreigner", "geologist",
                 "therapist", "vo", "crook", "pianist", "enthusiast", "jock", "quarterback", "abolitionist", "nemesis",
                 "diplomat", "professor", "journalist", "participant", "individualist", "captain", "philanthropist",
                 "innovator", "officer", "priest", "holder", "trustee", "screenwriter", "playwright", "superintendent",
                 "listener", "undertaker", "narrator", "lover", "mathematician", "choreographer", "ringmaster",
                 "drummer", "baton", "connector", "patron", "opponent", "exec", "senior", "antagonist", "tops",
                 "biologist", "scientist", "pioneer", "artist", "optimizers", "student", "columnist", "presidents",
                 "strategist", "comrade", "alchemist", "governor", "performer", "lawyer", "apprentice", "swimmer",
                 "guy",
                 "anthropologist", "proprietor", "senator", "interpreter", "prisoner", "correspondent", "fortune",
                 "regulator", "secretary", "manager", "banker", "apex", "translator", "chancellor", "writer", "scribe",
                 "worker", "soldier", "historian", "executive", "neighbor", "chair", "chemist", "promoter",
                 "industrialist", "perfectionist", "cartoonist", "applicant", "referee", "procrastinator", "controller",
                 "receptionist", "storyteller", "educator", "examiner", "instructor", "ranger", "batter", "waiter",
                 "synthesizer", "activist", "narcissist", "trader", "stars", "integrator", "hero", "statesman",
                 "assistant", "godfather", "actor", "astronaut", "counsel", "freak", "monarch", "healer", "gardener",
                 "farmer", "musician", "champion", "fan", "associate", "persuader", "psychoanalyst", "clergy",
                 "summaries", "harasser", "physicist", "diva", "amateur", "chairman", "commander", "achiever",
                 "conductor", "junior", "fellow", "goalkeeper", "disciple", "economist", "sadhu", "philosopher",
                 "colleague", "ally", "sprinter", "adviser", "catcher", "editor", "evangelist", "starter", "accountant",
                 "observer", "developer", "vice", "millionaire", "steward", "sailor", "librarian", "asshole",
                 "cofounder", "clown", "nominee", "blogger", "tokugawa", "waitress", "bitch", "mayor", "gangster",
                 "spokesperson", "ruler", "foreman", "avatar", "baseman", "swami", "recipient", "clerk", "expat", "ceo",
                 "hulk", "latter", "arbitrator", "apostle", "contender", "boss", "cum", "broker", "craftsman",
                 "politician", "theorist", "finalist", "guru", "chro", "messenger", "commissioner", "employee",
                 "psychiatrist", "designer", "lieutenant", "rabbi", "spokesman", "investigator", "novelist", "speaker",
                 "reviewer", "teacher", "foe", "dude", "servant", "admiral", "billionaire", "fool", "collector",
                 "protector", "chef", "actress", "programmer", "stranger", "survivor", "idiot", "lobbyist", "presenter",
                 "dentist", "veteran", "founder", "confidant", "filmmaker", "sergeants", "contributor", "dancer",
                 "guitarist", "sucker", "bottom", "owner", "coachee", "fundraiser", "helm", "follower", "reader",
                 "pitcher", "tyrant", "supporter", "jerk", "protege", "engineer", "photographer", "agent", "headmaster",
                 "ambassador", "sculptor", "candidate", "corporal", "gentleman", "inventor", "protagonist", "bulldog",
                 "proponent", "solver", "driver", "frontline", "treasurer", "facilitator", "rep", "planner",
                 "commentator", "liar", "skeptics", "redhead", "author", "economists", "coauthor", "wife", "brother",
                 "dad", "sons", "cousins", "girl", "apes", "widow", "mate", "daughter", "lady", "grandparents",
                 "nephew",
                 "spouse", "male", "fleas", "father", "feminist", "nanny", "maiden", "women", "children", "youth",
                 "scout", "maternal", "son", "stepmother", "mother", "sister", "female", "uncle", "families",
                 "offspring", "woman", "daddy", "cousin", "lesbian", "mom", "grandfather", "mover", "loser", "runner",
                 "laureate", "winner", "impostor", "leader"]
alter_subset2 = ["ceo", "president", "leader", "owner", "insider", "director", "founding", "entrepreneur", "executive",
                 "father", "head", "chair", "member", "editor", "man", "professor", "consultant", "employee",
                 "innovator", "candidate", "boss", "visionary", "successor", "designer", "colleague", "builder",
                 "creator", "son", "donor", "coach", "incumbent", "husband", "salesman", "predecessor", "spokesperson",
                 "star", "governor", "wizard", "cleric", "writer", "detective", "composer", "azerbaijani", "ringmaster",
                 "economist", "co", "steward", "researcher", "alchemist", "farmer", "businessman", "cartoonist",
                 "alumnus", "associate", "deputy", "devil", "confidant", "ambassador", "actor", "advocate",
                 "songwriter",
                 "master", "photographer", "millionaire", "legend", "teller", "comptroller", "pilgrim", "alpha",
                 "visitor", "catalyst", "sprinter", "boy", "anthropologist"]

alter_subset = None
# Load Configuration file
import configparser

config = configparser.ConfigParser()
print(check_create_folder(configuration_path))
config.read(check_create_folder(configuration_path))
# Setup logging
setup_logger(config['Paths']['log'], config['General']['logging_level'], "founder_ego_clusters.py")


import os

os.environ['NUMEXPR_MAX_THREADS'] = '16'
# First, create an empty network
semantic_network = neo4j_network(config)

level_list = [5]
weight_list = [0]
cl_clutoff_list = [None, 90]
depth_list = [1]
rs_list = [100]
rev_ties_list = [False]
algolist = [louvain_cluster, consensus_louvain]
algolist = [consensus_louvain]
focal_context_list = [("zuckerberg", ["facebook", "mark", "marc"]), ("jobs", ["steve", "apple", "next"]),
                       ("gates", ["bill", "microsoft"]),
                      ("page", ["larry", "google"]),("branson", ["richard", "virgin"]),("bezos", ["jeff", "amazon"]),]
alter_set = [None]
focaladdlist = [True]
comp_ties_list = [False]
back_out_list = [False]
param_list = product(depth_list, level_list, rs_list, weight_list, rev_ties_list, comp_ties_list, cl_clutoff_list,
                     algolist, back_out_list, focaladdlist, alter_set, focal_context_list)
logging.info("------------------------------------------------")
for depth, level, rs, cutoff, rev, comp, cluster_cutoff, algo, backout, fadd, alters, fc_list in param_list:
    focal_token, context = fc_list
    logging.info("Focal token:{} Context: {}".format(focal_token,context))
    del semantic_network
    np.random.seed(rs)
    semantic_network = neo4j_network(config)
    filename = "".join(
        [config['Paths']['csv_outputs'], "/cw_", str(context[0]), "_EgoCluster_", str(focal_token), "_backout",
         str(backout), "_fadd", str(fadd), "_alters", str(str(isinstance(alters, list))), "_rev", str(rev), "_norm",
         str(comp),
         "_lev", str(level), "_cut",
         str(cutoff), "_clcut", str(cluster_cutoff), "_algo", str(algo.__name__), "_depth", str(depth), "_rs", str(rs)])
    logging.info("Network clustering: {}".format(filename))
    # Random Seed
    # df = extract_all_clusters(level=level, cutoff=cutoff, times=years, cluster_cutoff=cluster_cutoff, focal_token=focal_token,
    #                          interest_list=alter_subset, snw=semantic_network,
    #                          depth=depth, algorithm=algo, filename=filename, to_back_out=backout, add_focal_to_clusters=fadd,
    #                         compositional=comp, reverse_ties=rev, seed=rs)
    df = average_cluster_proximities(focal_token=focal_token, nw=semantic_network, levels=level, interest_list=alters,
                                     times=years, do_reverse=True,
                                     depth=depth, weight_cutoff=cutoff, cluster_cutoff=cluster_cutoff,
                                     year_by_year=False, add_focal_to_clusters=fadd,
                                     moving_average=None, filename=filename, compositional=comp, to_back_out=backout,
                                     include_all_levels=True, add_individual_nodes=True,
                                     reverse_ties=rev, seed=rs, context=context)
#### Cluster yearly proximities

focal_context_list = [("zuckerberg", ["facebook", "mark", "marc"]), ("jobs", ["steve", "apple", "next"]),
                      ("gates", ["bill", "microsoft"]),
                      ("page", ["larry", "google"]),("brinn", ["sergej", "google"]),("branson", ["richard", "virgin"]),("bezos", ["jeff", "amazon"]),]
ma_list = [(2, 0)]
level_list = [5]
weight_list = [0.0]
cl_clutoff_list = [0, 90]
depth_list = [1]
rs_list = [100]
rev_ties_list = [False]
comp_ties_list = [False]
back_out_list = [False]
# context_list=[["ceo","business","president","chairman","new"],["market","companies","total","firm","consumer"],["people","financial","organization","best","team"],["company","industry","insider","year","yes"]]

algolist = [consensus_louvain]
alter_set = [None]
focaladdlist = [True]
param_list = product(depth_list, level_list, ma_list, weight_list, rev_ties_list, comp_ties_list, rs_list,
                     cl_clutoff_list, back_out_list, focaladdlist, alter_set, focal_context_list)
logging.info("------------------------------------------------")
for depth, levels, moving_average, weight_cutoff, rev, comp, rs, cluster_cutoff, backout, fadd, alters, fc_list in param_list:
    focal_token, context = fc_list
    logging.info("Focal token:{} Context: {}".format(focal_token,context))
    del semantic_network
    np.random.seed(rs)
    semantic_network = neo4j_network(config)
    # weight_cutoff=0
    filename = "".join(
        [config['Paths']['csv_outputs'], "/", str(context[0]), "_EgoClusterYOY_nocweight_", str(focal_token),
         "_backout", str(backout), "_fadd", str(fadd), "_alters", str(str(isinstance(alters, list))), "_rev", str(rev),
         "_norm", str(comp), "_lev",
         str(levels), "_clcut",
         str(cluster_cutoff), "_cut", str(weight_cutoff), "_algo", str(algo.__name__), "_depth", str(depth), "_ma",
         str(moving_average), "_rs",
         str(rs)])
    logging.info("YOY Network clustering: {}".format(filename))
    df = average_cluster_proximities(focal_token=focal_token, nw=semantic_network, levels=levels, interest_list=alters,
                                     times=years, do_reverse=False,
                                     depth=depth, weight_cutoff=weight_cutoff, cluster_cutoff=cluster_cutoff,
                                     year_by_year=True, add_focal_to_clusters=fadd,
                                     moving_average=moving_average, filename=filename, compositional=comp,
                                     to_back_out=backout,
                                     reverse_ties=rev, seed=rs, context=context)
