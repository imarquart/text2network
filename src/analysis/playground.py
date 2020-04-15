from NLP.src.neo4j_network import neo4j_network
import logging
import pandas as pd
import numpy as np

if __name__ == "__main__":

    db_uri = "http://localhost:7474"
    db_pwd = ('neo4j', 'nlp')
    neo_creds = (db_uri, db_pwd)
    neograph = neo4j_network(neo_creds, graph_direction="REVERSE")

    years = range(1990, 2013)
    year_list=[]
    hel_p_list=[]
    hel_c_list=[]
    for year in years:
        logging.info("---------- Starting year %i ----------" % year)

        y_start=int(''.join([str(max(years[0],year-1)), "0101"]))
        y_end=int(''.join([str(min(years[-1],year+1)), "0101"]))
        year_int={'start':y_start, 'end': y_end}
        year_var =int((year_int['end'] + year_int['start']) / 2)
        w1 = 'leader'
        w2 = 'ceo'
        w3 = 'quarterback'
        neograph.condition(year_int, [w1, w2, w3], weight_cutoff=0.001)
        id_leader = neograph.ensure_ids(w1)
        id_player = neograph.ensure_ids(w2)
        id_coach = neograph.ensure_ids(w3)

        id_list=[id_leader,id_player,id_coach]
        nb_df={}
        for idx in id_list:
            node_dict=neograph.graph[idx]
            neighbors=list(node_dict)
            tp=[(n,neograph.ensure_tokens(n),node_dict[n][year_var]['weight']) for n in neighbors]
            df=pd.DataFrame(tp,columns=['id','token','proximity']).sort_values('proximity',ascending=False)
            #df.set_index("id", inplace=True)
            #df.set_index("token", inplace=True)
            df['nprox']=df.proximity/sum(df.proximity)
            nb_df.update({idx:df})

        # overlap
        overlap_leader_player = pd.merge(nb_df[id_leader], nb_df[id_player], on=["id", "token"], suffixes=("leader", "player"), how="outer")
        overlap_leader_coach = pd.merge(nb_df[id_leader], nb_df[id_coach], on=["id", "token"], suffixes=("leader", "coach"), how="outer")
        overlap_leader_player=overlap_leader_player.fillna(0)
        overlap_leader_coach=overlap_leader_coach.fillna(0)
        hel_leader_player= np.sum(np.sqrt(np.array(np.multiply(overlap_leader_player.nproxleader,overlap_leader_player.nproxplayer))))
        hel_leader_coach= np.sum(np.sqrt(np.array(np.multiply(overlap_leader_coach.nproxleader,overlap_leader_coach.nproxcoach))))

        year_list.append(year)
        hel_c_list.append(hel_leader_coach)
        hel_p_list.append(hel_leader_player)

        overlap_leader_coach = pd.merge(nb_df[id_leader], nb_df[id_coach], on=["id", "token"], suffixes=("leader", "coach"))


        neograph.decondition()

    results=pd.DataFrame({'year': year_list, 'hel_coach': hel_c_list, 'hel_player': hel_p_list})
    results.to_excel("E:/NLP/cluster_xls/coca.xlsx",sheet_name='distances')


    db_uri = "http://localhost:7474"
    db_pwd = ('neo4j', 'nlp')
    neo_creds = (db_uri, db_pwd)
    neograph = neo4j_network(neo_creds, graph_direction="REVERSE")


    y_start = int(''.join([str(years[0]), "0101"]))
    y_end = int(''.join([str(years[-1]), "0101"]))
    year_int = {'start': y_start, 'end': y_end}
    year_var = int((year_int['end'] + year_int['start']) / 2)
    w1='leader'
    w2='ceo'
    w3='coach'
    neograph.condition(year_int, [w1, w2, w3], weight_cutoff=0.001)
    id_leader = neograph.ensure_ids(w1)
    id_player = neograph.ensure_ids(w2)
    id_coach = neograph.ensure_ids(w3)

    id_list = [id_leader, id_player, id_coach]
    nb_df = {}
    for idx in id_list:
        node_dict = neograph.graph[idx]
        neighbors = list(node_dict)
        tp = [(n, neograph.ensure_tokens(n), node_dict[n][year_var]['weight']) for n in neighbors]
        df = pd.DataFrame(tp, columns=['id', 'token', 'proximity']).sort_values('proximity', ascending=False)
        # df.set_index("id", inplace=True)
        # df.set_index("token", inplace=True)
        df['nprox'] = df.proximity / sum(df.proximity)
        nb_df.update({idx: df})
    neograph.decondition()

    coach1=nb_df[id_coach].copy()
    coach2=nb_df[id_coach].copy()
    leader=nb_df[id_leader].copy()


    coach1.to_excel("E:/NLP/cluster_xls/coach_hbr.xlsx",sheet_name='distances')
    leader.to_excel("E:/NLP/cluster_xls/leader_hbr.xlsx",sheet_name='distances')