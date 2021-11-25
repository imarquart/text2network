from typing import Optional, Union

import numpy as np
import pandas as pd

from text2network.classes.neo4jnw import neo4j_network
from typing import Optional, Union

import numpy as np
import pandas as pd

from text2network.classes.neo4jnw import neo4j_network


# from src.classes import neo4jnw
# from src.classes.neo4jnw import neo4j_network
#


def extract_dyadic_context(snw: neo4j_network, focal_substitutes: Optional[Union[list, str, int]] = None,
                           focal_occurrences: Optional[Union[list, str, int]] = None, groups: Optional[list] = None,
                           context_pos: Optional[str] = None, times: Union[list, int] = None,
                           context_mode: Optional[str] = "bidirectional", return_sentiment: Optional[bool] = True,
                           weight_cutoff: Optional[float] = None) -> (pd.DataFrame,dict):
    """
    This function returns a dataframe with a list of contextual tokens that appear in the context of another dyad.
    The focal dyad can be specified by occurrence tokens, and substitute tokens, given as lists. The algorithm
    will consider the combination of each element dyad=(occurrence, substitute).

    For each such dyad, contextual tokens are returned, also from a dyad that occurs in the same sequence.
    Which token gets returned, and with which weight, depends on the parameter context mode
    If context_mode="occuring", give the likelihood that the token appears as written in the context of a substitution
    If context_mode="bidirectional", give the likelihood that the token appears, or according to BERT could appear
    If context_mode="substitution", give the likelihood that the token could appear when it does not

    Values are aggregated across sequences with the substitution weight of the original dyad.


    Parameters
    ----------
    snw : neo4j_network
        Semantic Network

    focal_substitutes: list, str, int, Optional
        Terms that substitute for an occurring term in the focal dyad

    focal_occurrences:  list, str, int, Optional
        Terms that occur in the focal dyad

    groups: iterable over lists of tokens, Optional
        Provide some collection of lists of tokens, for example, clusters.
        Output dataframe will be grouped by the tokens.

    context_pos: str, Optional
        Only consider context terms, where the occurring word is classified as the given Part of Speech

    times: list, Optional
        Aggregate across these times

    context_mode: str, Optional, Default "bidirectional"
        If context_mode="occuring", give the likelihood that the token appears as written in the context of a substitution
        If context_mode="bidirectional", give the likelihood that the token appears, or according to BERT could appear
        If context_mode="substitution", give the likelihood that the token could appear when it does not

    return_sentiment: bool, Optional, Default True
        Return sentiment and subjectivity (Averaged) for the focal tie

    weight_cutoff: float, Optional, Default None
        Ignore any network ties that are less than this value in weight

    Returns
    -------
        Pandas DataFrame with all tokens and group associations

    """

    # Format inputs
    if focal_substitutes is not None:
        if not isinstance(focal_substitutes, (list, np.ndarray)):
            role_cluster = [focal_substitutes]
    if focal_occurrences is not None:
        if not isinstance(focal_occurrences, (list, np.ndarray)):
            role_cluster = [focal_occurrences]
    if times is None:
        times = snw.get_times_list()
    if not isinstance(times, (list, np.ndarray)):
        times = [times]

    df=pd.DataFrame(snw.get_dyad_context(focal_occurrences=focal_occurrences, focal_substitutes=focal_substitutes, times=times, weight_cutoff=weight_cutoff,
                                 context_pos=context_pos, context_mode=context_mode, return_sentiment=return_sentiment)[
                'dyad_context'])

    df["context_token"] = snw.ensure_tokens(df.idx)
    df_list=[]
    df_dict={}
    for group in groups:
        group_idx = snw.ensure_ids(group)
        group_tk = snw.ensure_tokens(group_idx)
        group_df = df[df.idx.isin(group_idx)]
        if len(group_df)>0:
            group_df = group_df.groupby(["idx","context_token", "pos"]).agg(weight=('weight', 'sum'), sentiment=('sentiment', 'mean'),
                                                subjectivity=('subjectivity', 'mean'))


            group_df = group_df.reset_index(drop=False).sort_values(by="weight", ascending=False)
            if return_sentiment:
                group_df = group_df[["idx", "context_token", "pos", "weight", "sentiment", "subjectivity"]]
            else:
                group_df = group_df[["idx", "context_token", "pos", "weight"]]
        else:
            group_df = None
        group_name=group_df.iloc[0:6].context_token
        group_name = "-".join(list(group_name.nlargest(5).index))
        group_df.group_name = group_name
        df_list.append(group_df)
        df_dict[group_name]=group_df

    df = pd.concat(df_list)

    return df, df_dict
