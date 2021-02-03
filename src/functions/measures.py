def yearly_centralities(nw, year_list, focal_tokens=None, types=["PageRank", "normedPageRank"], ego_nw_tokens=None,
                        depth=1, context=None, weight_cutoff=None, norm_ties=None):
    """
    Compute directly year-by-year centralities for provided list.

    Parameters
    ----------
    year_list : list
        List of years for which to calculate centrality.
    focal_tokens : list, str
        List of tokens of interest. If not provided, centralities for all tokens will be returned.
    types : list, optional
        types of centrality to calculate. The default is ["PageRank"].
    ego_nw_tokens : list, optional - used when conditioning
         List of tokens for an ego-network if desired. Only used if no graph is supplied. The default is None.
    depth : TYPE, optional - used when conditioning
        Maximal path length for ego network. Only used if no graph is supplied. The default is 1.
    context : list, optional - used when conditioning
        List of tokens that need to appear in the context distribution of a tie. The default is None.
    weight_cutoff : float, optional - used when conditioning
        Only links of higher weight are considered in conditioning.. The default is None.
    norm_ties : bool, optional - used when conditioning
        Please see semantic network class. The default is True.

    Returns
    -------
    dict
        Dict of years with dict of centralities for focal tokens.

    """

    # Get default normation behavior
    if norm_ties == None:
        norm_ties = nw.norm_ties

    cent_year = {}
    assert isinstance(year_list, list), "Please provide list of years."
    for year in year_list:
        cent_measures = nw.centralities(focal_tokens=focal_tokens, types=types, years=[
            year], ego_nw_tokens=ego_nw_tokens, depth=depth, context=context, weight_cutoff=weight_cutoff,
                                          norm_ties=norm_ties)
        cent_year.update({year: cent_measures})

    return {'yearly_centrality': cent_year}
