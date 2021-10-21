import tables


def get_uniques(split_hierarchy, db_folder):
    """
    Queries database to get unique values according to hierarchy provided.
    Determines how many models we would like to train.
    :param split_hierarchy: List of table parameters
    :return: dict including unique values, query strings and bert folder names

    Parameters
    ----------
    db_folder
    """
    # Create hierarchy splits
    hdf = tables.open_file(db_folder, mode="r")
    data = hdf.root.textdata.table
    # Create dict sets
    uniques = {}
    for param in split_hierarchy:
        uniques[param] = []
    uniques["query"] = []
    uniques["file"] = []
    uniques["query_filename"] = []

    # Iterate to get unqiue values and create query strings and file-names
    for row in data.iterrows():
        query = []
        filename = []

        for param in split_hierarchy:
            val = row[param]
            uniques[param].append(val)
            uniques[param] = list(set(uniques[param]))
            # Create query string
            query.append("(%s == %s)" % (param, val))
            if type(val) is bytes:
                val = val.decode("utf-8")
            filename.append("%s" % (val))

        # Add (uniquely) query strings and file-names
        query = " & ".join(query)
        filename = "-".join(filename)
        uniques["query_filename"].append((query, filename))
        uniques["query_filename"] = list(set(uniques["query_filename"]))

    # Split up tuple to get single instances
    uniques["query"] = [x[0] for x in uniques['query_filename']]
    uniques["file"] = [x[1] for x in uniques['query_filename']]

    return uniques
