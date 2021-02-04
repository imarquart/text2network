# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 17:23:33 2020

@author: marquart
"""
from typing import Union, List, Dict

import pandas as pd


def pd_format(format_list: Union[List, Dict])->List:
    """
    Formats measure outputs into pandas data frames

    Parameters
    ----------
    format_list : list
        List of dicts as supplied by measures.

    Returns
    -------
    result_list : list
        List of pandas data frames.

    """
    
    result_list=[]
    if not isinstance(format_list, list):
        format_list=[format_list]
        
        
    for fdict in format_list:
        if isinstance(fdict, dict):
            ftypes = list(fdict.keys())
            for ftype in ftypes:
                if ftype == "proximity":
                    proxdict=fdict[ftype]
                    output=pd.DataFrame(proxdict)
                    output=output.fillna(0)
                    output=output.sort_values(output.columns[0], ascending = False)
                    result_list.append(output)
                elif ftype =="centrality":
                    proxdict=fdict[ftype]
                    output=pd.DataFrame(proxdict)
                    output=output.fillna(0)
                    output=output.sort_values(output.columns[0], ascending = False)
                    result_list.append(output)
                elif ftype =="yearly_centrality":
                    proxdict=fdict[ftype]
                    year_list=list(proxdict.keys())
                    output_dict={}
                    for year in year_list:
                        asdf=pd.MultiIndex.from_product([[year],list(proxdict[year]['centralities'].keys())])
                        output=pd.DataFrame(proxdict[year]['centralities'])
                        output=output.fillna(0)
                        output=output.sort_values(output.columns[0], ascending = False)
                        output_dict.update({year: output})
                    output=pd.concat(output_dict)

                    result_list.append(output)
        else:
            pass


    return result_list
            
    
    
    