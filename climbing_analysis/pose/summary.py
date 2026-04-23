import dask
import dask.dataframe as dd
import matplotlib.pyplot as plt
import numpy as np

def plot_summary(ddf: dask.dataframe):
    """Currently testing -- code for summary stats on pose data

    Args:
        ddf (dask.dataframe): _description_

    Returns:
        _type_: _description_
    """

    GROUP_COLS = ['Trial', 'Date', 'Id']
    
    # ---
    # Max average speed
    # ---
    result = (ddf.groupby(GROUP_COLS)
              .apply(get_max_speed, 
                     meta={"Id": "object", 
                           "Date": "datetime64[ns]", 
                           "Trial":"object", 
                           "max_dy": "float64"})
                           .reset_index(drop=True)
    )

    result_average_speed = (result.groupby(['Id', 'Date'])
                            .mean())
    



    # Compute results
    res_speed = dask.compute(result_average_speed) # annoyingly have to run compute before calculating mean as data rows dissapear


    # Plotting
    

    
    

def get_max_speed(pdf, node: str = 'tail_Y'):

    pdf = pdf.sort_values("frame_id")
    dy = pdf[node].diff().abs()
    dys = pd.DataFrame({"Id": [pdf["Id"].iloc[0]],
                        "Date": [pdf["Date"].iloc[0]],
                        "Trial": [pdf["Trial"].iloc[0]],
                        "max_dy": [dy.max()]})

    return dys
