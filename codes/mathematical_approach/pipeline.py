import pandas as pd

from . import fetch_topix
import data_loader, config, data_processor, features, utils, evaluation

COLUMNS = ["Date", "SecuritiesCode", "Target"]
FOR_PARAMETERS_LOOKBACK = 70
FOR_PREDICTING_LOOKBACK = 20

def pre_processor() -> pd.DataFrame:
    """
    Only need to get Target in this OLS case, 
    can use get_features() from features.py
    """
    df = data_loader.main_loader(config.DATA_PATH)
    
    # Only requires Date, SecuritiesCode, Target 
    df = df[COLUMNS]
    df["Date"] = pd.to_datetime(df["Date"])
    """cleaned = features.get_valid_rows(
        df,
        ["Date", "SecuritiesCode", "Target"]
    )"""
    
    df = df[["Date", "SecuritiesCode", "Target"]]
    return df.set_index("Date").sort_index()
        

def make_window_score(
    target: pd.DataFrame,
    topix: pd.DataFrame,
    current_time: pd.Timestamp,
    for_parameters_lookback: int = FOR_PARAMETERS_LOOKBACK,
    for_predicting_lookback: int = FOR_PREDICTING_LOOKBACK
) -> pd.DataFrame:
    """
    for_parameters_lookback: B
    for_predicting_lookback: G
    current Time: T
    
    Topix: 
        - index: TimeData
        - Value: ToPix Return
        
    OLS: 
        - finding parameters: (T - G - B) ~ (T - G - 1)
        - Calculating predictions: (T - G) ~ (T - 1)
        
    Only requires past target and topix can derive future predictive target
        
    Formula:
        predicted = alpha + beta * topix_G
        predicted_score = predicted.mean()
        residual_score = (G_stock - predicted).mean()

        Each stock:
            alpha, beta: One number
            topix_G, G_stock, predicted: Each G amounts of values
            predicted_score, residual_score: Each one number
        
    Return:
        - prediction: 1 day Target
    """
    result = pd.DataFrame(columns=["SecuritiesCode", "PredictedScore", "ResidualScore", "Target"])
    
    T, G, B = pd.Timestamp(current_time), for_predicting_lookback, for_parameters_lookback
    future_target = target.loc[target.index == T].copy()
    
    # Build topix_G, topix_B
    topix = topix.loc[topix.index < T].copy().sort_index().tail(G + B)
    target = target[target.index.isin(topix.index)]
    
    topix_B = topix.head(B)
    topix_G = topix.tail(G)
    B_topix_mean = topix_B.mean()
    dx_B = topix_B - B_topix_mean
    
    # Build target_G, target_B
    target_B = target.loc[target.index.isin(topix_B.index)]
    target_G = target.loc[target.index.isin(topix_G.index)]
        
    # Use for loop of code and calculate each result
    for code in target["SecuritiesCode"].unique():
        # Build target for each stock
        B_stock = target_B.loc[
            target_B["SecuritiesCode"] == code,
            "Target"
        ]
        G_stock = target_G.loc[
            target_G["SecuritiesCode"] == code,
            "Target"
        ]
        
        B_stock_mean = B_stock.mean()
        dy_B = B_stock - B_stock_mean
        
        beta = (
            (dx_B * dy_B).sum() / 
            (dx_B ** 2).sum()
        )
        apha = B_stock_mean - beta * B_topix_mean
        
        predicted = apha + beta * topix_G
        residual = G_stock - predicted
        
        # Set index = "SecuritiesCode" (code)
        result.loc[code, "Date"] = T
        result.loc[code, "SecuritiesCode"] = code
        result.loc[code, "Prediction"] = residual.mean()
        result.loc[code, "Target"] = future_target.loc[
            future_target["SecuritiesCode"] == code,
            "Target"
        ].item() # Take the float value out of the Series (index: value)
        
    return result


def get_results(
    predictions: pd.DataFrame,
    topix: pd.DataFrame,
    curr_date: pd.Timestamp,
    n_long: int = 200,
    n_short: int = 200,
) -> pd.DataFrame:
    """
    Use evaluaitons to print the results
    """
    ranked = evaluation.rank(predictions, "Prediction")
    weighted = evaluation.calculate_weights(ranked, n_long, n_short)
    rank_IC = evaluation.calculate_rank_IC(weighted)
    daily_return = evaluation.daily_earning(weighted).sum()
    
    print("==========================================================")
    print(f"Rank IC: {rank_IC}")
    print(f"Gross Additive Sum: {daily_return}")
    print("==========================================================")

    return pd.DataFrame([{
        "Date": curr_date,
        "rank_ic": rank_IC,
        "daily_return": daily_return,
    }])
    

def everyday_calculation(
    target: pd.DataFrame,
    topix: pd.DataFrame,
    for_parameters_lookback: int = FOR_PARAMETERS_LOOKBACK,
    for_predicting_lookback: int = FOR_PREDICTING_LOOKBACK,
    n_long: int = 200,
    n_short: int = 200
) -> pd.DataFrame:
    final_result = pd.DataFrame()
    
    G, B = for_predicting_lookback, for_parameters_lookback
    topix = topix.sort_index()
    score_dates = topix.index[G + B:].intersection(target.index.unique())
    
    for curr_date in score_dates:
        prediction = make_window_score(
            target,
            topix,
            curr_date,
            for_parameters_lookback,
            for_predicting_lookback
        )
        result = get_results(prediction, topix,curr_date, n_long, n_short)
        final_result = pd.concat([final_result, result])
    
    return final_result
        
@utils.print_running_time
def run(
    n_long: int = 200,
    n_short: int = 200,
):
    """
    Run the whole pipeline
    Including:
        - Fetch data (JPX, TOPIX)
        - Do predictions
        - Evaluate current model
    """
    topix_data = fetch_topix.run()
    target = pre_processor()
    final_result = everyday_calculation(
        target=target,
        topix=topix_data,
        for_predicting_lookback=FOR_PREDICTING_LOOKBACK,
        for_parameters_lookback=FOR_PARAMETERS_LOOKBACK,
        n_long=n_long,
        n_short=n_short
    )
    
    daily_returns = final_result["daily_return"]
    sharpe = daily_returns.mean() / daily_returns.std(ddof=1)
    gross_additive_sum = daily_returns.sum()
    
    print("==========================================================")
    print(f"Rank IC: {final_result["rank_ic"].mean()}")
    print(f"Gross Additive Sum: {gross_additive_sum}")
    print(f"Sharpe: {sharpe}")
    print(f"TOPIX average eearnings: {topix_data[final_result["Date"]].mean()}")
    print("==========================================================")
    
    
if __name__ == "__main__":
    run()
    