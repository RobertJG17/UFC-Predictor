import pandas as pd


def get_results(model, cols, scaler):
    upcoming = pd.read_csv('upcoming-event.csv')
    scaled_features = scaler.transform(upcoming.loc[:, cols].drop("Winner", axis=1))

    predictions = model.predict(scaled_features)

    formatted = pd.DataFrame({
        "Red Fighter": upcoming.loc[:, "R_fighter"],
        "Blue Fighter": upcoming.loc[:, "B_fighter"],
        "Predictions": [f"{color} Fighter" for color in predictions]
    })

    formatted.loc[:, "Predictions"] = formatted.index.map(lambda idx: formatted.loc[idx, formatted.Predictions.iloc[idx]])

    return [{
        f"{formatted.loc[idx, 'Red Fighter']} vs. "
        f"{formatted.loc[idx, 'Blue Fighter']}": formatted.loc[idx, 'Predictions']
    } for idx in formatted.index]
