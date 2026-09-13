import pickle
import os

import dash
from dash import html, dcc, Input, Output, State
import numpy as np
import pandas as pd

# needed so pickle can find the LinearRegression class when loading best_model.pkl
from model_class import LinearRegression, Normal, Lasso, Ridge, Elastic, \
    NormalRegression, LassoRegression, RidgeRegression, ElasticRegression  # noqa: F401

import sys
import model_class

# fix for pickle: the models were pickled while the classes lived in the
# notebook's __main__ module, so we register them under __main__ here too,
# otherwise pickle.load() can't find them when running under gunicorn
sys.modules['__main__'].LinearRegression = model_class.LinearRegression
sys.modules['__main__'].NormalRegression = model_class.NormalRegression
sys.modules['__main__'].LassoRegression = model_class.LassoRegression
sys.modules['__main__'].RidgeRegression = model_class.RidgeRegression
sys.modules['__main__'].ElasticRegression = model_class.ElasticRegression
sys.modules['__main__'].Normal = model_class.Normal
sys.modules['__main__'].Lasso = model_class.Lasso
sys.modules['__main__'].Ridge = model_class.Ridge
sys.modules['__main__'].Elastic = model_class.Elastic

# ------------------------------------------------------------------
# Load models
# ------------------------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')

old_model = pickle.load(open(os.path.join(MODEL_DIR, 'car_price_model.pkl'), 'rb'))
model_columns = pickle.load(open(os.path.join(MODEL_DIR, 'model_columns.pkl'), 'rb'))
defaults = pickle.load(open(os.path.join(MODEL_DIR, 'defaults.pkl'), 'rb'))

new_model = pickle.load(open(os.path.join(MODEL_DIR, 'best_model.pkl'), 'rb'))
scaler = pickle.load(open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'rb'))
poly = pickle.load(open(os.path.join(MODEL_DIR, 'poly.pkl'), 'rb'))
feature_names = pickle.load(open(os.path.join(MODEL_DIR, 'feature_names.pkl'), 'rb'))
best_config = pickle.load(open(os.path.join(MODEL_DIR, 'best_config.pkl'), 'rb'))

# ------------------------------------------------------------------
# Choices shown in the form dropdowns
# ------------------------------------------------------------------
BRANDS = ['Ambassador', 'Ashok', 'Audi', 'BMW', 'Chevrolet', 'Daewoo', 'Datsun',
          'Fiat', 'Force', 'Ford', 'Honda', 'Hyundai', 'Isuzu', 'Jaguar', 'Jeep',
          'Kia', 'Land', 'Lexus', 'MG', 'Mahindra', 'Maruti', 'Mercedes-Benz',
          'Mitsubishi', 'Nissan', 'Opel', 'Peugeot', 'Renault', 'Skoda', 'Tata',
          'Toyota', 'Volkswagen', 'Volvo']
FUEL_TYPES = ['Diesel', 'Petrol']
TRANSMISSIONS = ['Manual', 'Automatic']
SELLER_TYPES = ['Individual', 'Dealer', 'Trustmark Dealer']
OWNERS = [1, 2, 3, 4]

TRANSMISSION_MAP = {'Manual': 1, 'Automatic': 0}
SELLER_MAP = {'Dealer': 0, 'Individual': 1, 'Trustmark Dealer': 2}


# ------------------------------------------------------------------
# Shared: turn form values into the exact feature vector the models expect
# ------------------------------------------------------------------
def encode_row(row, columns):
    df = pd.DataFrame([{
        'year': row['year'],
        'km_driven': row['km_driven'],
        'seller_type': SELLER_MAP.get(row['seller_type'], 1),
        'transmission': TRANSMISSION_MAP.get(row['transmission'], 1),
        'owner': row['owner'],
        'mileage': row['mileage'],
        'engine': row['engine'],
        'max_power': row['max_power'],
        'seats': row['seats'],
    }])

    for fuel in FUEL_TYPES[1:]:  # 'Diesel' is the dropped baseline category
        df[f'fuel_{fuel}'] = 1 if row['fuel'] == fuel else 0

    for brand in BRANDS[1:]:  # 'Ambassador' is the dropped baseline category
        df[f'brand_{brand}'] = 1 if row['brand'] == brand else 0

    # align exactly to the columns/order the model was trained on
    df = df.reindex(columns=columns, fill_value=0)
    return df


def predict_old(row):
    X = encode_row(row, model_columns)
    pred_log = old_model.predict(X)[0]
    return float(np.exp(pred_log))


def predict_new(row):
    X = encode_row(row, feature_names)
    X_scaled = scaler.transform(X.values.astype(float))
    if best_config['model_type'] == 'poly':
        X_scaled = poly.transform(X_scaled)
    X_final = np.hstack([np.ones((X_scaled.shape[0], 1)), X_scaled])
    pred_log = new_model.predict(X_final)[0]
    return float(np.exp(pred_log))


# ------------------------------------------------------------------
# Dash app
# ------------------------------------------------------------------
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = 'Car Price Prediction'
server = app.server  # needed for Docker/gunicorn deployment


def dropdown(id_, options, value):
    return dcc.Dropdown(id=id_, options=[{'label': o, 'value': o} for o in options],
                         value=value, clearable=False)


def number_input(id_, value, step=1):
    return dcc.Input(id=id_, type='number', value=value, step=step,
                      style={'width': '100%'})


def build_form(prefix):
    """Builds the same set of input fields for either page (old/new),
    with ids prefixed so both pages' callbacks stay independent."""
    return html.Div([
        html.Div([
            html.Label('Brand'),
            dropdown(f'{prefix}-brand', BRANDS, 'Maruti'),
        ], className='field'),

        html.Div([
            html.Label('Year'),
            number_input(f'{prefix}-year', defaults['year']),
        ], className='field'),

        html.Div([
            html.Label('Km driven'),
            number_input(f'{prefix}-km_driven', defaults['km_driven']),
        ], className='field'),

        html.Div([
            html.Label('Fuel'),
            dropdown(f'{prefix}-fuel', FUEL_TYPES, 'Petrol'),
        ], className='field'),

        html.Div([
            html.Label('Seller type'),
            dropdown(f'{prefix}-seller_type', SELLER_TYPES, 'Individual'),
        ], className='field'),

        html.Div([
            html.Label('Transmission'),
            dropdown(f'{prefix}-transmission', TRANSMISSIONS, 'Manual'),
        ], className='field'),

        html.Div([
            html.Label('Owner'),
            dropdown(f'{prefix}-owner', OWNERS, 1),
        ], className='field'),

        html.Div([
            html.Label('Mileage (kmpl)'),
            number_input(f'{prefix}-mileage', defaults['mileage'], step=0.1),
        ], className='field'),

        html.Div([
            html.Label('Engine (CC)'),
            number_input(f'{prefix}-engine', defaults['engine']),
        ], className='field'),

        html.Div([
            html.Label('Max power (bhp)'),
            number_input(f'{prefix}-max_power', defaults['max_power'], step=0.1),
        ], className='field'),

        html.Div([
            html.Label('Seats'),
            number_input(f'{prefix}-seats', defaults['seats']),
        ], className='field'),

        html.Button('Predict price', id=f'{prefix}-submit', n_clicks=0,
                    className='submit-btn'),

        html.Div(id=f'{prefix}-result', className='result-box'),
    ], className='form-grid')


navbar = html.Div([
    html.Div('Car Price Prediction', className='navbar__brand'),
    html.Div([
        dcc.Link('Old model (A1)', href='/', className='nav-link'),
        dcc.Link('New model (A2)', href='/new', className='nav-link'),
    ], className='navbar__links'),
], className='navbar')

old_page = html.Div([
    html.H1("Predict a used car's price"),
    html.P("This page uses the A1 model: a Random Forest trained with "
           "scikit-learn and tuned with grid search."),
    build_form('old'),
])

new_page = html.Div([
    html.H1("Predict a used car's price - new model"),
    html.P([
        "This page uses the A2 model: a linear regression implemented from "
        "scratch with gradient descent, chosen out of 144 configurations "
        "compared with MLflow (best config: ",
        html.Code(f"{best_config['model_type']} regression, "
                  f"{best_config['method']} gradient descent, "
                  f"{best_config['init_method']} initialization, "
                  f"learning rate {best_config['lr']}"),
        ")."
    ]),
    build_form('new'),
])

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    navbar,
    html.Div(id='page-content', className='page'),
])


@app.callback(Output('page-content', 'children'), Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/new':
        return new_page
    return old_page


def make_predict_callback(prefix, predict_fn):
    @app.callback(
        Output(f'{prefix}-result', 'children'),
        Input(f'{prefix}-submit', 'n_clicks'),
        State(f'{prefix}-brand', 'value'),
        State(f'{prefix}-year', 'value'),
        State(f'{prefix}-km_driven', 'value'),
        State(f'{prefix}-fuel', 'value'),
        State(f'{prefix}-seller_type', 'value'),
        State(f'{prefix}-transmission', 'value'),
        State(f'{prefix}-owner', 'value'),
        State(f'{prefix}-mileage', 'value'),
        State(f'{prefix}-engine', 'value'),
        State(f'{prefix}-max_power', 'value'),
        State(f'{prefix}-seats', 'value'),
        prevent_initial_call=True,
    )
    def _predict(n_clicks, brand, year, km_driven, fuel, seller_type,
                 transmission, owner, mileage, engine, max_power, seats):
        row = dict(brand=brand, year=year, km_driven=km_driven, fuel=fuel,
                   seller_type=seller_type, transmission=transmission,
                   owner=owner, mileage=mileage, engine=engine,
                   max_power=max_power, seats=seats)
        price = predict_fn(row)
        return f'Estimated selling price: ₹{price:,.0f}'
    return _predict


make_predict_callback('old', predict_old)
make_predict_callback('new', predict_new)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)