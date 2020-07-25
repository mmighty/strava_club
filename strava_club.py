import json
import numpy as np
import pandas as pd
import time
import plotly
import plotly.express as px
import plotly.graph_objects as go
from stravalib.client import Client
from plotly.subplots import make_subplots
import dash
import dash_core_components as dcc
import dash_html_components as html
import socket
import strava_config #this is local file with config values


def get_data():
    client = Client()
    first_run = None

    if first_run:
        code = strava_config.CODE
        token_response = client.exchange_code_for_token(client_id=strava_config.CLIENT_ID,
                                                        client_secret=strava_config.CLIENT_SECRET,
                                                        code=code)
        access_token = token_response['access_token']
        refresh_token = token_response['refresh_token']
        expires_at = token_response['expires_at']

        # Now store that short-lived access token somewhere (textfile)
        client.access_token = access_token
        # You must also store the refresh token to be used later on to obtain another valid access token
        # in case the current is already expired
        client.refresh_token = refresh_token

        sv_data = {'atoken': access_token, 'rtoken': refresh_token, 'expire': expires_at}
        with open('strava.txt', 'w') as outfile:
            json.dump(sv_data, outfile)

    with open('strava.txt') as json_file:
        data = json.load(json_file)

    # Assign client values
    client.access_token = data['atoken']
    client.refresh_token = data['rtoken']

    # An access_token is only valid for 6 hours, store expires_at somewhere and
    # check it before making an API call.
    client.token_expires_at = data['expire']

    # ... time passes ...
    if time.time() > client.token_expires_at:
        refresh_response = client.refresh_access_token(client_id=strava_config.CLIENT_ID,
                                                       client_secret=strava_config.CLIENT_SECRET,
                                                       refresh_token=client.refresh_token)
        access_token = refresh_response['access_token']
        refresh_token = refresh_response['refresh_token']
        expires_at = refresh_response['expires_at']

        sv_data = {'atoken': access_token, 'rtoken': refresh_token, 'expire': expires_at}
        with open('strava.txt', 'w') as outfile:
            json.dump(sv_data, outfile)

    athlete = client.get_athlete()
    print("For {id}, I now have an access token {token}".format(id=athlete.id, token=data['atoken']))

    # client.get_club_activities(club_id, limit=10)
    clubs = client.get_athlete_clubs()
    print(clubs)

    # club ID of interest
    sgate_l = 121898

    club = client.get_club(sgate_l)
    print(club)

    my_cols = [
        'name',
        'type',
        'distance',
        'moving_time',
        'elapsed_time',
        'total_elevation_gain'
    ]

    data = []
    # for activity in client.get_club_activities(sgate_l, limit=250):
    for activity in club.activities:
        my_dict = activity.to_dict()
        # print(my_dict)
        # print(activity.athlete.firstname + ' ' + activity.athlete.lastname)
        data.append([activity.athlete.firstname + ' ' + activity.athlete.lastname] + [my_dict.get(x) for x in my_cols])

    my_cols.insert(0, 'athlete')
    df = pd.DataFrame(data, columns=my_cols)

    #  Create a distance in km column
    df['distance_km'] = df['distance'] / 1e3
    #  Create a distance in mi column
    df['distance_mi'] = df['distance_km'] * 0.621371
    # Convert times to timedeltas
    df['moving_time'] = pd.to_timedelta(df['moving_time'])
    df['elapsed_time'] = pd.to_timedelta(df['elapsed_time'])
    df['total_elevation_gain_ft'] = df['total_elevation_gain'] * 3.28084

    # Save data
    ## unkown limit need to concatenate
    df.to_csv('strava_results.csv', index=False)

    # print(df)
    return df


def plot_data(data):
    print("entering plot")
    print(data)

    data['moving_time'] = pd.to_timedelta(data['moving_time'])
    data['elapsed_time'] = pd.to_timedelta(data['elapsed_time'])

    # numeric time in hours
    data['moving_time2'] = data['moving_time'].values.astype(np.int64) / 3.6e+12
    data['elapsed_time2'] = data['elapsed_time'].values.astype(np.int64) / 3.6e+12

    df1 = data.groupby(['type']).agg({
        'type': 'count',
        'distance_mi': ['mean', 'sum'],
        'total_elevation_gain_ft': ['mean', 'sum'],
        'moving_time2': 'mean',
        'elapsed_time2': 'mean',

    })

    df1.columns = list(map('_'.join, df1.columns))
    df1.index.name = 'type'
    print(df1)
    print(df1.columns)

    fig_pie = px.pie(df1, values='type_count', names=df1.index, title='Activity Count %')
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    # fig_pie.write_html("strava1.html")

    fig_hist = px.histogram(data, x="elapsed_time2", color='type')
    # fig.write_html("strava1.html")

    fig_table = go.Figure(data=[go.Table(
        header=dict(values=[df1.index.name] + list(df1.columns),
                    align='left'),
        cells=dict(values=[df1.index, df1.type_count, df1.distance_mi_mean, df1.distance_mi_sum,
                           df1.total_elevation_gain_ft_mean,
                           df1.total_elevation_gain_ft_sum, df1.moving_time2_mean, df1.elapsed_time2_mean],
                   align='left'))
    ])

    # fig_table.write_html("strava1.html")

    graph1 = dcc.Graph(
        id='graph1',
        figure=fig_pie,
        className="six columns"
    )

    graph2 = dcc.Graph(
        id='graph2',
        figure=fig_hist,
        className="six columns"
    )

    graph3 = dcc.Graph(
        id='graph3',
        figure=fig_table,
        className="twelve columns"
    )

    header = html.H2(children="Seagate Longmont Strava Data")

    row1 = html.Div(children=[graph1, graph2], )

    row2 = html.Div(children=[graph3])

    layout = html.Div(children=[header, row1, row2], style={"text-align": "center"})
    app.layout = layout
    # app.run_server(debug=True)


def main():
    # get_data()
    df = pd.read_csv('strava_results.csv')
    plot_data(df)


# global dash app
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

if __name__ == '__main__':
    main()

    # debug=True enables hot reload in new versions
    app.run_server(debug=False, threaded=True)

    # production
    # host = socket.gethostbyname(socket.gethostname())
    # print(host)
    # app.run_server(debug=False, host=host, port=8050)
