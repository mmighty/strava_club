#!C:\Python36\python.exe -u
import argparse
import json
import os
import pprint
import socket
import time
import urllib.parse
import webbrowser

import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from stravalib.client import Client

import keychain


class StravaAccess:
    '''
    Retrieves data from Strava API, using slightly interactive OAuth2 protocol, requiring user to log in, click approve, paste returned URL
    '''
    CACHE_FILE = os.path.join(os.path.dirname(__file__), r'strava_token.json')

    def __init__(self):
        self.client = Client()
        self.token_cache = {}
        self.load_token_cache()

        if time.time() > self.token_cache['expire']:
            print('Access token {old_token} expired, refreshing...'.format(old_token=self.token_cache['atoken']))

            refresh_response = self.client.refresh_access_token(client_id=keychain.app['client_id'],
                                                                client_secret=keychain.app['client_secret'],
                                                                refresh_token=self.token_cache['rtoken'])

            self.token_cache['atoken'] = refresh_response['access_token']
            self.token_cache['rtoken'] = refresh_response['refresh_token']
            self.token_cache['expire'] = refresh_response['expires_at']

            self.save_token_cache()

        self.client.access_token = self.token_cache['atoken']
        self.client.refresh_token = self.token_cache['rtoken']
        self.client.token_expires_at = self.token_cache['expire']

    def authorize_app(self):
        '''
        Opens new tab in default browser to Strava OAuth2 site for our app
        User is asked to authorize in browser
        User is asked to paste redirect URL (not a real URL but I don't want to set up a web server to capture this)
        Parses the redirect URL for code in query and uses it to generate token for user
        '''
        # Generate code from url copy from url on redirect
        redirect_domain = 'https://localhost/urlforauth/'
        authorize_url = self.client.authorization_url(client_id=keychain.app['client_id'], redirect_uri=redirect_domain)
        webbrowser.open_new_tab(authorize_url)

        print('Please Authorize application access in browser.')
        redirect_url_with_code = input('URL redirect after authorization (e.g. %s...) :' % redirect_domain)
        parsed_url = urllib.parse.urlparse(redirect_url_with_code)

        try:
            code = urllib.parse.parse_qs(parsed_url.query)['code'][0]
        except KeyError:
            raise KeyError('Invalid URL, expected copied URL from browser after clicking authorize containing auth code.')

        token_response = self.client.exchange_code_for_token(client_id=keychain.app['client_id'],
                                                             client_secret=keychain.app['client_secret'],
                                                             code=code)

        self.token_cache = {'atoken' : token_response['access_token'], 'rtoken' : token_response['refresh_token'], 'expire' : token_response['expires_at']}
        self.save_token_cache()

    def load_token_cache(self):
        '''
        load or initialize self.token_cache dictionary
        keys: atoken, rtoken, expire
        '''
        print('Loading token cache ({cache_file})...'.format(cache_file=self.CACHE_FILE))
        try:
            with open(self.CACHE_FILE, 'r') as json_file:
                self.token_cache = json.load(json_file)

        except FileNotFoundError:
            print('Could not load token cache, proceeding with full authorization...')
            self.authorize_app()

    def save_token_cache(self):
        print('Saving token cache ({cache_file})...'.format(cache_file=self.CACHE_FILE))
        with open(self.CACHE_FILE, 'w') as json_file:
                json.dump(self.token_cache, json_file)


def get_data(output='strava_results.csv', club_id=121898):
    strava = StravaAccess()

    athlete = strava.client.get_athlete()
    print("Authenticated as {cn} {sn} ({id}) with token {token}".format(cn=athlete.firstname, sn=athlete.lastname, id=athlete.id, token=strava.token_cache['atoken']))

    # client.get_club_activities(club_id, limit=10)
    clubs = strava.client.get_athlete_clubs()
    pprint.pprint(clubs)

    assert any(club_id == club.id for club in clubs), "Authenticated user must be a member of the club {id} to pull activities".format(id=club_id)

    my_cols = [
        'name',
        'type',
        'distance',
        'moving_time',
        'elapsed_time',
        'total_elevation_gain'
    ]
    data = []

    for activity in strava.client.get_club_activities(club_id):
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
    df.to_csv(output, index=False)

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
    parser = argparse.ArgumentParser(description='Club activity data for Strava club of interest.')
    parser.add_argument('-c', '--use_cache', action='store_true', help='Use cached data from Strava API, otherwise will query API for activity data')
    parser.add_argument('-o', '--out_file', action='store', default=os.path.join(os.path.dirname(__file__), 'output', 'strava_results.csv'), help='Output location of activity data, used both as seed for activity data if not refreshing and as output location')
    parser.add_argument('-id', '--club_id', action='store', type=int, default=121898, help='Club ID of interest. Default is "Seagate Longmont"')

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)

    if args.use_cache:
        try:
            df =  pd.read_csv(args.out_file)
        except FileNotFoundError:
            print('Could not load previous file, refreshing activity data...')
            df = get_data(output=args.out_file, club_id=args.club_id)
    else:
        df = get_data(output=args.out_file, club_id=args.club_id)

    plot_data(df)

    # debug=True enables hot reload in new versions
    app.run_server(debug=False, threaded=True)

    # production
    # host = socket.gethostbyname(socket.gethostname())
    # print(host)
    # app.run_server(debug=False, host=host, port=8050)



if __name__ == '__main__':
    # global dash app
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

    main()
