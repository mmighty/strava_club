import argparse
import json
import os
import pprint
import random
import re
import socket
import time
import urllib.parse
import webbrowser

import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html_dash
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from stravalib.client import Client

import keychain

from selenium.webdriver.common.by import By
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from lxml import html
from lxml import etree

#heatmap
import folium
from folium import plugins

# CHROMEDRIVER_PATH = '/usr/local/bin/chromedriver'
WINDOW_SIZE = "1920,1080"
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--window-size=%s" % WINDOW_SIZE)
chrome_options.add_argument('--no-sandbox')
# driver = webdriver.Chrome(executable_path=CHROMEDRIVER_PATH,
#                          options=chrome_options
#                          )

#driver = webdriver.Chrome(options=chrome_options)
#driver.implicitly_wait(10)  # seconds


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
            raise KeyError(
                'Invalid URL, expected copied URL from browser after clicking authorize containing auth code.')

        token_response = self.client.exchange_code_for_token(client_id=keychain.app['client_id'],
                                                             client_secret=keychain.app['client_secret'],
                                                             code=code)

        self.token_cache = {'atoken': token_response['access_token'], 'rtoken': token_response['refresh_token'],
                            'expire': token_response['expires_at']}
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
    print("Authenticated as {cn} {sn} ({id}) with token {token}".format(cn=athlete.firstname, sn=athlete.lastname,
                                                                        id=athlete.id,
                                                                        token=strava.token_cache['atoken']))

    # client.get_club_activities(club_id, limit=10)
    clubs = strava.client.get_athlete_clubs()
    pprint.pprint(clubs)

    assert any(club_id == club.id for club in
               clubs), "Authenticated user must be a member of the club {id} to pull activities".format(id=club_id)

    club_details = strava.client.get_club(club_id)

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

    df['club_id'] = club_id
    df['club_name'] = club_details.name

    #  Create a distance in km column
    df['distance_km'] = df['distance'] / 1e3
    #  Create a distance in mi column
    df['distance_mi'] = df['distance_km'] * 0.621371
    # Convert times to timedeltas
    df['moving_time'] = pd.to_timedelta(df['moving_time'])
    df['elapsed_time'] = pd.to_timedelta(df['elapsed_time'])
    df['total_elevation_gain_ft'] = df['total_elevation_gain'] * 3.28084

    # Save data
    df.to_csv(output, index=False)

    # print(df)
    return df


def clean_data(data):
    data['Moving Time'].fillna(data['Elapsed Time'], inplace=True)

    # convert from object to delta time to merge on value
    data['Moving Time'] = data.apply(
        lambda row: "".join(["0", row['Moving Time']]) if len(row['Moving Time']) == 4 else row['Moving Time'],
        axis=1)
    data['Moving Time'] = data.apply(
        lambda row: row['Moving Time'] if row['Moving Time'].count(":") > 1 else "".join(["00:", row['Moving Time']]),
        axis=1)
    data['Moving Time'] = data.apply(
        lambda row: row['Moving Time'] if len(row['Moving Time']) == 8 else "".join(["0", row['Moving Time']]),
        axis=1)

    # convert from object to delta time to merge on value
    data['Elapsed Time'] = data.apply(
        lambda row: "".join(["0", row['Elapsed Time']]) if len(row['Elapsed Time']) == 4 else row['Elapsed Time'],
        axis=1)
    data['Elapsed Time'] = data.apply(
        lambda row: row['Elapsed Time'] if row['Elapsed Time'].count(":") > 1 else "".join(
            ["00:", row['Elapsed Time']]),
        axis=1)
    data['Elapsed Time'] = data.apply(
        lambda row: row['Elapsed Time'] if len(row['Elapsed Time']) == 8 else "".join(["0", row['Elapsed Time']]),
        axis=1)

    data['Moving Time'] = pd.to_timedelta(data['Moving Time'])
    data['Elapsed Time'] = pd.to_timedelta(data['Elapsed Time'])

    # numeric time in hours
    data['moving_time2'] = data['Moving Time'].values.astype(np.int64) / 3.6e+12
    data['elapsed_time2'] = data['Elapsed Time'].values.astype(np.int64) / 3.6e+12

    data['Distance'] = data['Distance'].str.replace(',', '')

    data[['Distance', 'Distance_Unit']] = data['Distance'].str.extract(r'([\d.]+)(\D+)')

    unit_convert = {'mi': 1, 'm': (1 / 1609)}
    data['distance_mi'] = pd.to_numeric(data['Distance']) * data['Distance_Unit'].map(unit_convert).fillna(0)

    # data['distance_mi'] = data['Distance'].str.replace('mi', '')
    data['distance_mi'] = data['distance_mi'].astype(float)

    data['Elevation'].fillna('0ft', inplace=True)

    data['total_elevation_gain_ft'] = data['Elevation'].str.replace('ft', '').replace(',', '')
    data['total_elevation_gain_ft'] = data['total_elevation_gain_ft'].str.replace(',', '')
    data['total_elevation_gain_ft'] = data['total_elevation_gain_ft'].astype(float)
    data['total_elevation_gain_ft'].fillna(0, inplace=True)

    # for time formatting
    pd.to_datetime(data['activity_local_time'])
    return data


def dist_calc(df):
    data_list = []
    for index, row in df[df['latlng_stream'].notnull()].iterrows():
        # print(row)
        if len(row['latlng_stream']) > 25:
            if json.loads(row['latlng_stream']):
                # print(row['latlng_stream'])
                data_list.append(
                    pd.DataFrame(json.loads(row['latlng_stream'])['latlng'], columns=['Latitude', 'Longitude']))

    df_heat = pd.concat(data_list)

    df_heat['lat'] = np.radians(df_heat['Latitude'])
    df_heat['lon'] = np.radians(df_heat['Longitude'])

    return df_heat



def time_format(x):
    ts = x.total_seconds()
    hours, remainder = divmod(ts, 3600)
    minutes, seconds = divmod(remainder, 60)
    return '{}:{:02d}:{:02d}'.format(int(hours), int(minutes), int(seconds))

def plot_data(data):
    print("entering plot")

    data = clean_data(data)

    data['activity_code'], factor_list = pd.factorize(data['activity_type'], sort=True)
    factor_list = list(factor_list)


    data['activity_local_time2'] = pd.to_datetime(data['activity_local_time']).dt.tz_convert('US/Mountain')
    #print(data['activity_local_time2'])

    data['activity_start_time'] = data['activity_local_time2'].dt.time
    #data['activity_start_day'] = data['activity_local_time2'].dt.day
    #data['activity_start_day']= data['activity_local_time2'].dt.strftime('%Y-%m-%d')
    #print(data['activity_start_day'])

    df_sum = data.groupby([data['activity_type'],data['activity_local_time2'].dt.date]).agg({
        'distance_mi': 'sum',
        'total_elevation_gain_ft': 'sum',
        'moving_time2': 'sum',
        'elapsed_time2': 'sum',

    }).groupby(level=0).cumsum().reset_index()

    df_sum_athlete = data.groupby([data['athlete_name'],data['activity_local_time2'].dt.date]).agg({
        'distance_mi': 'sum',
        'total_elevation_gain_ft': 'sum',
        'moving_time2': 'sum',
        'elapsed_time2': 'sum',

    }).groupby(level=0).cumsum().reset_index()

    #print(df_sum)
    #print(df_sum.columns)

    # Build colorscale
    fix_color_map = {}
    custom_color_scale = []
    activity_type_list = data['activity_type'].unique()

    count = 0

    for i in activity_type_list:
        if count > 12:
            count =0
        fix_color_map[i]=px.colors.cyclical.Phase[count]
        custom_color_scale.append([float(factor_list.index(i))/(len(factor_list)-1),px.colors.cyclical.Phase[count]])
        count = count + 1

    custom_color_scale = sorted(custom_color_scale, key=lambda x: x[0])

    #https://plotly.com/python/v3/selection-events/

    fig_sum = px.line(df_sum, x='activity_local_time2', y='distance_mi', labels={
                     "distance_mi": "Distance (mi)",
                     "activity_local_time2": "Date ",
                     "activity_type": "Activity Type"
                 }, color_discrete_map=fix_color_map, template="simple_white", color='activity_type')
    fig_sum.update_layout(
        title={
            'text': "Cumulative Club Statistics",
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'})

    fig_sump = px.line(df_sum_athlete, x='activity_local_time2', y='elapsed_time2', labels={
                     "elapsed_time2": "Elapsed Time (hr)",
                     "activity_local_time2": "Date ",
                     "athlete_name": "Athlete Name"
                 }, color_discrete_map=fix_color_map, template="simple_white", color='athlete_name')
    fig_sump.update_layout(
        title={
            'text': "Cumulative Member Statistics",
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'})

    #print(df_sum.cum_sum())

    #print(df_cum)
    #df_cum.columns = list(map('_'.join, df_cum.columns))
    #print(df_cum)



    #print(data)
    #print(data.columns)
    #print(data.dtypes)



    #summary data
    df1 = data.groupby(['activity_type']).agg({
        'activity_type': 'count',
        'distance_mi': ['mean', 'sum'],
        'total_elevation_gain_ft': ['mean', 'sum'],
        'moving_time2': 'mean',
        'elapsed_time2': 'mean',

    })

    df1.columns = list(map('_'.join, df1.columns))
    df1.index.name = 'type'

    #summary data // athlete
    df2 = data.groupby(['athlete_name']).agg({
        'activity_type': 'count',
        'distance_mi': ['mean', 'sum'],
        'total_elevation_gain_ft': ['mean', 'sum'],
        'moving_time2': ['mean', 'sum'],
        'elapsed_time2': ['mean', 'sum'],

    })

    df2.columns = list(map('_'.join, df2.columns))
    df2.index.name = 'name'



    fig_scat1 = px.scatter(data, x="distance_mi", y="moving_time2", color="activity_type", marginal_y="box",
                           marginal_x="box",labels={
                     "distance_mi": "Distance (mi)",
                     "moving_time2": "Moving Time (hr)",
                     "activity_type": "Activity Type"
                 }, hover_name="athlete_name", hover_data=["title", "Elevation"],
                           template="simple_white", color_discrete_map=fix_color_map, title="Distance Vs Moving Time")
    fig_scat1.update_layout(
        title={
            'text': "Distance Vs Moving Time",
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'})

    #https://plotly.com/python/parallel-categories-diagram/
    #https://plotly.com/python/figurewidget-app/


    #fig_wig = go.FigureWidget()
    # Build figure as FigureWidget

    #trace1 = go.Scatter(x=data['distance_mi'], y=data['moving_time2'],)

    #https://plotly.com/python/parallel-coordinates-plot/


    fig_par = go.Figure(data=go.Parcoords(
        line=dict(color=data['activity_code'],
                  colorscale=custom_color_scale),
            dimensions = list([
            dict(tickvals = list(range(0, len(factor_list)+1)),ticktext = factor_list,label = 'Activity ID', values = data['activity_code']),
            dict(label = 'Distance (mi)', values = data['distance_mi']),
            dict(label = 'Elevation Gain (ft)', values = data['total_elevation_gain_ft']),
            dict(label = 'Moving Time (hr)', values = data['moving_time2']),
            dict(label='Elapsed Time (hr)', values=data['elapsed_time2']),
        ])
        )

    )

    fig_par.update_layout(
        title={
            'text': "Activity Parallel Coordinates",
            'y': 1,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'})



    #fig_par = px.parallel_coordinates(data, color="activity_code",
    #                                  dimensions={"distance_mi", "total_elevation_gain_ft", "moving_time2",
    #                                              "elapsed_time2", "activity_code"}, template="simple_white")

    #fig_wig.add_bar(y=[1, 4, 3, 2]);

    #fig_pie = px.pie(df1, values='activity_type_count', names=df1.index, title='Activity Count %')
    #fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    # fig_pie.write_html("strava1.html")

    fig_hist = px.histogram(data, x="activity_type", labels={
                     "activity_type": "Activity Type",}, color='activity_type', color_discrete_map=fix_color_map, template="simple_white")
    fig_hist.update_layout(
        title={
            'text': "Count Of Activity Types",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'})

    data['activity_start_hour'] = data['activity_local_time2'].dt.hour
    fig_hist2 = px.histogram(data, x="activity_start_hour", labels={
        "activity_start_hour": "Time Of Day", }, color='activity_type', color_discrete_map=fix_color_map,template="simple_white")
    fig_hist2.update_layout(
        title={
            'text': "Activity Start Time",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'})
    #
    # fig.write_html("strava1.html")

    df1['moving_time2_mean'] = pd.to_timedelta(df1['moving_time2_mean'], unit='h').apply(time_format)
    df1['elapsed_time2_mean'] = pd.to_timedelta(df1['elapsed_time2_mean'], unit='h').apply(time_format)
    df1 = df1.round(2)

    df2['moving_time2_mean'] = pd.to_timedelta(df2['moving_time2_mean'], unit='h').apply(time_format)
    df2['elapsed_time2_mean'] = pd.to_timedelta(df2['elapsed_time2_mean'], unit='h').apply(time_format)
    df2 = df2.round(2)

    headerColor = 'grey'
    rowEvenColor = 'lightgrey'
    rowOddColor = 'white'

    fig_table = go.Figure(data=[go.Table(
        header=dict(values=['Activity Type', 'Count','Average Distance (mi)',
                            'Total Distance (mi)', 'Average Elevation Gain (ft)','Total Elevation Gain (ft)',
                            'Average Moving Time (hrs)','Average Elapsed Time (hrs)'],
                    line_color='darkslategray',
                    fill_color=headerColor,
                    align=['left', 'center'],
                    font=dict(color='Black')),
        cells=dict(values=[df1.index, df1.activity_type_count, df1.distance_mi_mean, df1.distance_mi_sum,
                           df1.total_elevation_gain_ft_mean,
                           df1.total_elevation_gain_ft_sum, df1.moving_time2_mean, df1.elapsed_time2_mean],
                   line_color='darkslategray',
                   # 2-D list of colors for alternating rows
                   fill_color=[[rowOddColor, rowEvenColor, rowOddColor, rowEvenColor] * 8],
                   align=['left'],))
    ])

    df2.sort_values(by=['moving_time2_sum'], ascending=False, inplace=True)

    fig_table2 = go.Figure(data=[go.Table(
        header=dict(values=['Athlete Name', 'Count','Average Distance (mi)',
                            'Total Distance (mi)', 'Average Elevation Gain (ft)','Total Elevation Gain (ft)',
                            'Total Moving Time (hrs)','Average Elapsed Time (hrs)'],
                    line_color='darkslategray',
                    fill_color=headerColor,
                    align=['left', 'center'],
                    font=dict(color='Black')),
        cells=dict(values=[df2.index, df2.activity_type_count, df2.distance_mi_mean, df2.distance_mi_sum,
                           df2.total_elevation_gain_ft_mean,
                           df2.total_elevation_gain_ft_sum, df2.moving_time2_sum, df2.elapsed_time2_mean],
                   line_color='darkslategray',
                   # 2-D list of colors for alternating rows
                   fill_color=[[rowOddColor, rowEvenColor, rowOddColor, rowEvenColor] * 8],
                   align=['left'],))
    ])

    df1['activity_type1'] = df1.index

    cols = df1.columns.tolist()
    cols.insert(0, cols.pop(cols.index('activity_type1')))
    df1 = df1.reindex(columns=cols)





    #TODO Make sure only column of interest are in table// subset columns
    tab_cols = [{'name': 'Activity Type', 'id': 'activity_type1'},
                {'name': 'Count', 'id': 'activity_type_count'},
                {'name': 'Average Distance (mi)', 'id': 'distance_mi_mean'},
                {'name': 'Total Distance (mi)', 'id': 'distance_mi_sum'},
                {'name': 'Average Elevation Gain (ft)', 'id': 'total_elevation_gain_ft_mean'},
                {'name': 'Total Elevation Gain (ft)', 'id': 'total_elevation_gain_ft_sum'},
                {'name': 'Average Moving Time (hrs)', 'id': 'moving_time2_mean'},
                {'name': 'Average Elapsed Time (hrs)', 'id': 'elapsed_time2_mean'}]
    fig_table2_broken = dash_table.DataTable(
        id='table1',
        #columns=[{"name": i, "id": i} for i in df1.columns],
        columns=tab_cols,
        data=df1.to_dict("rows"),
        style_header={
            'textAlign': 'left',
        },
        style_cell_conditional=[
            {
                'if': {'column_id': 'activity_type1'},
                'textAlign': 'left'
            }
        ]
    )

    df_heat = dist_calc(data)

    m = folium.Map([40.087424, -105.190813], zoom_start=11)
    # convert to (n, 2) nd-array format for heatmap
    stationArr = df_heat[['Latitude', 'Longitude']].values
    # plot heatmap
    m.add_children(plugins.HeatMap(stationArr, radius=7))
    m.save(os.path.join(os.path.dirname(__file__), 'assets', "heatmap.html"))


    graph1 = dcc.Graph(
        id='graph1',
        figure=fig_hist,
        className="six columns"
    )
    graph1_2 = dcc.Graph(
        id='graph1_2',
        figure=fig_hist2,
        className="six columns"
    )

    graph2 = dcc.Graph(
        id='graph2',
        figure=fig_par,
        className="twelve columns",
        style={'height': '500px'}
    )

    table = dcc.Graph(
       id='table1',
       figure=fig_table,
       className="twelve columns"
    )

    graph4 = dcc.Graph(
        id='graph4',
        figure=fig_sum,
        className="six columns"
    )
    graph4_2 = dcc.Graph(
        id='graph4_2',
        figure=fig_sump,
        className="six columns"
    )

    table2 = dcc.Graph(
        id='table2',
        figure=fig_table2,
        className="twelve columns"
    )

    map_plot = html_dash.Iframe(id='map', srcDoc=open(os.path.join(os.path.dirname(__file__), 'assets', "heatmap.html"),
                                                    'r').read(), width='90%', height='500')

    header = html_dash.H2(children="Seagate Longmont Strava Data")

    row1 = html_dash.Div(children=[table])

    row2 = html_dash.Div(children=[graph1,graph1_2], )

    row3 = html_dash.Div(children=[ graph2, ], )


    #row3 = html_dash.Div(children=[graph4])

    row4 = html_dash.Div(children=[map_plot])

    row5 = html_dash.Div(children=[ graph4,graph4_2])

    row6 = html_dash.Div(children=[table2])

    layout = html_dash.Div(children=[header, row1, row2, row3, row4, row5, row6], style={"text-align": "center"})
    app.layout = layout
    # app.run_server(debug=True)


def strava_login():
    driver.get('https://www.strava.com')

    # Fill out login form
    # checks if any tag 'a' contains the class as specified
    driver.find_element(By.XPATH,
                        "//a[@class='btn btn-default btn-login']").click()
    time.sleep(1)

    driver.find_element(By.XPATH, "//input[@type='email']").send_keys(
        keychain.app['user_email'])  # checks if any tag 'input' contains the attribute type as specified
    driver.find_element(By.XPATH, "//input[@type='password']").send_keys(keychain.app['user_pass'])
    driver.find_element(By.XPATH, "//button[@id='login-button']").click()
    print("Logging In")
    time.sleep(2)
    # return driver


def parse_feed(data):
    print("Parsing Data")
    # with open("activities_3817014087.html") as file:

    # m = re.search("achievementsController.setHash(.*);.*</script>", data)
    m = re.search(r"achievementsController.setHash\((.*)\)", data)
    if m:
        result = m.group(1)
        print(result)

        # using json.loads()
        # convert dictionary string to dictionary
        act_dict = json.loads(result)

        activities = [*act_dict]

        tree = html.fromstring(data)

        actuple = []
        for a in activities:
            timestamp = tree.xpath('//div[@id="' + a + '"]//time/@datetime')
            actuple.append((a.replace('Activity-', ''), timestamp[0]))

        print(actuple)

        return actuple


def parse_activity(activity_list):
    compile_data = []
    for act_id in activity_list:

        # limit age of activity
        if int(act_id[0]) < 3816205386:
            print("too old")
            continue

        print("Getting Activity: " + act_id[0])
        driver.get("https://www.strava.com/activities/" + act_id[0])

        time.sleep(random.random() * 3)
        driver.execute_script("window.scrollBy(0," + str(random.random() * 300) + ")")

        data = driver.page_source
        print("Parsing Data")
        # f = open("activities_" + str(act_id[0]) + ".html", "w+")
        # f.write(data)
        # f.close()

        # empty dictionary
        activity_detail = {}

        # first summary data
        m = re.search(r"lightboxData = {(.*?)}", data, flags=re.DOTALL)

        if m:
            result = m.group(1).replace('\n', ' ')
            title = re.search(r"title: \"(.*?)\",", result).group(1)
            a_fname = re.search(r" athlete_firstname: \"(.*?)\",", result).group(1)
            a_name = re.search(r" athlete_name: \"(.*?)\",", result).group(1)
            activ_type = re.search(r" activity_type: \"(.*?)\"", result).group(1)
            activity_detail = {'title': title, 'athlete_firstname': a_fname, 'athlete_name': a_name,
                               'activity_type': activ_type}

        try:
            athlete_id = re.search(r"activity_athlete_id\":(.*),", data).group(1)
            activity_detail['athlete_id'] = athlete_id
        except AttributeError:
            continue

        activity_detail['activity_id'] = act_id[0]
        tree = html.fromstring(data)
        # Parsing title details for activity detail type
        activity_type_detail = tree.xpath('//span[@class="title"]/text()')[1].replace('–', '').strip('\n')
        activity_detail['activity_type_detail'] = activity_type_detail

        # activity_local_time = tree.xpath('//time/text()')[0].strip('\n')
        activity_detail['activity_local_time'] = act_id[1]

        inline_stats = tree.xpath('//ul[@class="inline-stats section"]/li')

        # iterate the inline stat values
        for i in inline_stats:
            stat_label = i.findall('.//div[@class="label"]')
            stat_label = stat_label[0].text_content().replace('(?)', '').strip()
            # print("stat_label: " + stat_label)

            stat_data = i.text_content().strip().replace(stat_label, '').replace('(?)', '').strip()
            if stat_label == 'Duration':
                stat_label = 'Elapsed Time'
            activity_detail[stat_label] = stat_data

        try:
            find_dev = tree.xpath('//div[@class="device spans8"]/text()')[0].strip()
            # print(find_dev)
            activity_detail['device'] = find_dev
        except:
            print("fail device stats")
            pass

        more_stats = tree.xpath('//div[@class="section more-stats"]//div[@class="row"]')
        # Formatted based off of activity type
        # Format for a Run like activity
        try:
            mstat_val = more_stats[1].xpath('//div[@class="spans3"]//strong/text()')
            # print(mstat_val)
            if len(mstat_val) == 3:
                activity_detail['Elevation'] = mstat_val[0]
                activity_detail['Calories'] = mstat_val[1]
                activity_detail['Elapsed Time'] = mstat_val[2]
            if len(mstat_val) == 2:
                activity_detail['Elevation'] = mstat_val[0]
                activity_detail['Elapsed Time'] = mstat_val[1]
        except:
            print("fail run stats")
            pass

        try:
            act_gear = tree.xpath('//span[@class="gear-name"]/text()')[0].strip()
            if '\n' in act_gear:
                act_gear = act_gear.split('\n')[0]
            activity_detail['gear'] = act_gear
        except:
            pass

        # Formatted based off of activity type #ride like
        more_stats = tree.xpath('//div[@class="section more-stats"]')
        try:
            mstat_table = more_stats[0].xpath('//table[@class="unstyled"]')
            dfs = pd.read_html(etree.tostring(mstat_table[0]))[0]
            summary = {k: v.iloc[0, 1].split('  ')[0] for k, v in dfs.groupby('Unnamed: 0')}
            activity_detail.update(summary)

        except:
            print("fail bike stats")
            pass

        try:
            act_gear = tree.xpath('//span[@class="gear-name"]/text()')[0].strip()
            if '\n' in act_gear:
                act_gear = act_gear.split('\n')[0]
            activity_detail['gear'] = act_gear
        except:
            pass

        print(activity_detail)

        # stream data for heatmap
        # driver.get("https://www.strava.com/activities/" + act_id[0] + "/streams?stream_types%5B%5D=latlng&_=1595971502148")
        driver.get("https://www.strava.com/activities/" + act_id[0] + "/streams?stream_types%5B%5D=latlng")

        root = html.document_fromstring(driver.page_source)
        ll_stream = root.text_content()  # extract text// remove html elements
        if 'latlng' in ll_stream:
            activity_detail['latlng_stream'] = ll_stream

        compile_data.append(activity_detail)
    return compile_data


def get_details(output='strava_results.csv', club_id=121898):
    strava = StravaAccess()
    club_details = strava.client.get_club(club_id)

    strava_login()
    driver.get("https://www.strava.com/clubs/" + str(club_id) + "/feed?num_entries=100")
    print("Getting Activities")

    # act like a user
    driver.execute_script("window.scrollBy(0," + str(random.random() * 300) + ")")

    club_feed = driver.page_source

    # f = open("club_feed.html", "w+")
    # f.write(club_feed)
    # f.close()

    activity_list = parse_feed(club_feed)

    if os.path.isfile(os.path.join(os.path.dirname(__file__), 'output', output)):
        have_activities = pd.read_csv(os.path.join(os.path.dirname(__file__), 'output', output),
                                      usecols=['activity_id'], squeeze=True)
        # convert to a list of strings for comparison
        have_activities = [str(i) for i in list(have_activities)]
        # remove if already have
        activity_list = [i for i in activity_list if i[0] not in have_activities]

    output_data = parse_activity(activity_list)
    if len(output_data) > 0:
        df_details = pd.DataFrame.from_dict(output_data)
        df_details['club_id'] = club_id
        df_details['club_name'] = club_details.name
        df_details['club_mem_count'] = club_details.member_count
        print(df_details)

        if os.path.isfile(os.path.join(os.path.dirname(__file__), 'output', output)):
            df_temp = pd.read_csv(os.path.join(os.path.dirname(__file__), 'output', output))
            df_details = pd.concat([df_details, df_temp], sort=False)

        out_path = os.path.join(os.path.dirname(__file__), 'output', output)
        df_details.to_csv(out_path, index=False)

        return df_details
    else:
        return pd.read_csv(os.path.join(os.path.dirname(__file__), 'output', output))


def main():
    parser = argparse.ArgumentParser(description='Club activity data for Strava club of interest.')
    parser.add_argument('-c', '--use_cache', action='store_true',
                        help='Use cached data from Strava API, otherwise will query API for activity data')
    parser.add_argument('-o', '--out_file', action='store',
                        default=os.path.join(os.path.dirname(__file__), 'output', 'strava_results.csv'),
                        help='Output location of activity data, used both as seed for activity data if not refreshing and as output location')
    parser.add_argument('-id', '--club_id', action='store', type=int, default=121898,
                        help='Club ID of interest. Default is "Seagate Longmont"')

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.join(os.path.dirname(__file__), 'assets')), exist_ok=True)

    if args.use_cache:
        try:
            # df = pd.read_csv(args.out_file)
            df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'output', args.out_file))
        except FileNotFoundError:
            print('Could not load previous file, refreshing activity data...')
            df = get_details(output=args.out_file, club_id=args.club_id)
    else:
        #df = get_details(output=args.out_file, club_id=args.club_id)
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'output', args.out_file))

    # df_details = get_details(output=args.out_file, club_id=args.club_id)

    # df_master = merge_data(df, df_details)

    #print(df)

    plot_data(df)

    # debug=True enables hot reload in new versions
    #app.run_server(debug=True, threaded=True)

    # production
    host = socket.gethostbyname(socket.gethostname())
    # print(host)
    app.run_server(debug=False, host=host, port=8050)


if __name__ == '__main__':
    # global dash app
    # external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
    # app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
    app = dash.Dash(__name__)

    main()
