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

from selenium.webdriver.common.by import By
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from lxml import html
import time

import random
import re

# CHROMEDRIVER_PATH = '/usr/local/bin/chromedriver'
WINDOW_SIZE = "1920,1080"
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--window-size=%s" % WINDOW_SIZE)
chrome_options.add_argument('--no-sandbox')
# driver = webdriver.Chrome(executable_path=CHROMEDRIVER_PATH,
#                          options=chrome_options
#                          )
driver = webdriver.Chrome(options=chrome_options)
driver.implicitly_wait(10)  # seconds


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

    data_list = []
    for index, row in data[data['latlng_stream'].notnull()].iterrows():
        print(row['latlng_stream'])
        data_list.append(pd.DataFrame(json.loads(row['latlng_stream'])['latlng'], columns=['Latitude', 'Longitude']))

    df_heat = pd.concat(data_list)

    fig_heat = px.density_mapbox(df_heat, lat='Latitude', lon='Longitude', radius=3,
                                 # center=dict(lat=0, lon=180), zoom=0,
                                 center=dict(lat=40.13801, lon=-105.1726), zoom=10,
                                 mapbox_style="stamen-terrain")

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

    graph4 = dcc.Graph(
        id='graph4',
        figure=fig_heat,
        className="six columns"
    )

    header = html.H2(children="Seagate Longmont Strava Data")

    row1 = html.Div(children=[graph1, graph2], )

    row2 = html.Div(children=[graph3])

    row3 = html.Div(children=[graph4])

    layout = html.Div(children=[header, row1, row2, row3], style={"text-align": "center"})
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
            stat_label = stat_label[0].text_content().strip()
            # print("stat_label: " + stat_label)

            stat_data = i.text_content().strip().replace(stat_label, '').strip()
            # print("stat_data: " + stat_data)
            activity_detail[stat_label.lower().replace(' ', '_')] = stat_data

        # more_stats = tree.xpath('//div[@class="section more-stats"]/div')

        # #Formatted based off of activity type
        # more_stat_list = []
        # for j in more_stats:
        #    stat_label = j.findall('.//div[@class="spans5"]')
        #    for k in stat_label:
        #        k.text_content().strip()
        #    print(stat_label[0].text_content().strip())
        #    stat_value = j.findall('.//div[@class="spans3"]')
        #    for k in stat_label:
        #        k.text_content().strip()
        #    print(stat_value[0].text_content().strip())
        #    #print(j.text_content().strip())
        #    #more_stat_list.append(j.text_content().strip())

        # print(more_stat_list)

        # parse comment/title data for word cloud?

        print(activity_detail)

        # stream data for heatmap
        driver.get("https://www.strava.com/activities/" + act_id[0] + "/streams?stream_types%5B%5D=latlng")

        root = html.document_fromstring(driver.page_source)
        ll_stream = root.text_content()  # extract text// remove html elements

        activity_detail['latlng_stream'] = ll_stream

        compile_data.append(activity_detail)
    return compile_data


def merge_data(df1, df2):
    # df1 is base data from api
    # df2 is detail activity data

    if df2 is None:
        return pd.read_csv(os.path.join(os.path.dirname(__file__), 'output', 'strava_master.csv'))

    # if df2 empty read file and return

    df1['athlete_firstname'] = df1['athlete'].str.split().str.get(0)

    # convert from object to delta time to merge on value
    df2['moving_time'] = df2.apply(
        lambda row: row['moving_time'] if row['moving_time'].count(":") > 1 else "".join(["00:", row['moving_time']]),
        axis=1)
    df2['moving_time'] = df2.apply(
        lambda row: row['moving_time'] if len(row['moving_time']) == 8 else "".join(["0", row['moving_time']]),
        axis=1)

    #df2['distance_mi'] = df2['distance'].str.replace('mi', '')

    # Make sure same type
    #df1['distance_mi'] = df1['distance_mi'].astype(float)
    #df2['distance_mi'] = df2['distance_mi'].astype(float)

    # Make sure same type
    df2['moving_time'] = pd.to_timedelta(df2['moving_time'])
    df1['moving_time'] = pd.to_timedelta(df1['moving_time'])

    df2.rename(columns={'title': 'name'}, inplace=True)

    # df1_sort = df1.sort_values('distance_mi')
    # df2_sort = df2.sort_values('distance_mi')
    # df3 = pd.merge_asof(df1_sort, df2_sort, on='distance_mi', by=['athlete_firstname', 'moving_time'], tolerance=0.2)
    df3 = pd.merge(df1, df2, on=['athlete_firstname', 'moving_time', 'name'])

    # delete rows with no merge and then join to master dataset
    df_update = df3[df3['activity_local_time'].notnull()]

    if os.path.isfile(os.path.join(os.path.dirname(__file__), 'output', 'strava_master.csv')):
        df_temp = pd.read_csv(os.path.join(os.path.dirname(__file__), 'output', 'strava_master.csv'))
        df_update = pd.concat([df_update, df_temp])

    out_path = os.path.join(os.path.dirname(__file__), 'output', 'strava_master.csv')
    df_update.to_csv(out_path, index=False)

    return df_update


def get_details(club_id=121898):
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

    if os.path.isfile(os.path.join(os.path.dirname(__file__), 'output', 'strava_master.csv')):
        have_activities = pd.read_csv(os.path.join(os.path.dirname(__file__), 'output', 'strava_master.csv'),
                                      usecols=['activity_id'], squeeze=True)
        # convert to a list of strings for comparison
        have_activities = [str(i) for i in list(have_activities)]
        # remove if already have
        activity_list = [i for i in activity_list if i[0] not in have_activities]

    output_data = parse_activity(activity_list)
    if len(output_data) > 0:
        df_details = pd.DataFrame.from_dict(output_data)
        print(df_details)
        out_path = os.path.join(os.path.dirname(__file__), 'output', 'strava_results_details.csv')
        df_details.to_csv(out_path, index=False)
        return df_details
    else:
        return None


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

    if args.use_cache:
        try:
            df = pd.read_csv(args.out_file)
        except FileNotFoundError:
            print('Could not load previous file, refreshing activity data...')
            df = get_data(output=args.out_file, club_id=args.club_id)
    else:
        df = get_data(output=args.out_file, club_id=args.club_id)

    df_details = get_details(club_id=args.club_id)

    df_master = merge_data(df, df_details)

    print(df_master)

    # TODO clean up the df on disk

    plot_data(df_master)

    # debug=True enables hot reload in new versions
    app.run_server(debug=False, threaded=True)

    # production
    # host = socket.gethostbyname(socket.gethostname())
    # print(host)
    # app.run_server(debug=False, host=host, port=8050)


if __name__ == '__main__':
    # global dash app
    # external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
    # app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
    app = dash.Dash(__name__)

    main()
