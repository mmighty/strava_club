from datetime import datetime, timedelta
import argparse
import json
import os
import socket
import time

import dash
import dash_core_components as dcc
import dash_html_components as html_dash
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import keychain

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# heatmap
import folium
from folium import plugins

from flask_caching import Cache

# https://flask-caching.readthedocs.io/en/latest/

app = dash.Dash(__name__)
#cache = Cache(app.server, config = {
#    "DEBUG": True,          # some Flask specific configs
#    "CACHE_TYPE": "simple", # Flask-Caching related configs
#    "CACHE_DEFAULT_TIMEOUT": 300
#})
#app.config.supress_callback_exceptions = True

RELOAD_INTERVAL = 3600 # reload interval in seconds

def refresh_data_every():
    while True:
        refresh_data()
        time.sleep(RELOAD_INTERVAL)

def refresh_data():
    global df
    ### some expensive computation function to update dataframe
    df = get_df()

#@cache.memoize(timeout=5)
def get_df():
    print('refresh base data')
    df_base = pd.read_csv(os.path.join(os.path.dirname(__file__), 'output', 'strava_results.csv'))
    pd.to_datetime(df_base['activity_local_time'])
    df_base['activity_local_time2'] = pd.to_datetime(df_base['activity_local_time']).dt.tz_convert('US/Mountain')
    df_base.drop(['activity_local_time'], axis=1, inplace=True)
    return df_base


df = get_df()

executor = ThreadPoolExecutor(max_workers=1)
executor.submit(refresh_data_every)

def clean_data(data):
    # set category for columns
    data['activity_type'] = data['activity_type'].astype('category')
    data['activity_type_detail'] = data['activity_type_detail'].astype('category')
    data['athlete_firstname'] = data['athlete_firstname'].astype('category')
    data['athlete_name'] = data['athlete_name'].astype('category')
    data['club_name'] = data['club_name'].astype('category')
    data['device'] = data['device'].astype('category')

    # get data organized for heatmap
    df_heat = dist_calc(data)

    # Remove not needed columns
    data.drop(['latlng_stream'], axis=1, inplace=True)
    data.drop(['Tough Relative Effort', 'Historic Relative Effort', 'Massive Relative Effort'], axis=1, inplace=True)

    # More cleanup
    data['Power'] = data['Power'].str.replace('W', '')
    data['Power'] = data['Power'].fillna(0).astype('int32')
    data['Cadence'] = data['Cadence'].fillna(0).astype('int32')
    data['club_mem_count'] = data['club_mem_count'].astype('int32')

    data['Temperature'] = pd.to_numeric(data['Temperature'].str.replace('℉', '')).fillna(-63).astype('int32')
    data['Heart Rate'] = pd.to_numeric(data['Heart Rate'].str.replace('bpm', '')).fillna(0).astype('int32')
    data['Calories'] = pd.to_numeric(data['Calories'].str.replace(',', '')).fillna(0).astype('int32')

    data['Moving Time'].fillna(data['Elapsed Time'], inplace=True)
    data['Elapsed Time'].fillna(data['Moving Time'], inplace=True)

    # convert from object to delta time
    data['Moving Time'] = data.apply(
        lambda row: "".join(["0", row['Moving Time']]) if len(row['Moving Time']) == 4 else row['Moving Time'],
        axis=1)
    data['Moving Time'] = data.apply(
        lambda row: row['Moving Time'] if row['Moving Time'].count(":") > 1 else "".join(["00:", row['Moving Time']]),
        axis=1)
    data['Moving Time'] = data.apply(
        lambda row: row['Moving Time'] if len(row['Moving Time']) == 8 else "".join(["0", row['Moving Time']]),
        axis=1)

    # convert from object to delta time
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

    # Make conversion to timedelta
    data['Moving Time'] = pd.to_timedelta(data['Moving Time'])
    data['Elapsed Time'] = pd.to_timedelta(data['Elapsed Time'])

    # numeric time in hours
    data['moving_time2'] = data['Moving Time'].values.astype(np.int64) / 3.6e+12
    data['elapsed_time2'] = data['Elapsed Time'].values.astype(np.int64) / 3.6e+12

    # Distance formatting
    data['Distance'] = data['Distance'].str.replace(',', '')
    data[['Distance', 'Distance_Unit']] = data['Distance'].str.extract(r'([\d.]+)(\D+)')
    unit_convert = {'mi': 1, 'm': (1 / 1609)}
    data['distance_mi'] = pd.to_numeric(data['Distance']) * data['Distance_Unit'].map(unit_convert).fillna(0)
    data['distance_mi'] = data['distance_mi'].astype('float32')
    data.drop(['Distance', 'Distance_Unit'], axis=1, inplace=True)

    # Elevation formatting
    data['Elevation'].fillna('0ft', inplace=True)
    data['total_elevation_gain_ft'] = data['Elevation'].str.replace('ft', '').replace(',', '')
    data['total_elevation_gain_ft'] = data['total_elevation_gain_ft'].str.replace(',', '')
    data['total_elevation_gain_ft'] = data['total_elevation_gain_ft'].astype('int32')
    data['total_elevation_gain_ft'].fillna(0, inplace=True)
    # Keep column for formatted column
    # data.drop(['Elevation'], axis=1, inplace=True)

    # for time formatting
    # pd.to_datetime(data['activity_local_time'])
    # data['activity_local_time2'] = pd.to_datetime(data['activity_local_time']).dt.tz_convert('US/Mountain')
    # data.drop(['activity_local_time'], axis=1, inplace=True)

    return data, df_heat


def dist_calc(frame):
    data_list = []
    for index, row in frame[frame['latlng_stream'].notnull()].iterrows():
        # print(row)
        if len(row['latlng_stream']) > 25:
            if json.loads(row['latlng_stream']):
                # print(row['latlng_stream'])
                data_list.append(
                    pd.DataFrame(json.loads(row['latlng_stream'])['latlng'], columns=['Latitude', 'Longitude']))

    df_heat = pd.concat(data_list)

    df_heat[['Latitude', 'Longitude']] = df_heat[['Latitude', 'Longitude']].apply(pd.to_numeric, downcast='float')

    # return part of dataset every 10th row
    return df_heat.iloc[0::10, :]


def time_format(x):
    ts = x.total_seconds()
    hours, remainder = divmod(ts, 3600)
    minutes, seconds = divmod(remainder, 60)
    return '{}:{:02d}:{:02d}'.format(int(hours), int(minutes), int(seconds))


def plot_data(data):
    print("entering plot")

    data, df_heat = clean_data(data)

    # Some recoding for scales and index
    data['activity_code'], factor_list = pd.factorize(data['activity_type'], sort=True)
    factor_list = list(factor_list)

    # Summary data
    df_sum = data.groupby([data['activity_type'], data['activity_local_time2'].dt.date]).agg({
        'distance_mi': 'sum',
        'total_elevation_gain_ft': 'sum',
        'moving_time2': 'sum',
        'elapsed_time2': 'sum',

    }).groupby(level=0).cumsum().reset_index()

    df_sum_athlete = data.groupby([data['athlete_name'], data['activity_local_time2'].dt.date]).agg({
        'distance_mi': 'sum',
        'total_elevation_gain_ft': 'sum',
        'moving_time2': 'sum',
        'elapsed_time2': 'sum',

    }).groupby(level=0).cumsum().reset_index()

    # Build colorscale
    fix_color_map = {}
    custom_color_scale = []
    activity_type_list = data['activity_type'].unique()

    count = 0

    if len(factor_list) > 1:
        for i in activity_type_list:
            if count > 12:
                count = 0
            fix_color_map[i] = px.colors.cyclical.Phase[count]
            custom_color_scale.append(
                [float(factor_list.index(i)) / (len(factor_list) - 1), px.colors.cyclical.Phase[count]])
            count = count + 1
    else:
        fix_color_map[0] = px.colors.cyclical.Phase[0]
        custom_color_scale.append([0.0, px.colors.cyclical.Phase[0]])

    custom_color_scale = sorted(custom_color_scale, key=lambda x: x[0])

    # https://plotly.com/python/v3/selection-events/

    fig_sum = px.line(df_sum, x='activity_local_time2', y='distance_mi', labels={
        "distance_mi": "Value",
        "activity_local_time2": "Date ",
        "activity_type": "Activity Type"
    }, color_discrete_map=fix_color_map, template="simple_white", color='activity_type')
    fig_sum.update_traces(connectgaps=True)
    fig_sum.update_layout(
        yaxis_title="Distance(mi)",
        title={
            'text': "Cumulative Club Statistics",
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'})

    fig_sum2 = px.line(df_sum, x='activity_local_time2', y='total_elevation_gain_ft', color_discrete_map=fix_color_map,
                       template="simple_white", color='activity_type')
    fig_sum3 = px.line(df_sum, x='activity_local_time2', y='elapsed_time2', color_discrete_map=fix_color_map,
                       template="simple_white", color='activity_type')
    # fig_sum2.update_traces(connectgaps=True)

    fig_sum.update_layout(
        updatemenus=[
            dict(

                buttons=list([
                    dict(label='Total Distance',
                         method='update',
                         args=[{'y': [fig_sum.data[j].y for j in range(len(fig_sum.data))]},
                               {'yaxis': {'title': 'Distance(mi)'}}
                               ]),
                    dict(label='Total Elevation',
                         method='update',
                         args=[{'y': [fig_sum2.data[j].y for j in range(len(fig_sum2.data))]},
                               {'yaxis': {'title': 'Elevation(ft)'}},
                               ]),
                    dict(label='Total Elapsed Time',
                         method='update',
                         args=[{'y': [fig_sum3.data[j].y for j in range(len(fig_sum3.data))]},
                               {'yaxis': {'title': 'Elapsed_Time(hr)'}},
                               ]),
                ]),
                # direction where the drop-down expands when opened
                direction='down',
                # positional arguments
                x=0.01,
                xanchor='left',
                y=0.99,
                yanchor='bottom',
                font=dict(size=11)
            )]
    )

    fig_sump = px.line(df_sum_athlete, x='activity_local_time2', y='distance_mi', labels={
        "distance_mi": "Value",
        "activity_local_time2": "Date ",
        "athlete_name": "Athlete Name"
    }, color_discrete_map=fix_color_map, template="simple_white", color='athlete_name')
    fig_sump.update_traces(connectgaps=True)
    fig_sump.update_layout(
        yaxis_title="Distance(mi)",
        title={
            'text': "Cumulative Member Statistics",
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'})

    fig_sump2 = px.line(df_sum_athlete, x='activity_local_time2', y='total_elevation_gain_ft',
                        color_discrete_map=fix_color_map,
                        template="simple_white", color='athlete_name')
    fig_sump3 = px.line(df_sum_athlete, x='activity_local_time2', y='elapsed_time2', color_discrete_map=fix_color_map,
                        template="simple_white", color='athlete_name')
    # fig_sum2.update_traces(connectgaps=True)

    fig_sump.update_layout(
        updatemenus=[
            dict(

                buttons=list([
                    dict(label='Total Distance',
                         method='update',
                         args=[{'y': [fig_sump.data[j].y for j in range(len(fig_sump.data))]},
                               {'yaxis': {'title': 'Distance(mi)'}}
                               ]),
                    dict(label='Total Elevation',
                         method='update',
                         args=[{'y': [fig_sump2.data[j].y for j in range(len(fig_sump2.data))]},
                               {'yaxis': {'title': 'Elevation(ft)'}},
                               ]),
                    dict(label='Total Elapsed Time',
                         method='update',
                         args=[{'y': [fig_sump3.data[j].y for j in range(len(fig_sump3.data))]},
                               {'yaxis': {'title': 'Elapsed_Time(hr)'}},
                               ]),
                ]),
                # direction where the drop-down expands when opened
                direction='down',
                # positional arguments
                x=0.01,
                xanchor='left',
                y=0.99,
                yanchor='bottom',
                font=dict(size=11)
            )]
    )

    # summary data // group
    df1 = data.groupby(['activity_type']).agg({
        'activity_type': 'count',
        'distance_mi': ['mean', 'sum'],
        'total_elevation_gain_ft': ['mean', 'sum'],
        'moving_time2': 'mean',
        'elapsed_time2': 'mean',

    })

    df1.columns = list(map('_'.join, df1.columns))
    df1.index.name = 'type'

    # summary data // athlete
    df2 = data.groupby(['athlete_name']).agg({
        'activity_type': 'count',
        'distance_mi': ['mean', 'sum'],
        'total_elevation_gain_ft': ['mean', 'sum'],
        'moving_time2': ['mean', 'sum'],
        'elapsed_time2': ['mean', 'sum'],

    })

    df2.columns = list(map('_'.join, df2.columns))
    df2.index.name = 'name'

    fig_par = go.Figure(data=go.Parcoords(
        line=dict(color=data['activity_code'],
                  colorscale=custom_color_scale),
        dimensions=list([
            dict(tickvals=list(range(0, len(factor_list) + 1)), ticktext=factor_list, label='Activity ID',
                 values=data['activity_code']),
            dict(label='Distance (mi)', values=data['distance_mi']),
            dict(label='Elevation Gain (ft)', values=data['total_elevation_gain_ft']),
            dict(label='Moving Time (hr)', values=data['moving_time2']),
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

    fig_hist = px.histogram(data, x="activity_type", labels={
        "activity_type": "Activity Type", }, color='activity_type', color_discrete_map=fix_color_map,
                            template="simple_white")
    fig_hist.update_layout(
        title={
            'text': "Count Of Activity Types",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'})

    data['activity_start_hour'] = data['activity_local_time2'].dt.hour
    fig_hist2 = px.histogram(data, x="activity_start_hour", labels={
        "activity_start_hour": "Time Of Day", }, color='activity_type', color_discrete_map=fix_color_map,
                             template="simple_white")
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

    header_color = 'grey'
    row_even_color = 'lightgrey'
    row_odd_color = 'white'

    fig_table = go.Figure(data=[go.Table(
        header=dict(values=['Activity Type', 'Count', 'Average Distance (mi)',
                            'Total Distance (mi)', 'Average Elevation Gain (ft)', 'Total Elevation Gain (ft)',
                            'Average Moving Time (hrs)', 'Average Elapsed Time (hrs)'],
                    line_color='darkslategray',
                    fill_color=header_color,
                    align=['left', 'center'],
                    font=dict(color='Black')),
        cells=dict(values=[df1.index, df1.activity_type_count, df1.distance_mi_mean, df1.distance_mi_sum,
                           df1.total_elevation_gain_ft_mean,
                           df1.total_elevation_gain_ft_sum, df1.moving_time2_mean, df1.elapsed_time2_mean],
                   line_color='darkslategray',
                   # 2-D list of colors for alternating rows
                   fill_color=[[row_odd_color, row_even_color, row_odd_color, row_even_color] * 8],
                   align=['left'], ))
    ])

    df2.sort_values(by=['moving_time2_sum'], ascending=False, inplace=True)

    fig_table2 = go.Figure(data=[go.Table(
        header=dict(values=['Athlete Name', 'Count', 'Average Distance (mi)',
                            'Total Distance (mi)', 'Average Elevation Gain (ft)', 'Total Elevation Gain (ft)',
                            'Total Moving Time (hrs)', 'Average Elapsed Time (hrs)'],
                    line_color='darkslategray',
                    fill_color=header_color,
                    align=['left', 'center'],
                    font=dict(color='Black')),
        cells=dict(values=[df2.index, df2.activity_type_count, df2.distance_mi_mean, df2.distance_mi_sum,
                           df2.total_elevation_gain_ft_mean,
                           df2.total_elevation_gain_ft_sum, df2.moving_time2_sum, df2.elapsed_time2_mean],
                   line_color='darkslategray',
                   # 2-D list of colors for alternating rows
                   fill_color=[[row_odd_color, row_even_color, row_odd_color, row_even_color] * 8],
                   align=['left'], ))
    ])

    m = folium.Map([40.087424, -105.190813], zoom_start=11)
    # convert to (n, 2) nd-array format for heatmap
    gpx_track = df_heat[['Latitude', 'Longitude']].values
    # plot heatmap
    m.add_child(plugins.HeatMap(gpx_track, radius=7))
    heat_output = m._repr_html_()
    # m.save(os.path.join(os.path.dirname(__file__), 'assets', "heatmap.html"))
    # m.save(heat_output)
    # print(heat_output)

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

    map_plot = html_dash.Iframe(id='map', srcDoc=heat_output, width='90%', height='500')

    row1 = html_dash.Div(children=[table])

    row2 = html_dash.Div(children=[graph1, graph1_2], )

    row3 = html_dash.Div(children=[graph2, ], )

    row4 = html_dash.Div(children=[map_plot])

    row5 = html_dash.Div(children=[graph4, graph4_2])

    row6 = html_dash.Div(children=[table2])

    layout = html_dash.Div(children=[row1, row2, row3, row4, row5, row6])

    return layout
    # return layout
    # app.layout = layout
    # app.run_server(debug=True)




header = html_dash.H2(children="Seagate Longmont Strava Data")

row0 = html_dash.Div([
    html_dash.Div(className="input_header",
                  children=[
                      dcc.DatePickerRange(
                          id='my-date-picker-range',
                          min_date_allowed=df['activity_local_time2'].min(),
                          max_date_allowed=df['activity_local_time2'].max(),
                          initial_visible_month=df['activity_local_time2'].max(),
                          end_date=df['activity_local_time2'].max(),
                          start_date=df['activity_local_time2'].max() - timedelta(days=14)
                      ),

                      dcc.Dropdown(
                          id='activity_filter',
                          multi=True,
                          clearable=False
                      ),
                  ]),
    html_dash.Div(id='output-container-date-picker-range')
])

app.layout = html_dash.Div(children=[header, row0], style={"text-align": "center"})


@app.callback(
    [dash.dependencies.Output('activity_filter', 'options'), dash.dependencies.Output('activity_filter', 'value')],
    [dash.dependencies.Input('my-date-picker-range', 'start_date'),
     dash.dependencies.Input('my-date-picker-range', 'end_date')
     ])
def get_activity_types(start_date, end_date):
    mask = (df['activity_local_time2'] >= start_date) & (df['activity_local_time2'] <= end_date)
    activity_type_list = df['activity_type'].loc[mask].unique()
    return [{'label': i, 'value': i} for i in activity_type_list], activity_type_list


@app.callback(
    dash.dependencies.Output('output-container-date-picker-range', 'children'),
    [dash.dependencies.Input('my-date-picker-range', 'start_date'),
     dash.dependencies.Input('my-date-picker-range', 'end_date'),
     dash.dependencies.Input('activity_filter', 'value')
     ])
def update_output(start_date, end_date, value):
    mask = (df['activity_local_time2'] >= start_date) & (df['activity_local_time2'] <= end_date) & (
        df['activity_type'].isin(value))
    filtered_df = df.loc[mask].reset_index()
    contents = plot_data(filtered_df)
    print('finish plot')
    return contents


if __name__ == '__main__':
    # debug=True enables hot reload in new versions
    app.run_server(debug=True, threaded=True)

    # production
    # host = socket.gethostbyname(socket.gethostname())
    # print(host)
    # app.run_server(debug=False, host=host, port=8050)
