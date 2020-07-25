# Strava Club Dashboard
Plotting of Strava club data


add strava_config.py to same folder as strava_club.py
format of file

CLIENT_SECRET = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'<br>
CLIENT_ID = 11111<br>
CODE = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'

Generate CODE from url copy from url on redirect
Replace client ID with value from app

http://www.strava.com/oauth/authorize?client_id=11111&response_type=code&redirect_uri=http://localhost/exchange_token&approval_prompt=force&scope=profile:read_all,activity:read_all


Data loaded from local file after downloading<br>
get_data() ## only needs to run this to get data, can comment out afterfule has been created<br>
df = pd.read_csv('strava_results.csv')<br>
plot_data(df)<br>

When getting data for the first time first_run = None needs to be set first_run = 1  (or any other not None value)
