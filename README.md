# Strava Club Dashboard
Plotting of Strava club data

## Usage
1. Rename keychain.py.dist to keychain.py in src folder.
2. Fill in Strava APP client_id and client_secret in the keychain (This can be found at https://www.strava.com/settings/api#).
3. run strava_club.py from terminal. Use -h to view options, default club is 'Seagate Longmont'.
4. If first run, default web browser will open to OAuth2 page and ask for user authentication.
5. When prompted, paste the redirect URL to terminal (e.g. https://localhost/...)
    Note: This page will not display anything as there is no web server running at the redirect URL. This is just to get the auth code used to generate OAuth Token.

Data will store to local file for future reuse if API interaction not needed<br>
Option -c to attempt to use local file, otherwise data will be refreshed from API<br>
df = pd.read_csv('strava_results.csv')<br>
plot_data(df)<br>
