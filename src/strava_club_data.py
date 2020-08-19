import argparse
import json
import os
import random
import re
import requests
import time
import urllib.parse
import webbrowser

import pandas as pd
from stravalib.client import Client

import keychain

from selenium.webdriver.common.by import By
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from lxml import html
from lxml import etree

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


def parse_feed(data):
    print("Parsing Data")

    # handy object to get some of the details
    m = re.search(r"achievementsController.setHash\((.*)\)", data)
    if m:
        result = m.group(1)
        print(result)

        act_dict = json.loads(result)

        activities = [*act_dict]

        tree = html.fromstring(data)

        actuple = []
        for a in activities:
            timestamp = tree.xpath('//div[contains(@id,"' + a + '")]//time/@datetime')
            if timestamp:
                actuple.append((a.replace('Activity-', ''), timestamp[0]))
            elif len(timestamp) == 0:

                group_act = tree.xpath('//div[@class="feed-entry group-activity"]')

                for group_part in group_act:
                    detail = group_part.xpath('//li[@id="' + a + '"]')
                    if detail:
                        timestamp = group_part.xpath('.//time/@datetime')
                        actuple.append((a.replace('Activity-', ''), timestamp[0]))
            else:
                actuple.append((a.replace('Activity-', ''), ''))

        print(actuple)

        return actuple


def parse_activity(activity_list):
    compile_data = []

    counter = 0

    for act_id in activity_list:

        # limit age of activity
        if int(act_id[0]) < 3816205386:
            print("too old")
            continue
        # if counter > 10:
        #    break

        print("Getting Activity: " + act_id[0])
        driver.get("https://www.strava.com/activities/" + act_id[0])

        time.sleep(random.random() * 3)
        driver.execute_script("window.scrollBy(0," + str(random.random() * 300) + ")")

        data = driver.page_source
        print("Parsing Data")

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

        activity_detail['activity_local_time'] = str(act_id[1])

        inline_stats = tree.xpath('//ul[@class="inline-stats section"]/li')

        # iterate the inline stat values
        for i in inline_stats:
            stat_label = i.findall('.//div[@class="label"]')
            stat_label = stat_label[0].text_content().replace('(?)', '').strip()

            stat_data = i.text_content().strip().replace(stat_label, '').replace('(?)', '').strip()
            if stat_label == 'Duration':
                stat_label = 'Elapsed Time'
            activity_detail[stat_label] = stat_data

        try:
            find_dev = tree.xpath('//div[@class="device spans8"]/text()')[0].strip()
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

        print(activity_detail)

        # Some data can be retrieved from a public page to save on stream requests
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.119 Safari/537.36"
        }

        latlng = ''
        r = requests.get("https://www.strava.com/activities/" + act_id[0], headers=headers)
        if r.status_code == 200:
            # does error out if too many requests // adding headers helped
            # https://www.scrapehero.com/how-to-rotate-proxies-and-ip-addresses-using-python-3/
            m = re.search('quot;latlng&quot;:(.*),&quot;unitSystem', r.text)
            if m:
                print("Getting track from public activity")
                latlng = '{"latlng":' + m.group(1) + '}'
                activity_detail['latlng_stream'] = latlng
        if len(latlng) < 5:
            print("Getting activity track from stream")
            driver.get("https://www.strava.com/activities/" + act_id[0] + "/streams?stream_types%5B%5D=latlng")

            root = html.document_fromstring(driver.page_source)
            ll_stream = root.text_content()  # extract text// remove html elements
            if 'latlng' in ll_stream:
                activity_detail['latlng_stream'] = ll_stream

        compile_data.append(activity_detail)
        counter = counter + 1
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

    activity_list = parse_feed(club_feed)

    if os.path.isfile(os.path.join(os.path.dirname(__file__), 'output', output)):
        have_activities = pd.read_csv(os.path.join(os.path.dirname(__file__), 'output', output), usecols=['activity_id'])
        # convert to a list of strings for comparison
        have_activities = [str(i) for i in list(have_activities['activity_id'])]
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
    parser.add_argument('-o', '--out_file', action='store',
                        default=os.path.join(os.path.dirname(__file__), 'output', 'strava_results.csv'),
                        help='Output location of activity data, used both as seed for activity data if not refreshing and as output location')
    parser.add_argument('-id', '--club_id', action='store', type=int, default=121898,
                        help='Club ID of interest. Default is "Seagate Longmont"')

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.join(os.path.dirname(__file__), 'assets')), exist_ok=True)

    # random delay to start data pull
    # time.sleep(random.randint(240, 3600))

    get_details(output=args.out_file, club_id=args.club_id)
    driver.quit()


if __name__ == '__main__':
    main()
