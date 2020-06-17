import os, json
import pandas as pd
import numpy as np

import spotipy
import spotipy.util as util

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

### set API keys
apikeys = json.load(open("data/api-keys.json"))
os.environ["SPOTIPY_CLIENT_ID"]     = apikeys["spotipy-client-id"]
os.environ["SPOTIPY_CLIENT_SECRET"] = apikeys["spotipy-client-secret"]
os.environ["SPOTIPY_REDIRECT_URI"]  = apikeys["redirect-url"]

### set user_id
user_id = '129874447'

### connect to spotify
token = util.prompt_for_user_token(user_id, \
                                   scope = 'user-library-read, playlist-modify-public, playlist-modify-private')
sp = spotipy.Spotify(auth = token)

### function to get the current user's saved tracks (track name, artist, id)
def get_saved_tracks(limit = 50, offset = 0):
    saved_tracks = [ ]
    
    # get initial list of tracks to determine length
    saved_tracks_obj = sp.current_user_saved_tracks(limit = limit, offset = offset)
    num_saved_tracks = saved_tracks_obj['total']
    
    # loop through to get all saved tracked
    while (offset < num_saved_tracks):
        saved_tracks_obj = sp.current_user_saved_tracks(limit = limit, offset = offset)
        
        # add track information to running list
        for track_obj in saved_tracks_obj['items']:
            saved_tracks.append({
                'name': track_obj['track']['name'],
                'artists': ', '.join([artist['name'] for artist in track_obj['track']['artists']]),
                'track_id': track_obj['track']['id']
            })
            
        offset += limit
        
    return saved_tracks

### function to get tracks from a specified playlist (track name, artist, id)
def get_playlist_tracks(user_id, playlist_id, limit = 100, offset = 0):
    playlist_tracks = [ ]
    
    # get initial initial list of tracks in playlist to determine length
    playlist_obj = sp.user_playlist_tracks(user = user_id, playlist_id = playlist_id, \
                                           limit = limit, offset = offset)
    num_playlist_tracks = playlist_obj['total']
    
    # loop through to get all playlist tracks
    while (offset < num_playlist_tracks):
        playlist_obj = sp.user_playlist_tracks(user = user_id, playlist_id = playlist_id, \
                                               limit = limit, offset = offset)

        # add track information to running list
        for track_obj in playlist_obj['items']:
            playlist_tracks.append({
                'name': track_obj['track']['name'],
                'artists': ', '.join([artist['name'] for artist in track_obj['track']['artists']]),
                'track_id': track_obj['track']['id']
            })
            
        offset += limit
        
    return playlist_tracks

### function to get spotify audio features when given a list of track ids
def get_audio_features(track_ids):
    saved_tracks_audiofeat = [ ]
    
    # iterate through track_ids in groups of 50
    for ix in range(0,len(track_ids),50):
        audio_feats = sp.audio_features(track_ids[ix:ix+50])
        saved_tracks_audiofeat += audio_feats
        
    return saved_tracks_audiofeat

### function to  get all of the current user's playlists (playlist names, ids)
def get_all_user_playlists(playlist_limit = 50, playlist_offset = 0):
    # get initial list of users playlists (first n = playlist_limit), determine total number of playlists
    playlists_obj = sp.user_playlists(user_id, limit = playlist_limit, offset = playlist_offset)
    num_playlists = playlists_obj['total']

    # start accumulating playlist names and ids
    all_playlists = [{'name': playlist['name'], 'id': playlist['id']} for playlist in playlists_obj['items']]
    playlist_offset += playlist_limit

    # continue accumulating through all playlists
    while (playlist_offset < num_playlists):
        playlists_obj = sp.user_playlists(user_id, limit = playlist_limit, offset = playlist_offset)
        all_playlists += [{'name': playlist['name'], 'id': playlist['id']} for playlist in playlists_obj['items']]
        playlist_offset += playlist_limit
        
    return(all_playlists)

### function to create "tracks plus" df (including normalized audio features) when given a tracks df
def build_tracks_plus_df(tracks_df, normalize = True):
    # get raw audio features
    _audiofeat    = get_audio_features(track_ids = list(tracks_df['track_id']))
    _audiofeat_df = pd.DataFrame(_audiofeat).drop(['analysis_url', 'track_href', 'type', 'uri'], axis = 1)
    
    # scale audio features (if desired)
    if normalize:
        scaler = StandardScaler()
        audiofeat    = scaler.fit_transform(_audiofeat_df.drop(['id'], axis = 1))
        audiofeat_df = pd.DataFrame(audiofeat, columns = _audiofeat_df.drop('id', axis = 1).columns)
        audiofeat_df['id'] = _audiofeat_df['id']
    else:
        audiofeat_df = _audiofeat_df
    
    # merge audio features with tracks_df
    tracks_plus_df = tracks_df.merge(audiofeat_df, how = 'left', left_on = 'track_id', right_on = 'id')
    return(tracks_plus_df)

### function to cluster tracks based on normalized audio features
def cluster_tracks_plus_df(tracks_plus_df, num_clusters, drop_vars = None):
    kmeans = KMeans(n_clusters = num_clusters).fit(tracks_plus_df.drop(['track_id', 'id', 'name', 'artists'] + \
                                                                       (drop_vars if drop_vars != None else []), \
                                                                       axis = 1))
    tracks_plus_df['cluster'] = pd.Series(kmeans.labels_) + 1
    return(tracks_plus_df)

### function to save list of tracks (based ont track_ids) to a playlist
def save_cluster_tracks_to_playlist(playlist_name, track_ids):
    # get all of the users playlists
    all_playlists = get_all_user_playlists()
    
    # check if playlist already exists
    if (playlist_name not in [playlist['name'] for playlist in all_playlists]):
        playlist = sp.user_playlist_create(user = user_id, name = playlist_name, public = True)
    else:
        playlist_id = [playlist['id'] for playlist in all_playlists if playlist['name'] == playlist_name][0]
        playlist = sp.user_playlist(user = user_id, playlist_id = playlist_id)

    # remove any existing tracks in playlist
    while (playlist['tracks']['total'] > 0):
        sp.user_playlist_remove_all_occurrences_of_tracks(user_id, playlist['id'], \
                                                          tracks = [track['track']['id'] for track in \
                                                                    playlist['tracks']['items']])
        playlist = sp.user_playlist(user = user_id, playlist_id = playlist_id)

    # add tracks from cluster
    sp.user_playlist_add_tracks(user_id, playlist_id = playlist['id'], tracks = track_ids)

### pull in list of saved songs, create relevant df for songs, and then cluster
saved_tracks = get_saved_tracks()
saved_tracks_df = pd.DataFrame(saved_tracks)
saved_tracks_plus_df = build_tracks_plus_df(saved_tracks_df, True)
saved_tracks_clustered_df = cluster_tracks_plus_df(saved_tracks_plus_df, 100)

### save clusters
rand_clusters = np.random.choice(saved_tracks_clustered_df['cluster'].unique(), 3, False).tolist()

for i in range(1, len(rand_clusters) + 1):
    cluster = saved_tracks_clustered_df[saved_tracks_clustered_df['cluster'] == rand_clusters[i - 1]]
    print(cluster[["name", "artists", "cluster"]].head())
    save_cluster_tracks_to_playlist("k-means, cluster %d" % i, list(cluster['id']))