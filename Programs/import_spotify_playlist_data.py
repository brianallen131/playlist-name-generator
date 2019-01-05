import json
import pickle
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy

############################################
### IMPORT SPOTIFY DEVELOPER CREDENTIALS ###
############################################

with open('config.json') as f:
    conf = json.load(f)

#####################################
### SPOTIFY DEVELOPER CREDENTIALS ###
#####################################

client_credentials_manager = SpotifyClientCredentials(client_id=conf['SPOTIPY_CLIENT_ID'],
                                                      client_secret=conf['SPOTIPY_CLIENT_SECRET'])
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

############################################
### PULL PLAYLIST-ARTIST DATA FROM USERS ###
############################################

users = ['Spotify',
         'Digster',
         'Filter']


def userPlaylists(user):
    '''
    :param: Spotify user name
    :return: List of user's public playlists and its metadata
    '''

    items = []
    i=0
    while True:
        playlists = sp.user_playlists(user, offset=i*50)
        if playlists['items'] == []:
            return items

        items = items + playlists['items']
        print("{} playlists: {}".format(user,len(items)))
        i+=1

def pullPlaylists(users=users):
    '''
    Loop over users and call for all playlist metadata
    '''

    playlists = {}
    for user in users:
        playlists[user] = userPlaylists(user)

    return playlists

playlists = pullPlaylists()


def userPlaylistArtists(user, playlists):
    '''
    Store the distinct artists for each playlist
    '''

    playlist_artist = {}
    for playlist in playlists:
        playlist_artist_set = set()
        try:
            tracks = sp.user_playlist_tracks(user, playlist['uri'])
            for track in tracks['items']:
                track_artists = set([artists['name'] for artists in track['track']['artists']])
                playlist_artist_set = playlist_artist_set.union(track_artists)

            print("{} adds {}".format(playlist['name'], playlist_artist_set))
            playlist_artist[playlist['name']] = playlist_artist_set
        except:
            print("{} does not have tracks available".format(playlist['name']))

    return playlist_artist


def pullPlaylistArtists(users=users):
    '''
    Loop over users and call for all playlist metadata
    '''

    playlistArtists = {}
    for user in users:
        playlistArtists[user] = userPlaylistArtists(user, playlists[user])

    return playlistArtists

playlistArtists = pullPlaylistArtists()


#############################################
### STORE PLAYLIST RESULTS TO PICKLE FILE ###
#############################################

with open('Data/spotify_playlist_artists.pickle', 'wb') as handle:
    pickle.dump(playlistArtists, handle, protocol=pickle.HIGHEST_PROTOCOL)


