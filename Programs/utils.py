import pickle
import glob
import json
import numpy as np
import pandas as pd
import re
import tensorflow as tf

def importAddlPlaylistData():
    '''
    Import playlists from external data source
    '''

    df_playlists = None

    for file in glob.glob('Data/mpd*.json'):
        with open(file) as f:
            js = f.read()
            json_slice = json.loads(js)
            df_slice = pd.DataFrame.from_dict(json_slice['playlists'], orient='columns')

            if df_playlists is None:
                df_playlists = df_slice
            else:
                df_playlists = pd.concat([df_playlists, df_slice], ignore_index=True)

    def extractArtists(track_list):

        playlist_artist_dict = dict()

        for track in track_list:
            playlist_artist_dict[track['artist_name']] = playlist_artist_dict.get(track['artist_name'], 0) + 1

        return playlist_artist_dict

    df_playlists['artists'] = df_playlists['tracks'].apply(lambda x: extractArtists(x))
    df_playlists['artists_dict'] = df_playlists.apply(lambda x: {x['name']: x['artists']}, axis=1)

    return list(df_playlists['artists_dict'])



def importSpotifyData():
    '''
    Import data and create tokenized dictionary
    '''

    with open('Data/spotify_playlist_artists.pickle', 'rb') as f:
        playlistArtists = pickle.load(f)

    playlistArtists['Addl'] = importAddlPlaylistData()

    artists = set()
    for _, p_list in playlistArtists.items():
        for p in p_list:
            for _, a_dict in p.items():
                for a, _ in a_dict.items():
                    artists.add(a)

    # Create tokenized dictionary of artists
    artists_dict = {v:k for k, v in enumerate(sorted(list(artists)))}
    artists_dict_rev = {k:v for k, v in enumerate(sorted(list(artists)))}

    # Remove users from the playlistArtists dictionary
    playlistArtists_full = []

    for key, v in playlistArtists.items():
        playlistArtists_full = playlistArtists_full + v

    return playlistArtists_full, artists_dict, artists_dict_rev



def tokenizeArtists(playlistArtists, artists_dict):
    '''
    Input: dictionary of playlist name: set of artists
    Output: matrix of encoded representation of artists in each playlist
    '''

    X = np.zeros((len(playlistArtists), len(artists_dict)))
    y = []
    for pos, playlist in enumerate(playlistArtists):
        for playlist_name, artists in playlist.items():
            y.append(playlist_name.lower())
            for artist, count in artists.items():
                X[pos][artists_dict[artist]]=count

    return X, y


def tokenizeCharacters(playlist_names, num_characters=25):
    '''
    Tokenize characters in playlist names and return list of encoding
    '''


    characters = set()
    for name in playlist_names:
        for char in name:
            characters.add(char)

    char_dict = {v:k+3 for k, v in enumerate(sorted(list(characters)))}
    char_dict_rev = {k+3:v for k, v in enumerate(sorted(list(characters)))}

    char_dict_rev[0]='<PAD>'
    char_dict_rev[1]='<START>'
    char_dict_rev[2]='<END>'

    char_dict['<PAD>'] = 0
    char_dict['<START>'] = 1
    char_dict['<END>'] = 2


    y = np.zeros((len(playlist_names),num_characters), dtype=np.int)
    for playlist_pos, name in enumerate(playlist_names):
        y[playlist_pos][0] = char_dict['<START>']

        for char_pos, char in enumerate(name):
            if char_pos == num_characters-2:
                break

            y[playlist_pos][char_pos+1]=char_dict[char]

        ### Add end
        if len(name) >= num_characters-1:
            y[playlist_pos][num_characters-1] = char_dict['<END>']
        else:
            y[playlist_pos][len(name)+1] = char_dict['<END>']

    return y, char_dict, char_dict_rev


def preprocess_sentence(w):
    '''
    Split strings by non-alpha numerics
    '''
    w = re.sub(r"([^a-zA-Z0-9])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    return w.split()



def tokenizeWords(playlist_names, num_words=10):
    '''
    Tokenize characters in playlist names and return list of encoding
    '''
    words = set()
    for name in playlist_names:
        for word in preprocess_sentence(name):
            words.add(word)

    word_dict = {v: k + 3 for k, v in enumerate(sorted(list(words)))}
    word_dict_rev = {k + 3: v for k, v in enumerate(sorted(list(words)))}

    word_dict_rev[0] = '<PAD>'
    word_dict_rev[1] = '<START>'
    word_dict_rev[2] = '<END>'

    word_dict['<PAD>'] = 0
    word_dict['<START>'] = 1
    word_dict['<END>'] = 2

    y = np.zeros((len(playlist_names), num_words), dtype=np.int)
    for playlist_pos, name in enumerate(playlist_names):

        split_name = preprocess_sentence(name)

        y[playlist_pos][0] = word_dict['<START>']

        for word_pos, word in enumerate(split_name):
            if word_pos == num_words - 2:
                break

            y[playlist_pos][word_pos + 1] = word_dict[word]

        ### Add end
        if len(split_name) >= num_words - 1:
            y[playlist_pos][num_words - 1] = word_dict['<END>']
        else:
            y[playlist_pos][len(split_name) + 1] = word_dict['<END>']

    return y, word_dict, word_dict_rev


def generateName(encoder, decoder, artist_input, artists2indx, output_dict):
    '''
    Generate a name from the encoder-decoder website
    '''
    hidden = decoder.reset_state(batch_size=1)

    input_vec = tokenize(artist_input, artists2indx)

    prediction_encoded = encoder(input_vec)
    dec_input = tf.expand_dims([0], 0)

    result = []

    for i in range(10):
        predictions, hidden = decoder(dec_input, prediction_encoded, hidden)

        predicted_id = tf.argmax(predictions[0]).numpy()
        result.append(output_dict[predicted_id])

        if output_dict[predicted_id] == '<END>':
            print("predicted playlist name:    {}".format(''.join(result[:-1])))
            return

        dec_input = tf.expand_dims([predicted_id], 0)

    print("predicted playlist name:    {}".format(''.join(result[:-1])))
    return


def tokenize(artist_input, artists2indx):
    '''
    Tokenize artist inputs for a single playlist
    '''
    X_test = np.zeros((1, len(artists2indx)))

    for artist, count in artist_input.items():
        try:
            X_test[0][artists2indx[artist]] = count
        except:
            print("{} not in dictionary".format(artist))
    return tf.convert_to_tensor(X_test)


