This file contains descriptions for all files and folders for this project

The only files that should be run are clustering.ipynb or large_clustering.ipynb, as all other files were only used to gather data, and calculate the artist embeddings.
I am not providing an environment with relevant packages as the only packages used in the two clustering files are conda default.

/artists: Folder containing edited artist lists scraped from wikipedia
/complete_artists: Folder containing un-edited artist lists scraped from wikipedia
/lyrics: Folder containing .txt files that contain lyrics for 2000 of the artists in /complete_artists

/scrapers: Folder containing webscraping notebooks and Useragents list. Do not run these files, they were only used to gather data, and are not meant
to be ran more than once.

500_artist_embeddings.json and 2000_artist_embeddings.json: store artist embeddings for all 2000 artists that have lyric data

clustering.ipynb: First clustering/similarity trial using 500_artist_embeddings.json
large_clustering.ipynb: Second clustering/similarity trial using 2000_artist_embeddings.json

embeddings_trial.ipynb: Notebook used to practive creating an artist embedding for a single artist
create_embeddings: Notebook used to create artist embeddings for all 2000 artists, do not run this, it takes a really really long time.

explore.ipynb: Used to explore/edit lyricsdata.csv

lyricsdata.csv: original dataset collected by webscrapers
finaldata.csv: edited version of lyricsdata.csv

writeup.pdf: final writeup for this project

