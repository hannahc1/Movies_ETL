# Create a function that takes in three arguments:
# Wikipedia data
# Kaggle metadata
# MovieLens rating data (from Kaggle)
# Use the code from your Jupyter Notebook so that the function performs all of the transformation steps. Remove any exploratory data analysis and redundant code.
# Add the load steps from the Jupyter Notebook to the function. You’ll need to remove the existing data from SQL, but keep the empty tables.
# Check that the function works correctly on the current Wikipedia and Kaggle data.
# Document any assumptions that are being made. Use try-except blocks to account for unforeseen problems that may arise with new data.
### See ASSUMPTIONS 1 through 6 below.

# Create a function that takes in three arguments.
def movie_etl(wiki_json,kaggle_csv,ratings_csv):
    
    # Import dependencies
    import json
    import pandas as pd
    import numpy as np
    import re
    from sqlalchemy import create_engine
    from config import db_password
    import time
    
    ## 1. Extract step
    # Open the Wikipedia JSON file to be read into the variable file
    with open(wiki_json,mode='r') as file:
        wiki_movies_raw = json.load(file)

    # Open the Kaggle metadata and MovieLens rating data to be read into Pandas DataFrames
    kaggle_metadata = pd.read_csv(kaggle_csv,low_memory=False)
    ratings = pd.read_csv(ratings_csv)

    ## 2. Transformation step
    # -- Wikipedia data
    wiki_movies_df = pd.DataFrame(wiki_movies_raw)
    # Filter for only movies with directors and IMDB link and without No of episodes.
    wiki_movies = [movie for movie in wiki_movies_raw 
                if("Director" in movie or "Directed by" in movie) 
                    and "imdb_link" in movie
                    and "No. of episodes" not in movie]
    ### ASSUMPTION 1: No. of episodes indicates non-movie entries.
    wiki_movies_df = pd.DataFrame(wiki_movies)
    # Combine all of the alternate titles by language into one column.
    def clean_movie(movie):
        movie = dict(movie) # Create a non-destructive copy (Constructors = special functions that initialize new objects)
        alt_titles = {} # Make an empty dict to hold all of the alt titles
        for key in ['Also known as','Arabic','Cantonese','Chinese','French',
                    'Hangul','Hebrew','Hepburn','Japanese','Literally',
                    'Mandarin','McCune–Reischauer','Original title','Polish',
                    'Revised Romanization','Romanized','Russian',
                    'Simplified','Traditional','Yiddish']:
            if key in movie:# Check if the current key exists in the movie object
                alt_titles[key] = movie[key] # Remove the key-value pair and add to the alt titles dict
                movie.pop(key)
        if len(alt_titles) > 0: # After looping through every key, add the alt titles dict to the movie object
            movie["alt_titles"] = alt_titles
        
        # merge column names
        def change_column_name(old_name, new_name):
            if old_name in movie:
                movie[new_name] = movie.pop(old_name)
        change_column_name('Adaptation by', 'Writer(s)')
        change_column_name('Country of origin', 'Country')
        change_column_name('Directed by', 'Director')
        change_column_name('Distributed by', 'Distributor')
        change_column_name('Edited by', 'Editor(s)')
        change_column_name('Length', 'Running time')
        change_column_name('Original release', 'Release date')
        change_column_name('Music by', 'Composer(s)')
        change_column_name('Produced by', 'Producer(s)')
        change_column_name('Producer', 'Producer(s)')
        change_column_name('Productioncompanies ', 'Production company(s)')
        change_column_name('Productioncompany ', 'Production company(s)')
        change_column_name('Released', 'Release Date')
        change_column_name('Release Date', 'Release date')
        change_column_name('Screen story by', 'Writer(s)')
        change_column_name('Screenplay by', 'Writer(s)')
        change_column_name('Story by', 'Writer(s)')
        change_column_name('Theme music composer', 'Composer(s)')
        change_column_name('Written by', 'Writer(s)')
        
        return movie

    # Make a list of cleaned movies with a list comprehension
    clean_movies = [clean_movie(movie) for movie in wiki_movies]
    wiki_movies_df = pd.DataFrame(clean_movies)

    # Extract the IMDB id (str.extract() method)
    wiki_movies_df['imdb_id']=wiki_movies_df["imdb_link"].str.extract(r'(tt\d{7})')
    # Drop the duplicates (drop_duplicates() method)
    ### ASSUMPTION 2: Duplicate IMDB id means the other columns are also duplicated.
    wiki_movies_df.drop_duplicates(subset="imdb_id", inplace=True)

    # List of columns that have less than 90% null values to remove mostly null columns
    wiki_columns_to_keep = [column for column in wiki_movies_df.columns if wiki_movies_df[column].isnull().sum()<len(wiki_movies_df)*0.9]
    wiki_movies_df = wiki_movies_df[wiki_columns_to_keep]

    # Create a Box Office variable and drop missing values.
    box_office = wiki_movies_df["Box office"].dropna()
    # Form To match “$123.4 million/billion.”
    form_one = r"\$\s*\d+\.?\d*\s*[mb]illi?on"
    # Form To match $123.456.789 but not 1.234 billion
    form_two = "\$\s*\d{1,3}(?:[,\.]\d{3})+(?!\s[mb]illion)"
    # Remove values in range
    box_office = box_office.str.replace(r'\$.*[-—–](?![a-z])', '$', regex=True)
    ### ASSUMPTION 3: Upper limit of the range represents the box office better than lower limit.
    # Define a function to take in a string and return a floating-point number.
    def parse_dollars(s):
        # if s is not a string, return NaN
        if type(s) != str:
            return np.nan
        
        # if input is of the form $###.# million
        if re.match(r'\$\s*\d+\.?\d*\s*milli?on', s, flags=re.IGNORECASE):
            # remove dollar sign and " million"
            s = re.sub("\$|\s|[a-zA-Z]","",s)
            # convert to float and multiply by a million
            value = float(s) * 10**6
            # return value
            return value
        
        # if input is of the form $###.# billion
        elif re.match(r'\$\s*\d+\.?\d*\s*billi?on', s, flags=re.IGNORECASE):
            # remove dollar sign and " billion"
            s = re.sub("\$|\s|[a-zA-Z]","",s)
            # convert to float and multiply by a billion
            value = float(s) * 10**9        
            # return value
            return value
        
        # if input is of the form $###,###,### (or $###.###.###)
        elif re.match(r'\$\s*\d{1,3}(?:[,\.]\d{3})+(?!\s[mb]illion)', s, flags=re.IGNORECASE):
            
            # remove dollar sign and commas
            s = re.sub("\$|,","",s)
            # convert to float
            value = float(s)
            # return value
            return value
        
        # otherwise, return NaN
        else:
            return np.nan
    # Converting the box office values to numeric values (extract the values from box_office)
    wiki_movies_df["box_office"] = box_office.str.extract(f'({form_one}|{form_two})',flags=re.IGNORECASE)[0].apply(parse_dollars)
    # Drop the box office column
    wiki_movies_df.drop("Box office", axis=1, inplace=True)

    # Create a Budget variable and drop missing values.
    budget = wiki_movies_df["Budget"].dropna()
    # Convert lists to strings
    budget = budget.map(lambda x: " ".join(x) if type(x) == list else x)
    # Remove any values between a dollar sign and a hyphen (for budget in ranges)
    budget = budget.str.replace(r'\$.*[-—–](?![a-z])',"$",regex=True)
    # Converting the budget values to numeric values (extract the values from budget)
    wiki_movies_df["budget"] = budget.str.extract(f'({form_one}|{form_two})',flags=re.IGNORECASE)[0].apply(parse_dollars)
    # Drop the original budget column
    wiki_movies_df.drop("Budget", axis=1, inplace=True)

    # Make a variable to hold the non-null values of Release date, convert lists to strings
    release_date = wiki_movies_df["Release date"].dropna().apply(lambda x: " ".join(x) if type(x) == list else x)
    ### ASSUMPTION 4: Most date values fall within the following 4 patterns.
    # Full month name, one- to two-digit day, four-digit year (i.e., January 1, 2000)
    date_form_one=r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s[123]\d,\s\d{4}'
    # Four-digit year, two-digit month, two-digit day, with any separator (i.e., 2000-01-01)
    date_form_two=r'd{4}.[01]\d.[123]\d'
    # Full month name, four-digit year (i.e., January 2000)
    date_form_three=r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{4}'
    # Four-digit year
    date_form_four=r'\d{4}'
    # Extract the dates as strings and use Pandas built in method to_datetime
    wiki_movies_df['release_date'] = pd.to_datetime(release_date.str.extract(f'({date_form_one}|{date_form_two}|{date_form_three}|{date_form_four})')[0], infer_datetime_format=True)

    # Parse Running Time, drop null, list to strings
    running_time=wiki_movies_df["Running time"].dropna().apply(lambda x: " ".join(x) if type(x) == list else x)
    ### ASSUMPTION 5: Most Running Time values fall in the following 2 patterns.
    # Extract digits from the hour + minute patters and the xx minutes pattern.
    running_time_extract = running_time.str.extract(r'(\d+)\s*ho?u?r?s?\s*(\d*)|(\d+)\s*m')
    # Convert strings to numeric values and turn empty string into NaN using coerce argument then change them to zeros.
    running_time_extract = running_time_extract.apply(lambda col: pd.to_numeric(col, errors='coerce')).fillna(0)
    # Convert the hour capture groups and minute capture groups to minutes if the pure minutes capture group is zero, and save the output to DF.
    wiki_movies_df['running_time'] = running_time_extract.apply(lambda row: row[0]*60 + row[1] if row[2] == 0 else row[2], axis=1)
    # Drop Running time column from the dataset
    wiki_movies_df.drop('Running time', axis=1, inplace=True)

    # -- Kaggle Metadata
    # Remove the bad data
    ### ASSUMPTION 6: Dropping Adult movies will clear all corrupted data.
    kaggle_metadata = kaggle_metadata[kaggle_metadata["adult"] == "False"].drop("adult",axis="columns")
    # Convert data type to Boolean
    kaggle_metadata['video'] = kaggle_metadata['video'] == 'True'
    # Convert data type to Integer
    kaggle_metadata['budget'] = kaggle_metadata['budget'].astype(int)
    # Use the to_numeric() method from Panda to conver the data to numbers
    kaggle_metadata['id'] = pd.to_numeric(kaggle_metadata['id'], errors='raise')
    kaggle_metadata['popularity'] = pd.to_numeric(kaggle_metadata['popularity'], errors='raise')
    # Convert the date format
    kaggle_metadata['release_date'] = pd.to_datetime(kaggle_metadata['release_date'])

    # -- Ratings Data
    # Convert the timestamp column to a datetime data type so that it can be stored in a SQL table.
    ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')

    # -- Merge Wikipedia and Kaggle Metadata
    movies_df = pd.merge(wiki_movies_df, kaggle_metadata, on="imdb_id", suffixes=["_wiki","_kaggle"])
    # Drop a bad data (merged data)
    movies_df = movies_df.drop(movies_df[(movies_df['release_date_wiki'] > '1996-01-01') & (movies_df['release_date_kaggle'] < '1965-01-01')].index)
    # Drop Wikipedia data entirely for the following columns
    movies_df.drop(columns=['title_wiki','release_date_wiki','Language','Production company(s)'], inplace=True)
    # Define a function to fill in missing data for a column pair and drop the redundant column.
    def fill_missing_kaggle_data(df, kaggle_column, wiki_column):
        df[kaggle_column] = df.apply(
            lambda row: row[wiki_column] if row[kaggle_column] == 0 else row[kaggle_column]
            , axis=1)
        df.drop(columns=wiki_column, inplace=True)
    # Fill in missing data with Wikidata
    fill_missing_kaggle_data(movies_df, 'runtime', 'running_time')
    fill_missing_kaggle_data(movies_df, 'budget_kaggle', 'budget_wiki')
    fill_missing_kaggle_data(movies_df, 'revenue', 'box_office')
    # Reorder the columns.
    movies_df = movies_df.loc[:, ['imdb_id','id','title_kaggle','original_title','tagline','belongs_to_collection','url','imdb_link',
                        'runtime','budget_kaggle','revenue','release_date_kaggle','popularity','vote_average','vote_count',
                        'genres','original_language','overview','spoken_languages','Country',
                        'production_companies','production_countries','Distributor',
                        'Producer(s)','Director','Starring','Cinematography','Editor(s)','Writer(s)','Composer(s)','Based on'
                        ]]
    # Rename the columns.
    movies_df.rename({'id':'kaggle_id',
                    'title_kaggle':'title',
                    'url':'wikipedia_url',
                    'budget_kaggle':'budget',
                    'release_date_kaggle':'release_date',
                    'Country':'country',
                    'Distributor':'distributor',
                    'Producer(s)':'producers',
                    'Director':'director',
                    'Starring':'starring',
                    'Cinematography':'cinematography',
                    'Editor(s)':'editors',
                    'Writer(s)':'writers',
                    'Composer(s)':'composers',
                    'Based on':'based_on'
                    }, axis='columns', inplace=True)
    # Use gourpby on "movieID" and "rating" columns to take count for each group of Ratings data 
    # Then pivot the data to make ratings values to be the columns and counts for each rating values to be the rows.
    rating_counts = ratings.groupby(['movieId','rating'], as_index=False).count() \
                            .rename({'userId':'count'}, axis=1) \
                            .pivot(index='movieId',columns='rating', values='count')
    # Rename the rating columns
    rating_counts.columns = ['rating_' + str(col) for col in rating_counts.columns]
    # Merge the rating counts into movies_df.
    movies_with_ratings_df = pd.merge(movies_df, rating_counts, left_on='kaggle_id', right_index=True, how='left')

    ## 3. Load step
    # Load movies_df to database.
    db_string = f"postgres://postgres:{db_password}@127.0.0.1:5432/movie_data"
    engine = create_engine(db_string)
    # Update the table with the new data.
    movies_df.to_sql(name='movies', con=engine, if_exists='replace')
    
    # Load ratings data to database.
    rows_imported = 0
    # get the start_time from time.time()
    start_time = time.time()
    for data in pd.read_csv(ratings_csv, chunksize=1000000):
        print(f'importing rows {rows_imported} to {rows_imported + len(data)}...', end='')
        data.to_sql(name='ratings', con=engine, if_exists='append')
        rows_imported += len(data)

        # add elapsed time to final print out
        print(f'Done. {time.time() - start_time} total seconds elapsed')

# Perform the movie_etl function
movie_etl('C:/users/hannah/class/Movies_ETL/wikipedia.movies.json',
    'C:/users/hannah/class/Movies_ETL/movies_metadata.csv',
    'C:/users/hannah/class/Movies_ETL/ratings.csv')