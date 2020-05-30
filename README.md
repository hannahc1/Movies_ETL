# Assumptions for the Module 8 Challenge

**Here are the 6 assumptions that went into this ETL process**

1. When filtering for only movies with directors and IMDB link and without No of episodes, an assumotion was made that No. of episodes indicates non-movie entries.
2. Duplicate IMDB id means the other columns are also duplicated.
3. The range has been replaced by just the upper limit: the assumption here is that the upper limit of the range represents the box office better than the lower limit.
4. Most date values fall within the following 4 patterns.
    - Full month name, one- to two-digit day, four-digit year (i.e., January 1, 2000)
    - Four-digit year, two-digit month, two-digit day, with any separator (i.e., 2000-01-01)
    - Full month name, four-digit year (i.e., January 2000)
    - Four-digit year
5. Most Running Time values fall in the following 2 patterns.
    - The hour + minute pattern
    - The xx minutes pattern
6. Dropping Adult movies will clear all corrupted data.
  
