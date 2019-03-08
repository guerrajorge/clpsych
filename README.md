# Computational Linguistics and Clinical Psychology Workshop (CLPSYCH) 2019


This repository contains code used for the http://clpsych.org/shared-task-2019-2/

## Notebooks  

subreddit_download Notebook:  

Contains code to download subreddits from http://files.pushshift.io/reddit/subreddits/  

After dowloading the files in order to decrompress the zst file:  

git clone https://github.com/facebook/zstd.git  
make  
zstd -xvf Reddit_Subreddits.ndjson.zst 
