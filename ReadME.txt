# Reddit Comment Analysis Script

## Overview
This Python script analyzes comments from Reddit threads, performing tasks such as topic modeling, sentiment analysis, and visualization.

## Prerequisites
Make sure you have Python installed on your machine.

## Installation
Install the required libraries using the following command:
```bash
pip install pandas praw bertopic textblob nltk matplotlib wordcloud flair gensim networkx



## Important 
Reddit API Credentials:
- You need to have a Reddit account and create a Reddit App to obtain the client_id, client_secret, and user_agent.
- Update the reddit_client_id, reddit_client_secret, and reddit_user_agent variables in the script with your Reddit App credentials.



## Notes

1. The script fetches comments from specified Reddit threads using the provided URLs. Ensure the URLs are valid and accessible.
2. Customize the script as needed, adjusting parameters such as the number of topics in topic modeling.
3. For security, consider handling Reddit API credentials securely, using environment variables or a configuration file.
4. Make sure to follow ethical guidelines and Reddit's API usage policies.
