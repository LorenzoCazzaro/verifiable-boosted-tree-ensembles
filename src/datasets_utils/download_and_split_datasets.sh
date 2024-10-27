#!/bin/bash

#be sure to execute it in the datasets_utils folder!

python3 split_mnist.py 2 6

python3 split_fmnist.py 0 3

python3 split_webspam.py

python3 split_twitter_spam_accounts.py

python3 split_twitter_spam_urls.py