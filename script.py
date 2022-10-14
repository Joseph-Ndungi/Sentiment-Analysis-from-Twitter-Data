import csv
from typing import List

dataset = []
with open("Variables3.csv") as file:
    dataset = list(csv.reader(file))
columns = dataset[0]
data = dataset[1:]


class Column:
    """ Class for indices of columns"""
    Location = 4
    Text = 1
    Protected = 3
    Folowers = 5
    VerifiedTweet = 6
    Verified = 2
    Source = 7


def source_cleanup()->None:
    """
    Reduces the number of devices used to web, android,and iphone and the rest to other
    """
    sources = ["Twitter for Android", "Twitter Web App", "Twitter for iPhone"]
    for row in data:
        if row[Column.Source] not in sources:
            row[Column.Source] = "other"


def location_cleanup()->None:
    """
    onverts the location data into numerical values
    If there is no location value is converted to 0 and 1 otherwise
    """
    for row in data:
        if not row[Column.Location]:
            row[Column.Location] = "NO"
        else:
            row[Column.Location] = "YES"


def followers_cleanup()->None:
    """
    IF number of followers is greater than 1000 it assigns it to 1 and zero otherwise
    """
    for row in data:
        if int(row[Column.Folowers]) > 1000:
            row[Column.Folowers] = 1
        else:
            row[Column.Folowers] = 0


def merge_back_data()->None:
    """Mergees back the titles of the data"""
    source_cleanup()
    followers_cleanup()
    location_cleanup()
    add_number_tweet_chars_column()
    data.insert(0, columns)


def add_number_tweet_chars_column()->None:
    """
    Add the a column of number of characters in a tweet
    """
    columns.append("NumTweetChars")
    for row in data:
        row.append(len(row[Column.Text]))


def write_to_file(filename: str)->None:
    """Writes data to a file"""
    merge_back_data()
    with open(filename, mode="w") as file:
        data_writer = csv.writer(file, delimiter=',')
        for row in data:
            data_writer.writerow(row)


write_to_file("new_cleaned_data.csv")