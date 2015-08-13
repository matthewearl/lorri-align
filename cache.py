#!/usr/bin/python

# Copyright (c) 2015 Matthew Earl
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
#     The above copyright notice and this permission notice shall be included
#     in all copies or substantial portions of the Software.
# 
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#     OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
#     NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#     OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
#     USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
Module for downloading and caching LORRI images.

"""

__all__ = (
    'update_metadata',
    'load_metadata',
    'check_images',
    'IMG_FORMAT',
    'MissingImage',
    'NoMetadataFile',
)

import calendar
import glob
import json
import os
import re
import time
import urllib2

_THUMBNAIL_URL_FORMAT = ("http://pluto.jhuapl.edu/soc/Pluto-Encounter/"
                         "index.php?page={}")
_IMG_URL_PREFIX = 'http://pluto.jhuapl.edu/soc/Pluto-Encounter/'
_TIMESTAMP_FORMAT = "%Y-%m-%d<br>%H:%M:%S %Z"
_IMG_PATH = "data/images/input/"
IMG_FORMAT = _IMG_PATH + "%Y-%m-%d_%H%M%S_%Z.jpg"
_METADATA_FILE = _IMG_PATH + "metadata.json"

FETCH_SLEEP = 1.0   # Avoid spamming the server!
MAX_FETCHES = 1000  # Should never need more than this number of HTTP
                    # requests.
num_fetches = 0

def _parse_line(line):
    l = []
    d = {}
    for cmd in line.split(";"):
        if cmd.startswith("thumbArr.push"):
            d["url"] = _IMG_URL_PREFIX + cmd.split('"')[1].replace(
                                                            "thumbnails/", "")
        elif cmd.startswith("UTCArr.push"):
            d["timestamp"] = calendar.timegm(time.strptime(cmd.split('"')[1],
                                             _TIMESTAMP_FORMAT))
            d["image_path"] = time.strftime(IMG_FORMAT,
                                            time.gmtime(d["timestamp"]))
        elif cmd.startswith("ExpArr.push"):
            d["exposure"] = cmd.split('"')[1]
            l.append(d)
            d = {}

    return l

class _InvalidPageNum(Exception):
    pass

def _urlopen(url):
    global num_fetches
    num_fetches += 1

    assert num_fetches < MAX_FETCHES, ("Too many HTTP requests ({}) "
                                             "attempted!".format(num_fetches))
    f = urllib2.urlopen(url)
    time.sleep(FETCH_SLEEP)

    return f

def _get_data_for_page(page_num):
    print "Fetching page {}".format(page_num)
    f = _urlopen(_THUMBNAIL_URL_FORMAT.format(page_num))

    for line in f.readlines():
        if line.startswith("StatusArr.push"):
            return _parse_line(line)

    raise _InvalidPageNum

def _metadata_gen():
    page_num = 1
    while True:
        try:
            for d in _get_data_for_page(page_num):
                yield d
        except _InvalidPageNum:
            return

        page_num += 1

class NoMetadataFile(Exception):
    pass

def load_metadata():
    """Load and return the meta-data."""
    if not os.path.exists(_METADATA_FILE):
        raise NoMetadataFile("Try running with -u?")

    with open(_METADATA_FILE, 'r') as f:
        return json.load(f)

def update_metadata():
    """Download any missing meta-data from the New Horizon's website."""

    try:
        old_metadata = load_metadata()
    except NoMetadataFile:
        old_metadata = []

    updates = []
    for d in _metadata_gen():
        if (len(old_metadata) > 0 and
            d['timestamp'] == old_metadata[0]['timestamp']):
            break
        updates.append(d)

    print "Downloaded new metadata for {} files".format(len(updates))

    with open(_METADATA_FILE, 'w') as f:
        json.dump(updates + old_metadata, f)

def _download_image(d):
    print "Downloading {} to {}".format(d['url'], d["image_path"])
    time.sleep(FETCH_SLEEP)
    in_f = _urlopen(d['url'])
    with open(d["image_path"], 'w') as out_f:
        out_f.write(in_f.read())

class MissingImage(Exception):
    pass

def check_images(metadata, download_missing=False):
    """
    Check images for the provided metadata have been downloaded.

    If the `download_missing` argument is False, `MissingImage` is raised for
    any missing images.

    """
    for d in metadata:
        if not os.path.exists(d["image_path"]):
            if download_missing:
                _download_image(d)
            else:
                raise MissingImage("Image {} has not been downloaded".format(
                                                             d["image_path"]))

