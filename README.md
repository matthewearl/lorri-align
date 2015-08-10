# New Horizons LORRI image aligner

This is a script which takes images from [New Horizon's](https://en.wikipedia.org/wiki/New_Horizons)
LORRI camera, and aligns them using background stars. With these aligned images
a flyby animation can be produced:

![Flyby animation](http://matthewearl.github.io/assets/lorri-align/anim.gif)

The above is generated with:

    ./lorri-align.py --from '2015-04-12 03:27:00' --to '2015-07-09 22:37:05' \
        --crop 555,343,631,1035  
    convert -delay 5 data/images/stacked/* anim.gif

*Note* pass `-u` and `-d` on the initial invocation in order to grab the
relevant metadata and imagery from the [John Hopkins LORRI website](http://pluto.jhuapl.edu/soc/Pluto-Encounter/index.php).
A generous `time.sleep` is inserted between HTTP requests to avoid DoS'ing the
website, but please be considerate when running with `-d` and `-u`!

If the metadata becomes corrupted for whatever reason deleting
`data/images/input/metadata.json` and re-running with `-u` should restore the
metadata. Similarly any corrupt images can be deleted from
`data/images/input/`. They will be restored the next time the script needs
them.
