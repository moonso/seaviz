# Seaviz

Cli to test some stuff from the [seaborn tutorial][seaborn_tut].
Some of the functions are more developed than others atm...

The idea here was to write small functions that makes it easy to test different layouts and settings for some of the plot types.

For example to check how different colors looks like one can run:

`seaviz --palette hls palplot --saturation 0.8 --lightness 0.3 -n 8`

![alt text](https://raw.githubusercontent.com/moonso/seaviz/master/seaviz/img/hls_palette.png)

Or try out the darkgrid style:

`seaviz --style darkgrid boxplot`

![alt text](https://raw.githubusercontent.com/moonso/seaviz/master/seaviz/img/box_test.png)


## Installation

In you favourite environment:

```
git clone https://github.com/moonso/seaviz
cd seaviz
python setup.py install
```

or `pip install seaviz`

## Run

```
seaviz --help
```





[seaborn_tut]: https://seaborn.pydata.org/tutorial.html