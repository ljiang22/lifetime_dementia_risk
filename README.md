# Submission for Coding Challenge of Insight Data Engineering - Lian Jiang, 2020

## Table of Contents
1. [Introduction](README.md#introduction)
1. [Input dataset](README.md#input-dataset)
1. [Methods](README.md#methods)
1. [System requirements](README.md#system-requirements)
1. [Run instruction](README.md#run-instruction)
1. [Questions?](README.md#questions?)

## Introduction
The goal of this work is to finish the coding task provided by Insight data engineering team. The problem that needs to be solved is to calculate the total number of times vehicles, equipment, passengers and pedestrians cross the U.S.-Canadian and U.S.-Mexican borders each month and the running monthly average of total number of crossings for that type of crossing and border. 

## Input dataset
The dataset used in this task is border crossing entry data collected from the bureau of transportation statistics (see more detail using the link: https://data.transportation.gov/Research-and-Statistics/Border-Crossing-Entry-Data/keg4-3bc2). 

The input file used in this work is `Border_Crossing_Entry_Data.csv`, residing in the top-most `input` directory of the repository. The file contains data of the form:

```
Port Name,State,Port Code,Border,Date,Measure,Value,Location
Derby Line,Vermont,209,US-Canada Border,03/01/2019 12:00:00 AM,Truck Containers Full,6483,POINT (-72.09944 45.005)
Norton,Vermont,211,US-Canada Border,03/01/2019 12:00:00 AM,Trains,19,POINT (-71.79528000000002 45.01)
Calexico,California,2503,US-Mexico Border,03/01/2019 12:00:00 AM,Pedestrians,346158,POINT (-115.49806000000001 32.67889)
Hidalgo,Texas,2305,US-Mexico Border,02/01/2019 12:00:00 AM,Pedestrians,156891,POINT (-98.26278 26.1)
Frontier,Washington,3020,US-Canada Border,02/01/2019 12:00:00 AM,Truck Containers Empty,1319,POINT (-117.78134000000001 48.910160000000005)
Presidio,Texas,2403,US-Mexico Border,02/01/2019 12:00:00 AM,Pedestrians,15272,POINT (-104.37167 29.56056)
Eagle Pass,Texas,2303,US-Mexico Border,01/01/2019 12:00:00 AM,Pedestrians,56810,POINT (-100.49917 28.70889)
```

The explanation for the fields used in this task is given as follows:
* `Border`: Designates what border was crossed
* `Date`: Timestamp indicating month and year of crossing
* `Measure`: Indicates means, or type, of crossing being measured (e.g., vehicle, equipment, passenger or pedestrian)
* `Value`: Number of crossings

## Methods
The procedure I use to calculate the total number of crossings of each type of measure, that crossed the border that month and the running monthly average of total crossings are:
1) Find all the dates recorded in the dataset;
2) Remove the duplicative dates and sort them from oldest to latest;
3) Repeat the same steps above on measure and border;
4) Loop each record in the dataset for each date, measure, and border;
5) Sum the number of crossings for all the records in which the date, measure, and borer match that in one loop;
6) Loop each of the total number of crossings calculated in step 5) for each date, measure, and border;
7) Calculate the average value for all the total number of crossings with the date earlier than the date in that specific loop and with the measure and border match in that loop;
8) Set the average value equals to 0 if there was no record with the date earlier. 

## System requirements
* Both Linux and Windows are supported.
* 64-bit Python 3.7 installation.
* The code is written using only the default python data structures.

## Run instruction
Copy the input dataset to the `input` folder with the format described in `Input dataset` section. Edit the `run.sh` file if you have different input file name and want to set different output file name. The example used in my work is as follows: 
`python3.7 ./src/border_analytics.py ./input/Border_Crossing_Entry_Data.csv ./output/report.csv`. After finish the editing, run the script by `./run.sh`, then the code will finish the calculation and write the result to the output file. 

One can run the test with command `insight_testsuite~$ ./run_tests.sh` from within the `insight_testsuite` folder. There are two tests in the `insight_testsuite/tests` folder. Both of them had passed the test on my code.

## Questions?
For any questions, concerns, and comments, please contact Lian Jiang at jiang2015leon@gmail.com
