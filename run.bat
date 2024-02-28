@echo off
SETLOCAL EnableDelayedExpansion

REM Output file name
SET output_file=all_outputs.txt

REM Make sure the output file is empty
IF EXIST %output_file% DEL %output_file%

REM Define the first set of numbers
SET first_numbers=100 1000 10000
REM Define the second set of numbers
SET second_numbers=0.1 0.5 0.9

REM Loop through each combination of numbers
FOR %%F IN (%first_numbers%) DO (
    FOR %%S IN (%second_numbers%) DO (
        REM Run each combination 5 times
        FOR /L %%I IN (1,1,5) DO (
            ECHO Running with %%F and %%S, iteration %%I >> %output_file%
            python cs581_P01_A20527707.py campus.csv 2 %%F %%S >> %output_file%
        )
    )
)

ENDLOCAL