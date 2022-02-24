
# Segunda Division Draw Analysis

Large project aimed at analyzing the Spanish 2nd football league in terms of draw results. The main goal is to create a bot that predicts matches results.


## Table of content

* [Repository content](#Repository-content)
* [Tech Stack And Requirements](#Tech-Stack-And-Requirements)
* [Draw Analysis Python Package](#Draw-Analysis-Python-Package)
* [How to use](#How-to-use)
* [Next features and tasks](#Next-features-and-tasks)

## Repository content
- 1_extract_transform_analyse
- 2_ms_sql
- 3_external_variables
- draw_analysis (python package)
- README.md

## Tech Stack And Requirements
- WIN10
- Data source: RapidAPI/ API – football, Web scraping
- Python 3.8 (OOP, sys, os, datetime, pandas, numpy, matplotlib, plotly, seaborn, pyenchant, 	SQLalchemy, pyodbc, munkres, bs4, requests)
- MS SQL Server
- ETL (SSIS, SQL Server Agent)
- Power BI
- draw_analysis (python package)
- IDE: Pycharm, Jupyter Notebook


## Draw Analysis Python Package
Python package provides tools for download and analyse football data.

**draw_analysis** is fully adopted to use Api-Football-Beta.
In order to get data, 'api headers' are required.
For more information please visit:

    https://rapidapi.com/
    https://rapidapi.com/api-sports/api/api-football-beta/

**draw_analysis** allows to analyze all available football leagues in terms of draw results.
Provides calculations and visualizations of the most important draw scores indicators.

### Modules
----------
    get_api: search and download fixtures data from API-Football-Beta.
    draw_analysis: general analysis for league data.
    team_draw_analysis: draw results performance for teams.
    visual: tools for visualization.
## How to use
Make sure you meet the requirements from section [Tech Stack And Requirements](Tech-Stack-And-Requirements).
Just download repository and run with IDE.

## Next features and tasks

- draw_analysis code refactoring
- data pipeline (SSIS, MS SQL Server, SQL Server Agent)
- staging , data warehouse of historical data and real time performance data
- data pipeline using Apache Airflow
