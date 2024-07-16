# Seasonal Mortality Monitor

This is the code of the dashboard called Seasonal Mortality Monitor, available on the [InterSCity website](http://seasonality.interscity.org/). The development of the dashboard was part of my (Victor Foscarini Almeida) master's degree research project, supervised by Fabio Kon and co-supervised by Raphael Camargo.

The dashboard aims to complement my main research [seasonal-mortality-hub](https://github.com/Victor-Foscarini/seasonal-mortality-hub), providing users with customized insights into mortality patterns. It allows users to tailor the visualizations to specific cities, encompassing all Brazil's capitals and causes of death according to ICD-10 codes.

We used Python's libraries, Dash and Plotly, for the foundation of the dashboard, complemented with some basic data science libraries for data processing and modeling: Pandas, Numpy, and Statsmodels.

# How to Run

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy `mortality_monitor.service` to `/etc/systemd/system`
and run (as root) `systemctl enable mortality_monitor`.
