#To-Do:
##add total count instead of just percentage
##fix the problem of deleting a code of a higher level that contains a chosen code of a lower level
##fix problem of having to process data in every new callback (either with Store or with DataManager), it has a delay if i dont
##show only year in deaths by year and not Jan-Year
##add all-Brazil option

#plotting
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
#data processing
import pandas as pd
import numpy as np
#dash
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
#modelling time series
from statsmodels.tsa.seasonal import seasonal_decompose

data_path = 'data/'
# n_causes = 50 #maximum number of causes to show on pie plots

capital_to_uf = {'Rio Branco - AC': 'AC', 'Maceió - AL': 'AL', 'Manaus - AM': 'AM', 'Macapá - AP': 'AP',
                     'Salvador - BA': 'BA', 'Fortaleza - CE': 'CE', 'Brasília - DF': 'DF', 'Vitória - ES': 'ES',
                     'Goiânia - GO': 'GO', 'São Luís - MA': 'MA', 'Belo Horizonte - MG': 'MG',
                     'Campo Grande - MS': 'MS', 'Cuiabá - MT': 'MT', 'Belém - PA': 'PA', 'João Pessoa - PB': 'PB',
                     'Recife - PE': 'PE', 'Teresina - PI': 'PI', 'Curitiba - PR': 'PR', 'Rio de Janeiro - RJ': 'RJ',
                     'Natal - RN': 'RN', 'Porto Velho - RO': 'RO', 'Boa Vista - RR': 'RR', 'Porto Alegre - RS': 'RS',
                     'Florianópolis - SC': 'SC', 'Aracaju - SE': 'SE', 'São Paulo - SP': 'SP', 'Palmas - TO': 'TO'}

#Selections are of the shape {'I21: Acute myocardyal infarction':'I21'}
icd = pd.read_csv(f'{data_path}icd_codes.csv')
selection_3lvl = list(icd['code_3lvl']+': '+icd['name_3lvl'])
selection_3lvl = {x:x[:3] for x in selection_3lvl}
selection_2lvl = list(icd['code_2lvl'].unique()+': '+icd['name_2lvl'].unique())
selection_2lvl = {x:x[:7] for x in selection_2lvl}
selection_1lvl = list(icd['code_1lvl'].unique()+': '+icd['name_1lvl'].unique())
selection_1lvl = {x:x[:7] for x in selection_1lvl}


selection_age = {'(0, 5]': pd.Interval(0, 5, closed='right'), '(5, 10]': pd.Interval(5, 10, closed='right'),
                  '(10, 15]': pd.Interval(10, 15, closed='right'), '(15, 20]': pd.Interval(15, 20, closed='right'),
                  '(20, 25]': pd.Interval(20, 25, closed='right'), '(25, 30]': pd.Interval(25, 30, closed='right'),
                  '(30, 35]': pd.Interval(30, 35, closed='right'), '(35, 40]': pd.Interval(35, 40, closed='right'),
                  '(40, 45]': pd.Interval(40, 45, closed='right'), '(45, 50]': pd.Interval(45, 50, closed='right'),
                  '(50, 55]': pd.Interval(50, 55, closed='right'), '(55, 60]': pd.Interval(55, 60, closed='right'),
                  '(60, 65]': pd.Interval(60, 65, closed='right'), '(65, 70]': pd.Interval(65, 70, closed='right'),
                  '(70, 75]': pd.Interval(70, 75, closed='right'), '(75, 80]': pd.Interval(75, 80, closed='right'),
                  '(80, 90]': pd.Interval(80, 90, closed='right'), '(90, 100]': pd.Interval(90, 100, closed='right'),
                  '(100, 120]': pd.Interval(100, 120, closed='right')}

class DataManager:

    def __init__(self):
    #initialize dashboard in SÃO PAULO
        self.df = pd.read_csv(f'{data_path}datasus/SP_standardized_1996_2019.csv', index_col=0)
        self.df['DTOBITO'] = pd.to_datetime(self.df.DTOBITO)
        self.df['IDADE'] = self.df['IDADE'].apply(lambda x: pd.Interval(*map(int, x.strip('[]()').split(','))))
        self.data = self.df.copy()
        self.city = 'São Paulo - SP'
        self.icd = icd
        self.keys_selected_causes_1lvl = list(selection_1lvl.keys())
        self.keys_selected_causes_2lvl = list(selection_2lvl.keys())
        self.keys_selected_causes_3lvl = list(selection_3lvl.keys())

    def __call__(self):
        return self.df

    # 1 - Location selection
    def read_data(self, city='São Paulo - SP'):
        if not self.city==city:
            self.df = pd.read_csv(f'{data_path}datasus/{capital_to_uf[city]}_standardized_1996_2019.csv', index_col=0)
            self.df['DTOBITO'] = pd.to_datetime(self.df.DTOBITO)
            self.df['IDADE'] = self.df['IDADE'].apply(lambda x: pd.Interval(*map(int, x.strip('[]()').split(','))))
            self.data = self.df.copy()
            self.city = city
            self.icd = icd
            self.keys_selected_causes_1lvl = list(selection_1lvl.keys())
            self.keys_selected_causes_2lvl = list(selection_2lvl.keys())
            self.keys_selected_causes_3lvl = list(selection_3lvl.keys())
            self.values_selected_causes_1lvl = []
            self.values_selected_causes_2lvl = []
            self.values_selected_causes_3lvl = []
        else:
            self.data = self.df.copy()
            self.icd = icd

    # 2 - Cause selection (1st level)
    def select_causes_1lvl(self, values_selected_causes_1lvl):
    	self.values_selected_causes_1lvl = values_selected_causes_1lvl
    	if values_selected_causes_1lvl:
            code_1lvl = np.array([selection_1lvl[x] for x in values_selected_causes_1lvl]).flatten()
            self.icd = self.icd[self.icd.code_1lvl.isin(code_1lvl)]
            selected_codes = self.icd.code_3lvl
            self.data = self.data[self.data.CAUSABAS.isin(selected_codes)]
            #update codes of lower levels - get keys (full name) from values (code)
            self.keys_selected_causes_2lvl = [list(selection_2lvl.keys())[list(selection_2lvl.values()).index(x[:7])] for x in self.icd.code_2lvl.unique()]
            self.keys_selected_causes_3lvl = [list(selection_3lvl.keys())[list(selection_3lvl.values()).index(x[:3])] for x in self.icd.code_3lvl.unique()]

    # 2 - Cause selection (2nd level)
    def select_causes_2lvl(self, values_selected_causes_2lvl):
        self.values_selected_causes_2lvl = values_selected_causes_2lvl
        if values_selected_causes_2lvl:
            code_2lvl = np.array([selection_2lvl[x] for x in values_selected_causes_2lvl]).flatten()
            self.icd = self.icd[self.icd.code_2lvl.isin(code_2lvl)]
            selected_codes = self.icd.code_3lvl
            self.data = self.data[self.data.CAUSABAS.isin(selected_codes)]
            #update codes of lower levels
            self.keys_selected_causes_3lvl = [list(selection_3lvl.keys())[list(selection_3lvl.values()).index(x[:3])] for x in self.icd.code_3lvl.unique()]
            self.select_causes_3lvl(self.keys_selected_causes_3lvl)
        else:
            self.data = self.df.copy()
            self.select_causes_1lvl(self.values_selected_causes_1lvl)


    # 2 - Cause selection (3rd level)
    def select_causes_3lvl(self, values_selected_causes_3lvl):
        if values_selected_causes_3lvl:
            code_3lvl = np.array([selection_3lvl[x] for x in values_selected_causes_3lvl]).flatten()
            self.icd = self.icd[self.icd.code_3lvl.isin(code_3lvl)]
            selected_codes = self.icd.code_3lvl
            self.data = self.data[self.data.CAUSABAS.isin(selected_codes)]
        else:
            self.data = self.df.copy()
            self.select_causes_1lvl(self.values_selected_causes_1lvl)
            self.select_causes_2lvl(self.values_selected_causes_2lvl)


    # 3 - Age selection
    def select_ages(self, ages):
        selected_ages = [selection_age[x] for x in ages]
        self.data = self.data[self.data.IDADE.isin(selected_ages)]

    # 4 - Sex selection
    def select_sexes(self, sexes):
        self.data = self.data[self.data.SEXO.isin(sexes)]

    # 5 - Racecolor selection
    def select_racecolors(self, racecolors):
        self.data = self.data[self.data.RACACOR.isin(racecolors)]

locations = list(capital_to_uf.keys())

##STYLE DEFINITION
app = Dash(__name__)
app.config.external_stylesheets = [dbc.themes.BOOTSTRAP]

app.layout = html.Div(
    [
    html.Div(html.H1("Seasonal Mortality Monitor"),
             style={'textAlign':'center'}),

    dbc.Row([
    dbc.Col([
        #location selection
        dbc.Row(dbc.Col(html.Label(['City:'], style={'font-weight': 'bold', 'text-align': 'center'}))),
        dbc.Row(dbc.Col(dbc.Row(dbc.Col(dcc.Dropdown(
                                    id='location_dropdown',
                                    options=[{'label': loc, 'value': loc} for loc in locations],
                                    value='São Paulo - SP',
                                    multi=False,
                                    searchable=True,
                                    placeholder='Please select city',
                                    style={'width':'100%'}))))),

        # 1lvl code
        dbc.Row(dbc.Col(html.Label(['ICD-10 code (1st level):'],
                        style={'font-weight': 'bold', 'text-align': 'center'}))),
        dbc.Row(dbc.Col(dcc.Dropdown(id='1lvlcode_dropdown', multi=True, options=[]))),

        # 2lvl code
        dbc.Row(dbc.Col(html.Label(['ICD-10 code (2nd level):'],
                        style={'font-weight': 'bold', 'text-align': 'center'}))),
        dbc.Row(dbc.Col(dcc.Dropdown(id='2lvlcode_dropdown', multi=True, options=[]))),

        # 3lvl code
        dbc.Row(dbc.Col(html.Label(['ICD-10 code (3rd level):'],
                        style={'font-weight': 'bold', 'text-align': 'center'}))),
        dbc.Row(dbc.Col(dcc.Dropdown(id='3lvlcode_dropdown', multi=True, options=[]))),
    ], width={'size':5, 'offset':0}, align='center'),
    dbc.Col(html.Div(dcc.Graph(id='time_series')), #time series and MA
                width={'size':7, 'offset':0}, align='center')
    ]),

    #pieplots (1lvl, 2lvl, 3lvl)
    html.Div(dcc.Graph(id='pie')),

    #decomposition and seasonality
    dbc.Row([dbc.Col(dcc.Graph(id='decomposition'), width={'size':6}, align='center'),
             dbc.Col(dcc.Graph(id='seasonality'), width={'size':6}, align='center'),]),

    #deaths by demographics
    dbc.Row([dbc.Col(dcc.Graph(id='age-groups'), width={'size':6}, align='center'),
             dbc.Col(dcc.Graph(id='racecolor_sex-groups'), width={'size':6}, align='center')]),

    ],
)

##CALLS
dataManager = DataManager()


#1lvl dropdown
@app.callback(
    Output('1lvlcode_dropdown', 'options'),
    Input('location_dropdown', 'value'),
)
def set_1vl_dropdown(location_data):
    dataManager.read_data(location_data)
    options = [{'label': loc, 'value': loc} for loc in selection_1lvl]
    return options

#2lvl dropdown
@app.callback(
    Output('2lvlcode_dropdown', 'options'),
    [Input('location_dropdown', 'value'),
     Input('1lvlcode_dropdown', 'value')]
)
def set_2vl_dropdown(location_data, code1lvl_data):
    dataManager.read_data(location_data)
    dataManager.select_causes_1lvl(code1lvl_data)
    options = [{'label': loc, 'value': loc} for loc in dataManager.keys_selected_causes_2lvl]
    return options

#3lvl dropdown
@app.callback(
    Output('3lvlcode_dropdown', 'options'),
    [Input('location_dropdown', 'value'),
     Input('1lvlcode_dropdown', 'value'),
     Input('2lvlcode_dropdown', 'value')]
)
def set_3vl_dropdown(location_data, code1lvl_data, code2lvl_data):
    dataManager.read_data(location_data)
    dataManager.select_causes_1lvl(code1lvl_data)
    dataManager.select_causes_2lvl(code2lvl_data)
    options = [{'label': loc, 'value': loc} for loc in dataManager.keys_selected_causes_3lvl]
    return options

#time-series
@app.callback(
    Output('time_series','figure'),
    [Input('location_dropdown','value'),
     Input('1lvlcode_dropdown','value'),
     Input('2lvlcode_dropdown','value'),
     Input('3lvlcode_dropdown', 'value')],
)
def update_plot(location_data, code1lvl_data, code2lvl_data, code3lvl_data):

    dataManager.read_data(location_data)
    dataManager.select_causes_1lvl(code1lvl_data)
    dataManager.select_causes_2lvl(code2lvl_data)
    dataManager.select_causes_3lvl(code3lvl_data)
    data = dataManager.data[['DTOBITO', 'COUNT_STANDARDIZED_BR']]
    data = data.groupby(by='DTOBITO').sum()
    data = data.resample('MS').sum()

    fig = go.Figure()
    #original TS
    fig.add_trace(go.Scatter(x=data.index, y=data.COUNT_STANDARDIZED_BR, name='Original'))
    #MA-TS
    for step in np.arange(1,25,1):
      fig.add_trace(
          go.Scatter(visible=False,
                    x=data.index,
                    y=data.COUNT_STANDARDIZED_BR.rolling(step).mean(), name='MA n='+str(step),
                    line=dict(color='red', width=2, dash='dash')))

    #start on step-12
    fig.data[12].visible=True

    #create and add slider
    steps = []
    for i in range(len(fig.data)):
        step=dict(
            method='update',
            args=[{'visible':[True]+[False]*(len(fig.data)-1)},
                    {'title': 'Time Series and Moving Average n='+str(i)}])
        step['args'][0]['visible'][i] = True
        steps.append(step)

    sliders = [dict(
        active=12,
        currentvalue={"prefix": "n= "},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders,
        title_text='Time Series and Moving Average n=12',
        title_font={'size':24},
        yaxis_title='Deaths/10k',
        xaxis_title='Date',
        title_x=0.5)

    # Create custom HTML annotations
    annotations = [
        dict(
            x=-0.1, y=1.2,
            xref="paper", yref="paper",
            text="?",  # Icon or text you want to use
            hovertext="The line plot depicts the time-series data for the chosen cause(s) <br>"+\
                        "of death. The data is presented with age-adjusted population <br>"+\
                        "standardization, using Brazil's age structure as the standard <br>"+\
                        "population, so that it is possible to compare cities with <br>"+\
                        "different age distributions. Additionaly, the Moving Average <br>"+\
                        "reveals longer-term patterns based on the selected window <br>"+\
                        "step size.",
            font=dict(size=24),
            showarrow=False,
        )
    ]
    # Add the custom annotations to the plot
    for annotation in annotations:
        fig.add_annotation(annotation)

    return fig

#pieplots
@app.callback(
    Output('pie','figure'),
    [Input('location_dropdown','value'),
     Input('1lvlcode_dropdown','value'),
     Input('2lvlcode_dropdown','value'),
     Input('3lvlcode_dropdown', 'value')],
)
def update_plot(location_data, code1lvl_data, code2lvl_data, code3lvl_data):
    #superior figure
    fig = make_subplots(rows=1, cols=3, specs=[[{"type": "pie"}, {"type": "pie"}, {"type": "pie"}]],
                        subplot_titles=('1st level', '2nd level', '3rd level'))
    dataManager.read_data(location_data)

    #1lvl pieplot
    dataManager.select_causes_1lvl(code1lvl_data)
    data = dataManager.data[['CAUSABAS', 'COUNT_STANDARDIZED_BR']][dataManager.data.CAUSABAS!='ALL'].groupby(
            by='CAUSABAS').sum().reset_index()
    data = data.merge(dataManager.icd[['code_3lvl','code_1lvl', 'name_1lvl']], left_on='CAUSABAS',
                    right_on='code_3lvl').drop(columns=['CAUSABAS','code_3lvl'])
    data = data.groupby(by=['code_1lvl', 'name_1lvl']).sum().reset_index()
    data = data.rename(columns={'code_1lvl': 'Code', 'name_1lvl': 'Name', 'COUNT_STANDARDIZED_BR': 'Deaths(%)'})
    data = data.sort_values(by='Deaths(%)', ascending=False)#[:n_causes]
    data['Deaths(%)'] = data['Deaths(%)']/data['Deaths(%)'].sum()
    data['Deaths(%)'] = data['Deaths(%)'].round(4)
    fig.add_trace(go.Pie(values=data['Deaths(%)'], labels=data['Code'], hovertext=data['Name']), row=1, col=1)
    fig.update_traces(textposition='inside', textinfo="percent+label",
                        hovertemplate='<b>%{hovertext}</b><br><br>Code: %{label}<br>Deaths(%): %{value:.2%}<extra></extra>',
                        row=1,col=1)

    #2lvl pieplot
    dataManager.select_causes_2lvl(code2lvl_data)
    data = dataManager.data[['CAUSABAS', 'COUNT_STANDARDIZED_BR']][dataManager.data.CAUSABAS!='ALL'].groupby(
            by='CAUSABAS').sum().reset_index()
    data = data.merge(dataManager.icd[['code_3lvl','code_2lvl', 'name_2lvl']], left_on='CAUSABAS',
                  right_on='code_3lvl').drop(columns=['CAUSABAS','code_3lvl'])
    data = data.groupby(by=['code_2lvl', 'name_2lvl']).sum().reset_index()
    data = data.rename(columns={'code_2lvl': 'Code', 'name_2lvl': 'Name', 'COUNT_STANDARDIZED_BR': 'Deaths(%)'})
    data = data.sort_values(by='Deaths(%)', ascending=False)#[:n_causes]
    data['Deaths(%)'] = (data['Deaths(%)']/data['Deaths(%)'].sum())#*100
    data['Deaths(%)'] = data['Deaths(%)'].round(4)
    fig.add_trace(go.Pie(values=data['Deaths(%)'], labels=data['Code'], hovertext=data['Name']), row=1, col=2)
    fig.update_traces(textposition='inside', textinfo="percent+label",
                        hovertemplate='<b>%{hovertext}</b><br><br>Code: %{label}<br>Deaths(%): %{value:.2%}<extra></extra>',
                        row=1, col=2)

    #3lvl pieplot
    dataManager.select_causes_3lvl(code3lvl_data)
    data = dataManager.data[['CAUSABAS', 'COUNT_STANDARDIZED_BR']][dataManager.data.CAUSABAS!='ALL'].groupby(
            by='CAUSABAS').sum().reset_index()
    data = data.merge(dataManager.icd[['code_3lvl', 'name_3lvl']], left_on='CAUSABAS', right_on='code_3lvl').drop(columns='CAUSABAS')
    data.columns = ['Deaths(%)', 'Code', 'Name']
    data = data.sort_values(by='Deaths(%)', ascending=False)#[:n_causes]
    data['Deaths(%)'] = (data['Deaths(%)']/data['Deaths(%)'].sum())#*100
    data['Deaths(%)'] = data['Deaths(%)'].round(4)
    fig.add_trace(go.Pie(values=data['Deaths(%)'], labels=data['Code'], hovertext=data['Name']), row=1, col=3)
    fig.update_traces(textposition='inside', textinfo="percent+label",
                    hovertemplate='<b>%{hovertext}</b><br><br>Code: %{label}<br>Deaths(%): %{value:.2%}<extra></extra>',
                    row=1, col=3)

    fig.update_layout(title_text=f"Mortality Breakdown by Cause",
                        title_font={'size':28}, title_x=0.5, showlegend=False,)

    #question mark explaining the plot
    fig.add_annotation(
        x=0, y=1,
        xref="paper", yref="paper",
        text="?",
        hovertext="The pie plots show the total deaths by cause, beaking down<br>"+\
                            "the causes of death in three levels according to the<br>"+\
                            "ICD-10 codes. The 1st level codes can be broken in 2nd<br>"+\
                            "level codes, which in turn can be broken in 3rd level codes.",
        font=dict(size=24),
        showarrow=False
    )

    return fig


#decomposition
@app.callback(
    Output('decomposition','figure'),
    [Input('location_dropdown','value'),
     Input('1lvlcode_dropdown','value'),
     Input('2lvlcode_dropdown','value'),
     Input('3lvlcode_dropdown', 'value')],
)
def update_plot(location_data, code1lvl_data, code2lvl_data, code3lvl_data):

    dataManager.read_data(location_data)
    dataManager.select_causes_1lvl(code1lvl_data)
    dataManager.select_causes_2lvl(code2lvl_data)
    dataManager.select_causes_3lvl(code3lvl_data)

    data = dataManager.data[['DTOBITO', 'COUNT_STANDARDIZED_BR']]
    data = data.groupby(by='DTOBITO').sum()
    data = data.resample('MS').sum()

    #plot decomposition
    dec = seasonal_decompose(data, model='additive', period=12)
    fig = make_subplots(rows=4, cols=1,
                        subplot_titles=('Observed', 'Trend', 'Seasonal',  'Residuals'))
    fig.add_trace(go.Scatter(
                  x = dec.observed.index,
                  y = dec.observed,
                  mode='lines',
                  name='observed'),
                  row=1, col=1)
    fig.add_trace(go.Scatter(
                  x = dec.trend.index,
                  y = dec.trend,
                  mode='lines',
                  name='trend'),
                  row=2, col=1)
    fig.add_trace(go.Scatter(
                  x = dec.seasonal.index,
                  y = dec.seasonal,
                  mode='lines',
                  name='seasonal',
                  line=dict(color='green', width=0.5)),
                  row=3, col=1)
    fig.add_trace(go.Scatter(
                  x = dec.resid.index,
                  y = dec.resid,
                  mode='markers',
                  name='resid'),
                  row=4, col=1)

    fig.add_hline(y=0, row=4, col=1, line_color='#000000')



    fig.update_layout(showlegend=False,
                      title_text='Decomposition',
                      title_font= {'size':30},
                      title_x=0.5,
                      autosize=True,
                      width=600,
                      height=800
                    )

    fig.update_xaxes(title_text='Date')
    fig.update_yaxes(title_text='Deaths/10k')

    fig.add_annotation(
        x=-0.05, y=1.05,
        xref="paper", yref="paper",
        text="?",  # Icon or text you want to use
        hovertext="We generated the decompostion plot using the python library <br>"+\
                    "statsmodels.tsa.seasonal.seasonal_decompose. The library <br>"+\
                    "does the decomposition with moving averages, breaking the <br>"+\
                    "original time series in a trend component, a seasonal <br>"+\
                    "component, and the model's residuals.",
        font=dict(size=24),
        showarrow=False,
    )

    return fig


#seasonality
@app.callback(
    Output('seasonality','figure'),
    [Input('location_dropdown','value'),
     Input('1lvlcode_dropdown','value'),
     Input('2lvlcode_dropdown','value'),
     Input('3lvlcode_dropdown', 'value')],
)
def update_plot(location_data, code1lvl_data, code2lvl_data, code3lvl_data):

    dataManager.read_data(location_data)
    dataManager.select_causes_1lvl(code1lvl_data)
    dataManager.select_causes_2lvl(code2lvl_data)
    dataManager.select_causes_3lvl(code3lvl_data)

    data = dataManager.data[['DTOBITO', 'COUNT_STANDARDIZED_BR']]
    data = data.groupby(by='DTOBITO').sum()
    data = data.resample('D').sum()

    fig = make_subplots(rows=3, cols=1,
                subplot_titles=('Deaths by year',
                                'Deaths by month',
                                'Deaths by weekday'))

    #yearly
    year_mean = data.resample('AS').sum()
    fig.add_trace(go.Bar(
                    x=year_mean.index,
                    y=year_mean.COUNT_STANDARDIZED_BR,
                    name='year'),
                    row=1, col=1)

    #monthly
    month_mean = data.groupby(data.index.month).sum()
    month_mean.index =  ['January', 'February', 'March', 'April', 'May',
                         'June', 'July', 'August', 'September', 'October',
                         'November', 'December']
    fig.add_trace(go.Bar(
                    x=month_mean.index,
                    y=month_mean.COUNT_STANDARDIZED_BR,
                    name='month'),
                    row=2, col=1)

    #weekly
    weekday_mean = data.groupby(data.index.weekday).sum()
    weekday_mean.index =  ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
                           'Friday', 'Saturday', 'Sunday']

    fig.add_trace(go.Bar(
                    x=weekday_mean.index,
                    y=weekday_mean.COUNT_STANDARDIZED_BR,
                    name='weekday'),
                    row=3, col=1)


    fig.update_layout(showlegend=False,
                      title_text='Deaths by period',
                      title_font= {'size':30},
                      title_x=0.5,
                      autosize=True,
                      width=600,
                      height=800
                    )

    fig.update_xaxes(title_text='Date')
    fig.update_yaxes(title_text='Deaths/10k')

    fig.add_annotation(
        x=-0.05, y=1.05,
        xref="paper", yref="paper",
        text="?",  # Icon or text you want to use
        hovertext="To understand the variation of deaths over time, we group the <br>"+\
                    "data by year, month and weekday. This approach provides <br>"+\
                    "insights into the mortality trends associated with different <br>"+\
                    "time periods.",
        font=dict(size=24),
        showarrow=False,
    )

    return fig

#age-groups
@app.callback(
    Output('age-groups','figure'),
    [Input('location_dropdown','value'),
     Input('1lvlcode_dropdown','value'),
     Input('2lvlcode_dropdown','value'),
     Input('3lvlcode_dropdown', 'value')],
)
def update_plot(location_data, code1lvl_data, code2lvl_data, code3lvl_data):

    dataManager.read_data(location_data)
    dataManager.select_causes_1lvl(code1lvl_data)
    dataManager.select_causes_2lvl(code2lvl_data)
    dataManager.select_causes_3lvl(code3lvl_data)

    data = dataManager.data[['IDADE', 'COUNT_STANDARDIZED_BR']].groupby(by='IDADE').sum()['COUNT_STANDARDIZED_BR']

    fig = px.bar(x=data.index.astype(str), y=data.values,
                 labels={'x':'Age group', 'y':'Deaths/10k'})

    # Create custom HTML annotations
    annotations = [
        dict(
            x=-0.15, y=1.25,
            xref="paper", yref="paper",
            text="?",  # Icon or text you want to use
            hovertext="The histogram presents the total number of deaths within<br>"+\
                        "each age bracket, divided by the total population <br>"+\
                        "across all brackets. This highlights how many deaths <br>"+\
                        "come from each age segment, rather than presenting <br>"+\
                        "the mortality rate specific to that segment.",
            font=dict(size=24),
            showarrow=False,
        )
    ]
    # Add the custom annotations to the plot
    for annotation in annotations:
        fig.add_annotation(annotation)

    fig.update_layout(
        title_text='Total deaths from each age group',
        title_font={'size':24},
        title_x=0.5,
        autosize=True,
        width=600,
        height=300)

    return fig

#racecolor_sex-groups
@app.callback(
    Output('racecolor_sex-groups','figure'),
    [Input('location_dropdown','value'),
     Input('1lvlcode_dropdown','value'),
     Input('2lvlcode_dropdown','value'),
     Input('3lvlcode_dropdown', 'value')],
)
def update_plot(location_data, code1lvl_data, code2lvl_data, code3lvl_data):

    dataManager.read_data(location_data)
    dataManager.select_causes_1lvl(code1lvl_data)
    dataManager.select_causes_2lvl(code2lvl_data)
    dataManager.select_causes_3lvl(code3lvl_data)

    data = dataManager.data[['IDADE', 'SEXO', 'RACACOR', 'COUNT_STANDARDIZED_BR']]

    interval = data.IDADE.unique()
    data['IDADE'] = data.IDADE.astype(str).str.extract(r'\((\d+)').astype(int)+2.5
    data['IDADE'] = data.IDADE*data.COUNT_STANDARDIZED_BR
    data = data.groupby(['SEXO', 'RACACOR']).agg({'IDADE':'sum', 'COUNT_STANDARDIZED_BR': 'sum'})
    data['IDADE_MEDIA'] = data['IDADE']/data['COUNT_STANDARDIZED_BR']
    data['IDADE_MEDIA'] = (data['IDADE_MEDIA']//5)*5
    data['IDADE_MEDIA_INTERVALO'] = data['IDADE_MEDIA'].apply(lambda x: pd.Interval(left=x, right=x+5))
    data['IDADE_MEDIA_INTERVALO'] = data['IDADE_MEDIA_INTERVALO'].astype(str)
    data = data[['IDADE_MEDIA', 'IDADE_MEDIA_INTERVALO']].reset_index()

    data = data.rename(columns={'SEXO': 'SEX', 'RACACOR': 'RACECOLOR',
                                'IDADE_MEDIA_INTERVALO': 'AVERAGE_AGE', 'IDADE_MEDIA':'AGE_AXIS'})

    data['AVERAGE_AGE'] = data['AVERAGE_AGE'].astype(str)  # Convert intervals to strings
    data['SEX'] = data['SEX'].astype(str)  # Convert intervals to strings

    # Define a color mapping for SEX values
    color_mapping = {'Male': 'blue', 'Female': 'red'}

    #Rename AGE_AXIS
    data = data.rename(columns={'AGE_AXIS':'AGE'})

    # Create the bar plot
    fig = px.bar(data, x='RACECOLOR', y='AGE', color='SEX',
                 color_discrete_map=color_mapping,
                 hover_data=['SEX', 'RACECOLOR', 'AVERAGE_AGE'], barmode='group')

    # Create custom HTML annotations
    annotations = [
        dict(
            x=-0.15, y=1.25,
            xref="paper", yref="paper",
            text="?",  # Icon or text you want to use
            hovertext="The histogram displays the average age of death by racecolor <br>"+\
                        "and sex, considering the aggregated number of deaths of <br>"+\
                        "every five years age group by racecolor and sex, and then <br>"+\
                        "taking the weighted average of each group, where the weight <br>"+\
                        "is the percentage population of an age group in Brazil.",
            font=dict(size=24),
            showarrow=False,
        )
    ]
    # Add the custom annotations to the plot
    for annotation in annotations:
        fig.add_annotation(annotation)


    # Update layout
    fig.update_traces(hovertemplate='RACECOLOR: %{x}<br>SEX: %{customdata[0]}<br>AVERAGE_AGE: %{customdata[1]}')
    fig.update_layout(
        title_text='Average age of death by racecolor and sex',
        title_font={'size': 24},
        title_x=0.5,
        autosize=True,
        width=600,
        height=300,
        margin=dict(t=50)  # Adjust top margin to make space for annotation
    )

    return fig


if __name__ == '__main__':
    #app.run_server(debug=True, port=8000)
    app.run_server(debug=False, host="0.0.0.0", port=8000)
