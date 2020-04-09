import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
import pandas as pd
import seaborn as sns
from dash.dependencies import Input, Output, State
import dash_table
import pickle
import numpy as np

def generate_table(dataframe, page_size=10):
    return dash_table.DataTable(
        id='dataTable',
        columns=[{
            "name": i,
            "id": i
        } for i in dataframe.columns],
        data=dataframe.to_dict('records'),
        page_action="native",
        page_current=0,
        page_size=page_size,
    )

data = pd.read_csv('Churn_modelling.csv')

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

loadModel = pickle.load(open('bank_customer_xgb.sav', 'rb'))

app.layout = html.Div(
    children=[
        html.H1('Bank Customer Dashboard'),
        html.Div(children='''by: Stevano'''),
        dcc.Tabs(children=[
                dcc.Tab(value ='Tab1',label ='Data Frame', children=[   
                    html.Div(children =[
                        html.Div([
                            html.P('Has a Credit Card'),
                            dcc.Dropdown(value='None',
                            id='filter-cc',
                            options=[{'label':'None', 'value': 'None' },
                                {'label':0, 'value': 0 },
                                {'label':1, 'value': 1 }])
                            ], className='col-3'),
                        html.Div([
                            html.P('Geography'),
                            dcc.Dropdown(value='None',
                            id='filter-geo',
                            options=[{'label':'None', 'value': 'None' },
                                {'label':'France', 'value': 'France' },
                                {'label':'Germany', 'value': 'Germany' },
                                {'label':'Spain', 'value': 'Spain' }])
                            ], className='col-3'),
                        html.Div([
                            html.P('Gender'),
                            dcc.Dropdown(value='None',
                            id='filter-gen',
                            options=[{'label':'None', 'value': 'None' },
                                {'label':'Male', 'value': 'Male' },
                                {'label':'Female', 'value': 'Female' }])
                            ], className='col-3'),   
                        html.Div([
                            html.P('Active Member'),
                            dcc.Dropdown(value='None',
                            id='filter-act',
                            options=[{'label':'None', 'value': 'None' },
                                {'label': 0 , 'value': 0 },
                                {'label': 1 , 'value': 1 }])
                            ], className='col-3'), 
                        html.Div([
                            html.P('Number of Products'),
                            dcc.Dropdown(value='None',
                            id='filter-np',
                            options=[{'label':'None', 'value': 'None' },
                                {'label':1, 'value': 1 },
                                {'label':2, 'value': 2 },
                                {'label':3, 'value': 3 },
                                {'label':4, 'value': 4}])
                            ], className='col-3'),
                        html.Div([
                                html.P('Exited'),
                                dcc.Dropdown(value='None',
                                id='filter-exit',
                                options=[{'label':'None', 'value': 'None' },
                                    {'label': 1, 'value': 1 },
                                    {'label': 0, 'value': 0 }])
                            ], className='col-3')
                    ], className='row'),
                    html.Div(children =[
                        html.Div([
                            html.P('Max Rows'),
                            dcc.Input(id="filter-row",
                            placeholder="input number",
                            type="number",
                            value=10)
                        ], className='col-3'),
                    ], className='row'),

                    html.Div(children =[
                        html.Div([
                        html.Button('Search', id='search-button')
                        ], className='col-3')
                    ], className='row'),
                    html.Br(),
                    html.Div(id='div-table',
                    children=[generate_table(data)])
                    ]),
                dcc.Tab(value='Tab2',label='Bar Chart',children=[
                    html.Div(children = [
                        html.P('Category:'),
                        dcc.Dropdown(
                            id='filter-cat',
                            options=[{'label': 'Age', 'value': 'Age'},
                            {'label': 'Tenure', 'value': 'Tenure'},
                            {'label': 'HasCrCard', 'value': 'HasCrCard'},
                            {'label': 'NumOfProducts', 'value': 'NumOfProducts'},
                            {'label': 'Geography', 'value': 'Geography'},
                            {'label': 'Gender', 'value': 'Gender'}
                            ],
                            value='Age'
                        )
                    ], className = 'row col - 1'),
                    html.Br(),
                    dcc.Graph(id='graph-bank')
                ]),
                dcc.Tab(value='Tab3', label='Predict Result', children=[
                            html.Div(children=[
                                html.Div(children=[
                                    html.Div([
                                        html.P('Has Credit Card'),
                                        dcc.Dropdown(id='s_CC',
                                        options=[{'label':'No', 'value':0},
                                                {'label':'Yes', 'value':1}],
                                        value=0)], className='col-3'),
                                    html.Div([
                                        html.P('Is Active Member'),
                                        dcc.Dropdown(id='s_ActMem',
                                        options=[{'label':'No', 'value':0},
                                                {'label':'Yes', 'value':1}],
                                        value=0)], className='col-3'),
                                    html.Div([
                                        html.P('Number Of Product'),
                                        dcc.Dropdown(id='s_Npr',
                                        options=[{'label':1, 'value':1},
                                                {'label':2, 'value':2},
                                                {'label':3, 'value':3},
                                                {'label':4, 'value':4}],
                                        value=1)], className='col-3'),
                                    html.Div([
                                        html.P('Gender'),
                                        dcc.Dropdown(id='s_Gender',
                                        options=[{'label':'Female', 'value':'Female'},
                                                {'label':'Male', 'value':'Male'}],
                                        value='Male')], className='col-3'),
                                    html.Div([
                                        html.P('Geography'),
                                        dcc.Dropdown(id='s_Geo',
                                        options=[{'label':'France', 'value':'France'},
                                                {'label':'Germany', 'value':'Germany'},
                                                {'label':'Spain', 'value':'Spain'}],
                                        value='France')
                                    ], className='col-3')
                                ], className = 'row'),

                                html.Br(),
                                html.Div(children=[
                                    html.Div([
                                        html.P('Credit Score'),
                                        dcc.Input(id='s_CS',
                                            type='number',
                                            value=0)
                                    ], className='col-3'),
                                    html.Div([
                                        html.P('Age(18-92)'),
                                        dcc.Input(id='s_Age',
                                            type='number',
                                            value=18)
                                    ], className='col-3'),
                                    html.Div([
                                        html.P('Tenure'),
                                        dcc.Input(id='s_Tenure',
                                            type='number',
                                            value=0)
                                    ], className='col-3'),
                                    html.Div([
                                        html.P('Balance'),
                                        dcc.Input(id='s_Balance',
                                            type='number',
                                            value=0)
                                    ], className='col-3'),
                                    html.Div([
                                        html.P('EstimatedSalary'),
                                        dcc.Input(id='s_Salary',
                                        type='number',
                                        value=0)], 
                                        className='col-3'),
                                ], className = 'row')
                            ]),
                    html.Br(),
                    html.Div([
                        html.Button('Predict', id='buttonpredict', style=dict(width='100%'))
                    ], className='col-2 row'),       
                    html.Br(),
                    html.Div(id='display-selected-values')
                    ])
            ],
    content_style={
        'fontFamily': 'Calibri',
        'borderBottom': '1px solid #d6d6d6',
        'borderLeft': '1px solid #d6d6d6',
        'borderRight': '1px solid #d6d6d6',
        'padding': '44px'
            })
    ],
    style={
        'maxWidth': '1200px',
        'margin': '0 auto'
    })


## UPDATE TABLE CALLBACK
@app.callback(
    Output(component_id = 'div-table', component_property = 'children'),
    [Input(component_id = 'search-button', component_property = 'n_clicks')],
    [State(component_id = 'filter-row', component_property = 'value'),
    State(component_id = 'filter-cc', component_property = 'value'),
    State(component_id='filter-geo',component_property='value'),
    State(component_id='filter-gen',component_property='value'),
    State(component_id='filter-act',component_property='value'),
    State(component_id='filter-np',component_property='value'),
    State(component_id='filter-exit',component_property='value')]
)

def update_table(n_clicks, row, cc, geo, gen, act, npr, exit):
    data = pd.read_csv('Churn_Modelling.csv')
    if cc != 'None':
        data = data[data['HasCrCard'] == cc]
    if geo != 'None':
        data = data[data['Geography'] == geo]
    if gen != 'None':
        data = data[data['Gender'] == gen]
    if act != 'None':
        data = data[data['IsActiveMember'] == act]
    if npr != 'None':
        data = data[data['NumOfProducts'] == npr]
    if exit != 'None':
        data = data[data['Exited'] == exit]
    
    children = [generate_table(data, page_size = row)]
    return children


## UPDATE GRAPH1 CALLBACK
@app.callback(
    Output(component_id = 'graph-bank', component_property='figure'),
    [Input('filter-cat', 'value')]
)

def update_figure(category_dropdown_name):
    df = pd.read_csv('Churn_Modelling.csv')
    counts1 = df[df['Exited']==1][category_dropdown_name].value_counts()
    counts2 = df[df['Exited']==0][category_dropdown_name].value_counts()
    keys1 = counts1.index.tolist()
    keys2 = counts2.index.tolist()
    values1 = counts1.values.tolist()
    values2 = counts2.values.tolist()

    data = data=[
        go.Bar(name='Churn', x=keys1, y=values1),
        go.Bar(name='Not Churn', x=keys2, y=values2)
    ]
    bar_figure = {'data': data}

    return bar_figure


## Prediction
@app.callback(
    Output('display-selected-values', 'children'),
    [Input(component_id = 'buttonpredict', component_property='n_clicks')],
    [State('s_CS', 'value'), 
    State('s_Age', 'value'),
    State('s_Tenure', 'value'),
    State('s_Balance', 'value'),
    State('s_Npr', 'value'),
    State('s_CC', 'value'),
    State('s_ActMem', 'value'),
    State('s_Salary', 'value'),
    State('s_Geo', 'value'),
    State('s_Gender', 'value')])

def set_display_children(n_clicks, CS, Age, Tenure, Balance, npr, CC, Act, Salary, Geo, Gen):
    file = []
    files= []
    file.append(CS)
    file.append(Age)
    file.append(Tenure)
    file.append(Balance)
    file.append(npr)
    file.append(CC)
    file.append(Act)
    file.append(Salary)
    if Geo == 'Germany':
        file.append(1)
        file.append(0)
    elif Geo == 'France':
        file.append(0)
        file.append(0)
    elif Geo == 'Spain':
        file.append(0)
        file.append(1)
    if Gen == 'Male':
        file.append(1)
    elif Gen == 'Female':
        file.append(0)
    
    for i in file:
        if i == 'Germany':
            files.extend([1,0])
        elif i == 'France':
            files.extend([0,0])
        elif i == 'Spain':
            files.extend([0,1])
        else:
            files.append(i)
    file2 = np.array(files)
    # return file2
    loadModel = pickle.load(open('bank_customer_pipe_xgb.sav', 'rb'))
    result = loadModel.predict(file2.reshape(1,11))
    res = []
    for i in result:
        if i == 1:
            res.append('Churning Customer')
        else:
            res.append('Not Churning Customer')
    if n_clicks != None:
        return 'This is a {}'.format(res[0])

if __name__ == '__main__':
    app.run_server(debug=True)