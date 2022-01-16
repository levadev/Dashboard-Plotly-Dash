import dash
import webbrowser
from dash import dcc
from dash import html
import base64
import io
from dash.dependencies import Input, Output, State
import pandas as pd
import datetime as dt
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from dateutil.relativedelta import relativedelta
from copy import deepcopy
import math

app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
variants = ['Показать все',
            'Сколько всего приборов на каждом месторождении?',
            'Сколько всего приборов по каждому виду измерения?',
            'Количество приборов по виду измерения на каждом месторождении',
            'Количество приборов по стране производства с разделением по месторождению',
            'У какого количества приборов уже прошел срок службы?',
            'У какого количества приборов есть протокол?',
            'Какое количество приборов просрочено?',
            'Минимальное, среднее и максимальное значение погрешности по видам измерений',
            'В каких организациях поверяется больше приборов?',
            'Чтобы узнать пройдет ли прибор следующую поверку введите его ID']

app.layout = html.Div([
                        html.Div([
                                html.Div([
                                          html.H1(["Это твой "]),
                                          html.H1([" DASH"],
                                                  style={
                                                    'margin-left': '1.5rem',
                                                    'color':'orange'}),
                                          html.H1(["board!"])],
                                          style={'display':'flex'}), 
                        dcc.Upload(id='upload-data', children=html.Div(['Перенесите или ', html.A('выберите файл')], style={'margin': '9px 100px 0 100px'}),
                                  style={
                                      'height': '40px',
                                      'borderWidth': '5px',
                                      'borderStyle': 'dashed',
                                      'borderRadius': '50px',
                                      'borderColor': 'orange',
                                      'textAlign': 'center',
                                      'padding': '10px'
                                  },
                                   multiple=True)], style={'display':'flex',
                                                          'justify-content': 'space-evenly'}),
                        html.Hr(), 
                        html.Div(id='output-data-upload'),
                        html.Div([dcc.Dropdown(id='demo-dropdown', placeholder='Выберите...',
                                    options=[{'label': c, 'value': c} for c in variants],
                                    style={'padding':'0 20px',
                                            'width': '100%',
                                            'color':'black'}),
                        dcc.Input(id='demo', type='number', placeholder='Введите ID', style={'margin-right':'20px','width': '50%'})], style={'display': 'flex'}),
                        html.Hr(),
                        html.Div(id='dd-output-container'),
                        html.Div(id='d2out')
                        
                      ], 
                      style={'backgroundColor': '#111111',
                              'color':'white',
                              'padding-top': '40px'})
ch = pd.DataFrame()
today = dt.date.today()

def parse_contents(contents, filename, date):
  content_type, content_string = contents.split(',')

  decoded = base64.b64decode(content_string)
  try:
    if 'csv' in filename:
      df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), parse_dates=True, index_col='id')
    elif 'xlsx' in filename:
      df = pd.read_excel(io.BytesIO(decoded), parse_dates=True, index_col='id')
  except Exception as e:
    print(e)
    return html.Div(['Ошибка загрузки файла!'])

  df = df.dropna()
  df.field = pd.to_numeric(df.field,downcast='integer').astype(str) 
  
  tom = df['type of measurement'].reset_index()['type of measurement'].str.lower()
  df['country'] = df['country'].str.lower().str.title()
  df['die time'] = [df['release date'][i] + relativedelta(years=df['life time'][i]) for i in df.index]
  df['next time'] = [df['last date'][i] + relativedelta(month=int((df['interval'][i]/12 - int(df['interval'][i]/12))*12)) + relativedelta(days=365*int(df['interval'][i]/12)) for i in df.index]
  df['organization'] = df['organization'].str.upper()

  df['die time'] = df['die time'].dt.date
  df['next time'] = df['next time'].dt.date
  df['last date'] = df['last date'].dt.date
  df['release date'] = df['release date'].dt.date

  def find(i, what):
    return tom[i].find(what)>=0

  for i in range(len(tom)):
    try:
      if find(i,'уровн') or find(i,'расх') or find(i,'объ') or find(i,'вмести') or find(i,'масс'):
        tom[i] = 'Измерения уровня, объема, расхода, вместимости'
        continue
      if find(i,'давлен') or find(i,'уум'):
        tom[i] = 'Измерения давления, вакуума'
        continue
      if find(i,'акусти') or find(i,'вибр') or find(i,'перемещ') or find(i,'движ') or find(i,'линей'):
        tom[i] = 'Измерения акустических величин'
        continue
      if find(i,'сост') or find(i,'загаз') or find(i,'влаг') or find(i,'плот') or find(i,'лаж') or find(i,'войст') or find(i,'хим'):
        tom[i] = 'Физико-химические измерения'
        continue
      if find(i,'темп') or find(i,'тепл'):
        tom[i] = 'Измерения температуры'  
        continue
      if find(i,'опти'):
        tom[i] = 'Оптические измерения'
        continue
      if find(i,'комплекс') or find(i,'систем'):
        tom[i] = 'Измерительные системы' 
        continue
      if find(i,'врем'):
        tom[i] = 'Измерения времени' 
        continue
    except:
      continue
  df['type of measurement'] = list(tom)
  return df

fig_G = go.Figure().update_layout(hovermode="x",
                                    template="plotly_dark",
                                    height=500,
                                    legend_orientation="h")
fig_bp = make_subplots(1,2, specs=[[{"type": "bar"}, {"type": "pie"}]])
marker = dict(color='yellow',
                  line=dict(width=2,
                            color='red'))
def fig_bar(x, y, name, xaxis_title, yaxis_title, title):
  fig_bar = deepcopy(fig_G)
  fig_bar.update_layout(xaxis_title=xaxis_title,
                          yaxis_title=yaxis_title,
                          title=title,
                          xaxis_tickmode='linear')
  fig_bar.add_trace(go.Bar(x=x, y=y, name=name,
                            marker=marker))
  return fig_bar

def fig_pie(values, labels, name, xaxis_title, yaxis_title, title):
  fig_pie = deepcopy(fig_G)
  fig_pie.update_layout(xaxis_title=xaxis_title,
                          yaxis_title=yaxis_title,
                          title=title)
  fig_pie.add_trace(go.Pie(values=values,
                            labels=labels,
                            pull=0.08,
                            hole=0.7,
                            marker=dict(colors=['red', 'yellow'])))
  return fig_pie

def count_of_devicies_on_the_Fields(df):
  x=df.groupby('field').count().index
  y=df.groupby('field').count()['name']
  return fig_bar(x, y,'Количество приборов','Месторождения', 'Количество приборов','КОЛИЧЕСТВО ПРИБОРОВ НА МЕСТОРОЖДЕНИЯХ')

def count_of_type(df):
  fig = deepcopy(fig_bp)
  y=df.groupby('type of measurement').count().index
  x=df.groupby('type of measurement').count()['field']
  fig.update_layout(hovermode="y",
                      template="plotly_dark",
                      height=600,
                      showlegend=False,
                      title='КОЛИЧЕСТВО ПРИБОРОВ С РАЗДЕЛЕНИЕ ПО ВИДУ ИЗМЕРЕНИЙ')
  traces = [go.Bar(x=x, y=y,
                      name='Количество приборов',
                      orientation='h',
                      marker=marker),
              go.Pie(values=x,
                      labels=y,
                      pull=0.08,
                      hole=0.7)]
  for i in traces:
    fig.add_trace(i, 1, traces.index(i)+1)
  return fig

def hist_allType_allField(df):
  fig = go.Figure().update_layout(hovermode="x",
                                    template="plotly_dark",
                                    legend_orientation='h',
                                    legend=dict(x=.5,
                                                xanchor="center"),
                                    height=600)
  fig.update_layout(xaxis_title='Месторождения',
                      yaxis_title='Количество приборов',
                      title='КОЛИЧЕСТВО ПРИБОРОВ С РАЗДЕЛЕНИЕ ПО ВИДУ ИЗМЕРЕНИЙ НА МЕСТОРОЖДЕНИЯХ')
  for i in df.groupby('type of measurement').count().index:
    df1 = df[df['type of measurement'] == i].groupby('field').count()
    fig.add_trace(go.Bar(x=df1.index, y=df1['name'], name=i))
  return fig

def country_one_field(df):
  fig = make_subplots(3, 3, subplot_titles=list(['Месторождение %s'%str(i+1) for i in range(9)]),
                        specs=[[{"type": "pie"} for i in range(3)] for i in range(3)])
  fig.update_layout(template="plotly_dark",
                      height=1000,
                      legend_orientation="h")
    
  def cyklus(x,y):
    for i in range(3):
      dfe = df[df.field == str(i+y)].groupby('country').count()['name']
      dfe = dfe.append(pd.Series(dfe[dfe < 5].sum(), index=['Другие'])).drop(dfe[dfe < 5].index)
      fig.add_trace(go.Pie(values=dfe,
                            labels=dfe.index,
                            pull=0.08,
                            hole=0.7,
                            name='Месторождение ' + str(i+y)),
                      x,i+1)
  cyklus(1,1)
  cyklus(2,4)
  cyklus(3,7)
  return fig

def by_the_time(what_time, time1_name, time2_name, title, df):
  time1 = df[['field',what_time]][df[what_time] < today].groupby('field').count()[what_time]  
  time2 = df[['field',what_time]][df[what_time] > today].groupby('field').count()[what_time]
  fig = deepcopy(fig_bp)
  fig.update_layout(hovermode="y",
                    template="plotly_dark",
                      height=600,
                      showlegend=False, 
                      title=title)
  traces = [go.Bar(y=time1.index,
                        x=time1,
                        name=time1_name,
                        orientation="h", 
                        marker=dict(color='red')),
            go.Bar(y=time2.index,
                    x=time2,
                    name=time2_name,
                    orientation="h",
                    marker=dict(color='yellow'))]
  for i in traces:
    fig.add_trace(i,1,1)
  colors = ['red', 'yellow']
  pc = pd.DataFrame([time1.sum(), time2.sum()], index=[time1_name, time2_name])
  fig.add_trace(go.Pie(values=pc[0],
                        labels=pc.index,
                        name='Круговая<br>диаграмма',
                        pull=0.08,
                        hole=0.7,
                        marker=dict(colors=colors))
                    ,1,2)
  return fig

def relative_error(df):
  tm = df.groupby('type of measurement')
  fig = go.Figure()
  traces = [go.Bar(y=tm.min().index,
                    x=tm.min()['relative error'],
                    visible=False,
                    orientation='h',
                    name='Минимум',
                    marker=marker),
            go.Bar(y=tm.mean().index,
                    x=tm.mean()['relative error'],
                    visible=False, orientation='h',
                    name='Среднее',
                    marker=marker),
            go.Bar(y=tm.max().index,
                    x=tm.max()['relative error'],
                    visible=True,
                    orientation='h',
                    name='Максимум',
                    marker=marker)]
  for i in traces:
    fig.add_trace(i)

  steps = []
  status = ['Минимум', 'Среднее', 'Максимум']

  for i in status:
    step = dict(method="update",
                  label=i,
                  args=[{"visible": [False] * len(fig.data)}])
    step["args"][0]["visible"][status.index(i)] = True
    steps.append(step)
      
  fig.update_layout(hovermode="y",
                    template="plotly_dark",
                      height=500,
                      xaxis_title='Относительная погрешность, %',
                      title='ЗНАЧЕНИЯ ПОГРЕШНОСТИ ПО ВИДАМ ИЗМЕРЕНИЙ',
                      sliders=[dict(currentvalue={"prefix": "Агрегирующая функция: "},
                                    pad={"t": 50},
                                    steps=steps,
                                    name='Статус')])
  return fig

def organization(df):
  orgntn = df.groupby('organization').count().name.drop(['ООО "ГАЗПРОМНЕФТЬ-АВТОМАТИЗАЦИЯ"','ФБУ «ТЮМЕНСКИЙ ЦСМ»'], axis=0)
  orgntn = orgntn.append(pd.Series(orgntn[orgntn <= 5].sum(), index=['ДРУГОЕ'])).drop(orgntn[orgntn <= 5].index,axis=0)

  fig = go.Figure().update_layout(hovermode="y",
                                  template="plotly_dark",
                                    height=700,
                                    legend_orientation="h",
                                    xaxis_title='Количество приборов',
                                    title='КОЛИЧЕСТВО ПРИБОРОВ, КОТОРОЕ ПОВЕРЯЕТСЯ В ОРГАНИЗАЦИЯХ')
  fig.add_trace(go.Bar(x=orgntn,
                        y=orgntn.index,
                        name='Организация',
                        orientation='h',
                        marker=marker))
  return fig

def have_forms(df):
  no = list(df[df.form == 'нет'].groupby('method').count().index)
  yes = list(df[df.form == 'да'].groupby('method').count().index)
  search = [i for i in no if i in yes]
  for i in search:
    no.remove(i)
  form = pd.DataFrame([len(no),len(yes)], index=['Нет формы протокола','Есть форма протокола'])
  return fig_pie(form[0], form.index, 'Формы', '', '', 'ФОРМЫ ПРОТОКОЛОВ')

def prediction(id, df):
  x = list(range(1,21))
  l_df = df['range'][id]
  find = l_df.find('-')
  ot = float(l_df[:find])
  do = float(l_df[find+1:])
  e = (do-ot)*0.7+ot
  y = [round(abs((df.loc[id]['measurement '+str(i+1)] - e)/e),4) for i in range(20)]

  def error(f):
    return round(np.sum((f(x) - y)**2),8)

  def poly1d_y(degree):
    return np.poly1d(np.polyfit(x,y,degree))

  linspace_y = np.linspace(start=0, stop=x[-1]*2, num=2000)

  df_id = df.loc[id][['field', 'name', 'type', 'model', 'manufacturer', 'relative error', 'last date', 'die time', 'next time']]
  df_id = df_id.append(pd.Series([error(poly1d_y(1)),
                                      error(poly1d_y(2))],
                                    index=['Linear error',
                                            'Parabolic error']))   
  def fsolve(y):
    try:
      b = list(y)[1]
      a = list(y)[0]
      c = list(y)[2]
      discr = b ** 2 - 4 * a * c
      if discr > 0:
        x1 = (-b + math.sqrt(discr)) / (2 * a)
        x2 = (-b - math.sqrt(discr)) / (2 * a)
        return [x1, x2]
      elif discr == 0:
        x = -b / (2 * a)
        return [x]
      else:
         return 0
    except e:
      return [np.nan]
  
  x_fsolve_y = fsolve(poly1d_y(2) - df_id['relative error'])
 
  if (int(x_fsolve_y[0]) < 0 and df_id['die time'] > today) or int(x_fsolve_y[0]) > 50:
    title = 'Похоже, что прибор (id %s) пройдет следующую поверку!'%id
  elif int(x_fsolve_y[0]) > 0 and df_id['die time'] < today:
    title = 'Вам стоит присмотреться к прибору (id %s),<br>так как заявленый срок службы уже прошел!'%id
  elif int(x_fsolve_y[0]) < 0 and df_id['die time'] < today:
    title = 'Похоже, что прибор (id %s) не пройдет следующую поверку!'%id
  else:
    title = 'Примерно через %s контрольных периодов<br>отклонение превысит допустимое' %int(x_fsolve_y[0]) 

  fig = make_subplots(1,2,
                          column_widths=[0.6,0.4],
                          subplot_titles=['Регрессионный анализ (id %s)'%id,''],
                          specs=[[{"type": "scatter"}, {"type": "table"}]])
  fig.update_layout(template="plotly_dark",
                        height=500,
                        legend_orientation="h",
                        legend=dict(x=.5,
                                    xanchor="center"), 
                        xaxis_title='Период',
                        yaxis_title='Отклонение',
                        title=title,
                        xaxis_range=[0,50])
  traces11 = [go.Scatter(x=x, y=y,
                            name='Отклонение',
                            mode='markers',
                            marker=dict(size=10,
                                        color='yellow',
                                        line=dict(width=2,
                                                  color='red'))),
              go.Scatter(x=linspace_y,
                            y=poly1d_y(1)(linspace_y),
                            name='Линейный',
                            mode='lines',
                            line=dict(color='red')),
              go.Scatter(x=linspace_y,
                            y=poly1d_y(2)(linspace_y),
                            name='Параболический',
                            mode='lines',
                            line=dict(color='yellow'))] 
  for i in traces11:
    fig.add_trace(i,1,1)

  fig.add_trace(go.Table(header=dict(values=['feature', 'value']),
                            cells=dict(values=[df_id.index,df_id],
                                        align='left'))
                    ,1,2)
  return fig

@app.callback(Output('output-data-upload', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))

def update_output(list_of_contents, list_of_names, list_of_dates):
  if list_of_contents is not None:
    global ch
    for c, n, d in zip(list_of_contents, list_of_names, list_of_dates):
      ch = parse_contents(c, n, d)
    return html.H3(["Файл успешно загружен!"], style={'backgroundColor':'green','color':'black','textAlign': 'center'})

@app.callback(
    Output('dd-output-container', 'children'),
    Input('demo-dropdown', 'value')
)
def update_output2(value):
  if value is not None:
    try:
      global ch
      style={'textAlign': 'center', 'width': '100%'}
      all =   [count_of_devicies_on_the_Fields(ch),
            count_of_type(ch), 
            hist_allType_allField(ch), 
            country_one_field(ch), 
            by_the_time('die time', 'Срок службы уже истек', 'Прибор еще послужит', 'КОЛИЧЕСТВО ПРИБОРОВ ПО СРОКУ СЛУЖБЫ', ch),
            have_forms(ch),
            by_the_time('next time', 'Просрочено', 'Поверено', 'КОЛИЧЕСТВО ПРИБОРОВ ПО ДАТЕ СЛЕДУЮЩЕЙ ПОВЕРКИ', ch), 
            relative_error(ch), 
            organization(ch)
      ] 
      if value == variants[0]:
        return [dcc.Graph(figure=i, style=style) for i in all]
      for i in range(1,10):
        if value == variants[i]:
          return dcc.Graph(figure=all[i-1], style=style)
    except:
      return html.Div(["Упс... Файл еще не загрузили :("], style={'textAlign': 'center'})


@app.callback(
    Output('d2out', 'children'),
    Input('demo', 'value')
)
def update_output3(value):
  global ch
  if value is not None:
    try:
      return dcc.Graph(figure=prediction(value,ch), style={'textAlign': 'center', 'width': '100%'})
    except:
      if len(ch) != 0:
        return html.Div(["Такого ID не существует :("], style={'textAlign': 'center'})
      else:
        return html.Div(["Упс... Файл еще не загрузили :("], style={'textAlign': 'center'})


if __name__ == '__main__':
  webbrowser.open('http://127.0.0.1:8050/')
  app.run_server(debug=False)