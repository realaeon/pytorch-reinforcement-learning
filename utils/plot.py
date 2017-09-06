import plotly
import torch
from plotly.graph_objs import Scatter, Line

def plot_line(xs, ys_population):
    max_colour = 'rgb(0, 132, 180)'
    mean_colour = 'rgb(0, 172, 237)'
    std_colour = 'rgba(29, 202, 255, 0.2)'

    ys = torch.Tensor(ys_population)
    ys_min = ys.min(1)[0].squeeze()
    ys_max = ys.max(1)[0].squeeze()
    ys_mean = ys.mean(1).squeeze()
    ys_std = ys.std(1).squeeze()
    ys_upper, ys_lower = ys_mean + ys_std, ys_mean - ys_std

    trace_max = Scatter(x=xs, y=ys_max.numpy(), line=Line(color=max_colour, dash='dash'), name='Max')
    trace_upper = Scatter(x=xs, y=ys_upper.numpy(), line=Line(color='transparent'), name='+1 Std. Dev.', showlegend=False)
    trace_mean = Scatter(x=xs, y=ys_mean.numpy(), fill='tonexty', fillcolor=std_colour, line=Line(color=mean_colour), name='Mean')
    trace_lower = Scatter(x=xs, y=ys_lower.numpy(), fill='tonexty', fillcolor=std_colour, line=Line(color='transparent'), name='-1 Std. Dev.', showlegend=False)
    trace_min = Scatter(x=xs, y=ys_min.numpy(), line=Line(color=max_colour, dash='dash'), name='Min')

    plotly.offline.plot({
        'data': [trace_upper, trace_mean, trace_lower, trace_min, trace_max],
        'layout': dict(title='Rewards',
                   xaxis={'title': 'Step'},
                   yaxis={'title': 'Average Reward'})
    }, filename='rewards.html', auto_open=False)
