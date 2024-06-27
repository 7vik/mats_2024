import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_training_loss(train_losses):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=train_losses, mode='lines', name='', line=dict(color='darkred', width=2)))
    fig.update_layout({'plot_bgcolor': 'rgba(255, 255, 255, 1)',})
    fig.update_layout(
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGray'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGray'),
    )
    fig.update_xaxes(title_text='Optimization Step')
    fig.update_yaxes(title_text='CrossEntropy Loss')
    fig.update_layout(width=600, height=400, autosize=False)
    return fig

def plot_kl_divs(kl_divs, sorted=False):
    fig = go.Figure()
    if sorted:
        kl_divs = sorted(kl_divs)
    fig.add_trace(go.Scatter(y=kl_divs, mode='lines', name='', line=dict(color='darkred', width=2)))
    fig.update_layout({'plot_bgcolor': 'rgba(255, 255, 255, 1)',})
    fig.update_layout(
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGray'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGray'),
    )
    fig.update_xaxes(title_text='Neuron Index')
    fig.update_yaxes(title_text='KL Divergence')
    fig.update_layout(width=600, height=400, autosize=False)
    return fig

def plot_direction_metrics(kl_divs, entropies, sort=False, title=''):
    if sort:
        kl_divs_t = sorted(kl_divs, reverse=True)
        entropies_t = sorted(entropies, reverse=True)
    else:
        kl_divs_t = kl_divs
        entropies_t = entropies

    # Subplot for entropy differences and KL divergence
    fig = make_subplots(rows=1, cols=2, shared_yaxes=False, shared_xaxes=True, horizontal_spacing=0.1)

    fig.add_trace(go.Scatter(
        y=entropies_t, 
        mode='lines', 
        name='',
        line=dict(color='darkblue', width=2),
        fill='tozeroy',
        fillcolor='rgba(0, 0, 139, 0.2)'  # RGBA for darkblue with lower opacity
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        y=kl_divs_t, 
        mode='lines', 
        name='',
        line=dict(color='darkred', width=2),
        fill='tozeroy',
        fillcolor='rgba(139, 0, 0, 0.2)'  # RGBA for darkred with lower opacity
    ), row=1, col=2)

    fig.update_layout({
        'plot_bgcolor': 'rgba(255, 255, 255, 1)',
        'showlegend': False,
        'width': 1000,
        'height': 400,
        'autosize': False
    })

    # Update x and y axes for both subplots
    fig.update_xaxes(
        title_text='Feature #', 
        showgrid=True, 
        gridwidth=1, 
        gridcolor='LightGray',
        row=1, col=1
    )
    fig.update_xaxes(
        title_text='Feature #', 
        showgrid=True, 
        gridwidth=1, 
        gridcolor='LightGray',
        row=1, col=2
    )
    fig.update_yaxes(
        title_text='Entropy Difference', 
        showgrid=True, 
        gridwidth=0.1, 
        gridcolor='LightGray',
        row=1, col=1
    )
    fig.update_yaxes(
        title_text='KL Divergence', 
        showgrid=True, 
        gridwidth=0.1, 
        gridcolor='LightGray',
        row=1, col=2
    )
    fig.update_layout(title_text=title)
    return fig