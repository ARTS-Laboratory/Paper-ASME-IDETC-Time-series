from plotly.graph_objects import Figure

# const
GLOBAL_PLOTLY_FONT = {
    'family': 'Serif, Droid Serif, Times New Roman',
    'color': 'black'
}

def update_font(fig: Figure):
    fig['layout']['font'].update(**GLOBAL_PLOTLY_FONT)