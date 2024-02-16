import dash_bootstrap_components as dbc
from dash import html

import dash

from app.views.designer_view import designer_layout

dash.register_page(
    __name__,
    path="/plates",
    title='PlatePlanner',
    name='Plates')


layout = html.Div(
    [
        designer_layout
    ]
)