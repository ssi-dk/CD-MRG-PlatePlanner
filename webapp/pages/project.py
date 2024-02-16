import dash_bootstrap_components as dbc
from dash import html

import dash

from app.views.study_view import study_layout

dash.register_page(
    __name__,
    path="/project",
    title='PlatePlanner',
    name='Project')


layout = html.Div(
    [
        dbc.Row(dbc.Col(), className="mt-4"),
        study_layout
    ]
)