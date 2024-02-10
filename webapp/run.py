
# external
import dash
from dash import html, dcc, clientside_callback, Input, Output, no_update, CeleryManager, DiskcacheManager
import dash_bootstrap_components as dbc

# views
from app.views.navbar_view import navbar

# callbacks
import app.callbacks.designer_callback as designer_callbacks 
import app.callbacks.study_callbacks as study_callbacks

# component ids
from app.constants.component_ids import DashIdMisc, DashIdStore



# Create dash app object
app = dash.Dash(__name__, 
                external_stylesheets=[dbc.themes.JOURNAL, dbc.icons.FONT_AWESOME],
                suppress_callback_exceptions=True, # handle ID not found error for components created dynamically
                # prevent_initial_callbacks=True,
                use_pages=True,
                # meta_tags=[{'name': 'viewport',
                #             'content': 'width=device-width, initial-scale=1.0'}],
                # background_callback_manager=background_callback_manager,
                update_title=None,
                )

app.title = 'PlatePlanner' 

# register callbacks
designer_callbacks.register_callbacks(app)
study_callbacks.register_callbacks(app)


app.layout = html.Div(
    [
        dcc.Location(id=DashIdMisc.LOCATION.value, refresh=False), # stores the url of the current page 
        
        dcc.Store(id=DashIdStore.CURRENT_PLATE_DESIGN.value, storage_type="session"),
        dcc.Store(id=DashIdStore.PLATE_LIBRARY.value, storage_type="session", data={}),
        dcc.Store(id=DashIdStore.SAMPLE_LIST.value, storage_type="session", data=[]),

        dcc.Store(id=DashIdStore.STUDY.value, storage_type="session", data=[]),
        dcc.Store(id=DashIdStore.PLATE_LAYOUTS.value, storage_type="session", data=[]),

        navbar,
        html.Div(dash.page_container, className="dbc dbc-ag-grid"),
        # html.Div(designer_layout, className="dbc dbc-ag-grid"),
        # html.Div(className="mt-4"),
        # html.Div(study_layout, className="dbc dbc-ag-grid"),

    ]
)

# Control dark mode theme change 
clientside_callback(
    """
    (switchOn) => {
       switchOn
         ? document.documentElement.setAttribute('data-bs-theme', 'light')
         : document.documentElement.setAttribute('data-bs-theme', 'dark')
       return window.dash_clientside.no_update
    }
    """,
    Output(DashIdMisc.DARK_MODE_SWITCH.value, "id"),
    Input(DashIdMisc.DARK_MODE_SWITCH.value, "value"),
)

# HACK to override the styling of the switch for dark mode
@app.callback(
    Output(DashIdMisc.DARK_MODE_SWITCH.value, 'className'),
    Input(DashIdMisc.DARK_MODE_SWITCH.value, 'value')
)
def toggle_dark_mode(dark_mode_off):
    if dark_mode_off:
        return 'd-inline-block dark-mode-inactive'
    else:
        return 'd-inline-block'

@app.callback(
        [
            Output(DashIdMisc.NAVBAR.value, "color")
        ],
        [
            Input(DashIdMisc.DARK_MODE_SWITCH.value, "value"),
        ]
)
def navbar_darkmode(dark_mode_off):
    if dark_mode_off:
        return ["light"]
    else:
        return ["dark"]


if __name__ == "__main__":
    app.run_server(debug=True)