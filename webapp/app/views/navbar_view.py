from dash import dcc, html
import dash_bootstrap_components as dbc

from app.constants.component_ids import DashIdMisc

# color switch
dark_mode_switch = html.Span(
    [
        dbc.Label(
            className="fa fa-moon",
            # style={"color": "white"}
            # html_for="color-mode-switch"
        ),
        dbc.Switch(
            id=DashIdMisc.DARK_MODE_SWITCH.value,
            value=True,
            className="d-inline-block",
            style={"margin-left": "8px"},
            persistence=True,
            
        ),
        dbc.Label(
            className="fa fa-sun",
            # style={"color": "white"}
            # html_for="color-mode-switch"
        ),
    ],
)

navbar = dbc.NavbarSimple(
            [
                dbc.NavItem(dbc.NavLink("Home", href="/"), style={"font-size":"1.1rem"}),
                dbc.NavItem(dbc.NavLink("Plates", href="/plates"), style={"font-size":"1.1rem"}),
                dbc.NavItem(dbc.NavLink("Project", href="/project"), style={"font-size":"1.1rem"}),
                dbc.NavItem(dbc.NavLink("  ")),
                dbc.NavItem(
                    html.Div(dark_mode_switch,style={"padding-top": "10px"}),
                    
                ),
                dbc.NavItem(dbc.NavLink("  ")),
                dbc.NavItem(dbc.NavLink(
                    html.I(className="fa-brands fa-github", style={"font-size":"2.0rem"}),
                    href="https://github.com/ssi-dk/CD-MRG-DashPlatePlanner",
                    target="_blank"),
                ),
            ],
            brand=html.Img(src="./assets/logo.png", width="300"),
            brand_href="/",
            color="light",
            id=DashIdMisc.NAVBAR.value
        )