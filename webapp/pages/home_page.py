import dash_bootstrap_components as dbc
from dash import html

import dash

dash.register_page(
    __name__,
    path="/",
    title='PlatePlanner',
    name='Home')

footer = dbc.Row(
    dbc.Col(
        [
            html.P("PlatePlanner - Streamline your plate designs for LC-MS analysis and beyond.", className="footer-title"),
            html.A("GitHub Repository", href="https://github.com/YourRepo/PlatePlanner", className="footer-link"),
            html.A("PlatePlanner Python Package", href="https://pypi.org/project/plateplanner/", className="footer-link"),
            html.P("Developed by the PlatePlanner Team", className="footer-text")
        ],
        md=12,
        className="ms-4"
    ),
    className="mt-5 py-3 bg-light",  # Add your background color and padding here.
    # style={"width": "100%", "margin-top": "auto"}   # Ensure the footer spans the full width.
)



body = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        # html.H4("Let's plate!"),
                        html.P(
                            [
                            """
                            Welcome to PlatePlanner, a tool for designing multi-well plate layouts
                            for LC-MS analysis and beyond!
                            """,
                            ],
                            style={"font-size":"1.2rem"}
                        ),
                        # html.P(
                        #     """
                        #     PlatePlanner streamlines the process of creating custom 
                        #     plate templates, distributing samples, and generating detailed run lists and visualizations.
                            
                        #     """
                        # )
                    ],
                    width=6
                ),
               
            ],
            class_name="mt-4"
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div( style={"margin-top": "50px"}),
                    ]
                )
            ],
            class_name="mt-4"
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H4("Getting started"),
                        html.Div(className="mt-4"),

                        dbc.Stack(
                            [
                                html.A(
                                    [
                                        html.I(className="fa-solid fa-pen-to-square", style={"font-size": "2.0rem"}),
                                        html.H5("Design plate templates", className="d-inline ms-2")  # Adjusted for inline display and spacing
                                    ],
                                    href="/plates",  
                                    className="text-decoration-none",  
                                )
                            ],
                            direction="horizontal",
                            gap=3
                        ),

                        

                        html.Div(
                            [
                                """
                                Create a plate template: set plate size and add desired QC samples and frequency patterns.
                                """
                            ],
                            className="mt-2"
                        ),

                        html.Div( style={"margin-top": "50px"}),

                        dbc.Stack(
                            [
                                html.A(
                                    [
                                        html.I(className="fa-solid fa-layer-group", style={"font-size": "2.0rem"}),
                                        html.H5("Distribute samples to plates", className="d-inline ms-2")  # Adjusted for inline display and spacing
                                    ],
                                    href="/project",  
                                    className="text-decoration-none",  
                                )
                            ],
                            direction="horizontal",
                            gap=3
                        ),

                        html.Div(
                            [
                                """
                                Upload a sample list and let PlatePlanner distribute the samples to your plate templates. Samples or sample groups (blocks) can be uniformly randomized across plates. 
                                """

                            ],
                            className="mt-2"
                        ),

                        html.Div( style={"margin-top": "50px"}),

                        dbc.Stack(
                            [
                                html.A(
                                    [
                                        html.I(className="fa-solid fa-file-arrow-down", style={"font-size": "2.0rem"}),
                                        html.H5("Inspect/download", className="d-inline ms-2")  # Adjusted for inline display and spacing
                                    ],
                                    href="/project",  
                                    className="text-decoration-none",  
                                )
                            ],
                            direction="horizontal",
                            gap=3
                        ),

                        html.Div(
                            [
                                """
                                Explore your new project plates - download run lists or custom plate visualization`                                """

                            ],
                            className="mt-2"
                        ),
                        
                    ],
                    width=6
                ),
            ]
        ),
        html.Div(className="mt-4 mb-4"),
    ],
)

layout = html.Div(
    [
        body,
        # footer
    ],
    # style={'display': 'flex', 'flex-direction': 'column', 'height': '100vh'}
)