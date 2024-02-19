from dash import html, dcc
import dash_bootstrap_components as dbc

import dash_ag_grid as dag

import plotly.graph_objects as go

# component IDs
from app.callbacks.designer_callback import DashIdPlateDesigner

# views
from app.views.library_view import lib_layout

COLOR_OPTIONS = ["name", "sample_code", "sample_name"]
COLOR_OPTION_DEFAULT = "sample_code"
LABEL_OPTIONS = ["name", "sample_code", "sample_name"]
LABEL_OPTION_DEFAULT = "name"

def create_valid_alert(id):
    return dbc.Alert(
            "Define QC samples above and make sure input values are valid.",
            id=id,
            color="primary",
            dismissable=True,
            is_open=False,
            duration=4000
    )


tab_spec_rounds = html.Div(
    dbc.Row(
        [
            dbc.Col(
                [
                    dbc.Button(
                        "Add",
                        id=DashIdPlateDesigner.ADD_SPEC_ROUND_BTN.value,
                        color="success",
                        outline=True,
                        disabled=False,
                    ),
                ],
                width=2
            ),
            dbc.Col(
                [
                    html.Div(create_valid_alert(DashIdPlateDesigner.SPEC_ROUND_ALERT.value)),
                    html.Div(
                        id=DashIdPlateDesigner.SPEC_ROUND_DIV.value,
                        children=[]
                    )
                ]
            )
        ],
        className="mt-3"
    )
)

tab_start_rounds = html.Div(
    dbc.Row(
        [
            dbc.Col(
                [
                    dbc.Button(
                        "Add",
                        id=DashIdPlateDesigner.ADD_START_ROUND_BTN.value,
                        color="success",
                        outline=True,
                    )
                ],
                width=2
            ),
            dbc.Col(
                [
                    html.Div(create_valid_alert(DashIdPlateDesigner.START_ROUND_ALERT.value)),
                    html.Div(
                        id=DashIdPlateDesigner.START_ROUND_DIV.value,
                        children=[]
                    )
                ]
            )
        ],
        className="mt-3"
    )
)

tab_end_rounds = html.Div(
    dbc.Row(
        [
            dbc.Col(
                [
                    dbc.Button(
                        "Add",
                        id=DashIdPlateDesigner.ADD_END_ROUND_BTN.value,
                        color="success",
                        outline=True
                    )
                ],
                width=2
            ),
            dbc.Col(
                [
                    html.Div(create_valid_alert(DashIdPlateDesigner.END_ROUND_ALERT.value)),
                    html.Div(
                        id=DashIdPlateDesigner.END_ROUND_DIV.value,
                        children=[]
                    )
                ]
            )
        ],
        className="mt-3"
    )
)

tab_alternate_rounds = html.Div(
    dbc.Row(
        [
            dbc.Col(
                [
                    dbc.Button(
                        "Add",
                        id=DashIdPlateDesigner.ADD_ALTERNATE_ROUND_BTN.value,
                        color="success",
                        outline=True
                    )
                ],
                width=2
            ),
            dbc.Col(
                [
                    html.Div(create_valid_alert(DashIdPlateDesigner.ALTERNATE_ROUND_ALERT.value)),
                    html.Div(
                        id=DashIdPlateDesigner.ALTERNATE_ROUND_DIV.value,
                        children=[]
                    )
                ]
            )
        ],
        className="mt-3"
    )
)

qc_sample_def = html.Div(
    [
        dbc.Accordion(
            [
                dbc.AccordionItem(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Button(
                                            "Add",
                                            id=DashIdPlateDesigner.ADD_QC_SAMPLE_BTN.value,
                                            color="success",
                                            outline=True
                                        ),
                                    ],
                                    width="auto"
                                ),
                                dbc.Col(
                                    [
                                        html.Div(
                                            id=DashIdPlateDesigner.QC_SAMPLE_DIV.value,
                                            children=[

                                            ]
                                        )
                                       
                                    ]
                                )
                            ]
                        )
                    ],
                    title="QC samples"
                ),
                dbc.AccordionItem(
                    [
                        dbc.Row(
                            [
                                dbc.Label("Number of samples between QC rounds", width="auto"),
                                dbc.Col(
                                    [
                                        dbc.Input(id=DashIdPlateDesigner.QC_SAMPLE_SPACING_INPUT.value, type="number", min=0, value=11),
                                    ],
                                    width=2
                                ),
                            ],
                            className="mt-2"
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Checkbox(
                                            id=DashIdPlateDesigner.START_WITH_QC_CHECKBOX.value,
                                            label="Start with QC round",
                                            value=True
                                        )
                                    ],
                                    width="auto"
                                )
                            ],
                            className="mt-2"
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                         dbc.Tabs(
                                            [
                                                dbc.Tab(label="Start", tab_id="start_pattern", children=tab_start_rounds),
                                                dbc.Tab(label="End", tab_id="end_pattern", children=tab_end_rounds),
                                                dbc.Tab(label="Specific Round", tab_id="specific_round_patterns", children=tab_spec_rounds),
                                                dbc.Tab(label="Repeat / Alternate", tab_id="alternate_patterns", children=tab_alternate_rounds),
                                                # dbc.Tab(label="Repeat", tab_id="repeat_pattern"),
                                            ],
                                            id="pattern-tabs",
                                            active_tab="start_pattern",
                                        )
                                    ]
                                )
                            ],
                            className="mt-3"
                        )##
                    ],
                    title="Pattern"
                )
            ],
        )
    ]
    )

design_tab = html.Div(
    [
            dbc.Row(
                [
                    dbc.Label("Name", width=2),
                    dbc.Col(
                        [
                            dbc.Input(
                                id=DashIdPlateDesigner.NAME_INPUT.value,
                                value="StudyX"
                            )
                        ],
                        # sm=12, md="auto"
                    ),
                ],
                className="mt-2"
            ),
            dbc.Row(
                [
                    dbc.Label("Size", width=2),
                    dbc.Col(
                        [
                            dbc.Select(
                                id=DashIdPlateDesigner.SIZE_SELECT.value,
                                options=["6", "12", "24", "48", "96", "custom"],
                                value="96"
                            )
                        ],
                        md=12, xl=3
                    ),
                    dbc.Col(
                        [
                            dbc.Collapse(
                                [
                                    dbc.InputGroup(
                                        [
                                            dbc.Input(id=DashIdPlateDesigner.N_ROWS_SELECT.value,type="number", min=1, value=8),
                                            dbc.InputGroupText("rows"),
                                            dbc.Input(id=DashIdPlateDesigner.N_COLUMNS_SELECT.value,type="number", min=1, value=12),
                                            dbc.InputGroupText("columns"),
                                        ]
                                    )
                                ],
                                is_open=True
                            )
                        ],
                        md=12, xl=7
                    ),
                ],
                className="mt-2"
            ),
            dbc.Row(dbc.Col(html.Hr())),
            dbc.Row(dbc.Col(), className="mt-2 mb-2"),
            dbc.Row(dbc.Col(qc_sample_def)),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.ButtonGroup(
                                [
                                    dbc.Button("Create", id=DashIdPlateDesigner.CREATE_BTN.value, color="success"),
                                    dbc.Button("Preview", id=DashIdPlateDesigner.PREVIEW_BTN.value, color="secondary", outline=True),
                                    dbc.Button("Reset", id=DashIdPlateDesigner.RESET_BTN.value, color="secondary", outline=True),
                                ]
                            )
                        ],
                        width=5
                    ),
                    dbc.Col(
                        [
                            html.Div(
                                dbc.Alert(
                                    id=DashIdPlateDesigner.ADDED_TO_LIB_ALERT.value,
                                    children="kalle",
                                    dismissable=True,
                                    duration=4000,
                                    is_open=False
                                )
                            )
                        ],
                        width=7
                    )
                ],
                className="mt-4 mb-4"
            ),
    ]
)

library_tab = html.Div(lib_layout)


### Plate preview
plate_design_fig_tab = html.Div(
    [
        dbc.Spinner(
            children=html.Div(
                dcc.Graph(
                    id=DashIdPlateDesigner.PREVIEW_GRAPH.value,
                    figure=go.Figure(),
                    responsive=True,
                    style={"width":"750px", "height": "500px"},
                    config={
                        'displayModeBar': False  # hide the toolbar
                    }
                )
            ),
            type="grow",
            color="secondary"
        ),
        dbc.Row(
            [
                dbc.Label("Color", width="auto"),
                dbc.Col(
                    [
                        dbc.Select(
                            id=DashIdPlateDesigner.COLOR_SELECT.value,
                            options=COLOR_OPTIONS,
                            value=COLOR_OPTION_DEFAULT
                        )
                    ],                                    
                ),
                dbc.Label("Label", width="auto"),
                dbc.Col(
                    [
                        dbc.Select(
                            id=DashIdPlateDesigner.LABEL_SELECT.value,
                            options=LABEL_OPTIONS,
                            value=LABEL_OPTION_DEFAULT
                        )
                    ],
                )
            ],
            className="ms-4 me-4",
            style={"width": "700px"}
        )
    ]
)

### table
column_defs = [
    {
        "headerName": "Well Index",
        "field": "index"
    },
    {
        "headerName": "Well",
        "field": "name"
    },
    {
        "headerName": "Sample Type",
        "field": "sample_type"
    },
        {
        "headerName": "Sample Code",
        "field": "sample_code"
    }
]

plate_preview_dag = dag.AgGrid(
    id=DashIdPlateDesigner.PREVIEW_DAG.value,
    rowData=[],
    columnSize="responsiveSizeToFit",

    dashGridOptions={
        "rowSelection": "single",
    },

    columnDefs=column_defs,

    className="dbc ag-theme-alpine ag-theme-alpine2",

    style={"width": "100%", "height": "500px"},
)

plate_design_table_tab = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        plate_preview_dag
                    ]
                )
            ],
            
        ),
        
    ]
)


### MAIN
designer_layout = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H4("Plate templates"),
                    ]
                )
            ],
            className="mt-2 ms-4 me-4"
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Tabs(
                            [
                                dbc.Tab(label="Design", children=design_tab),
                                dbc.Tab(label="Library", children=library_tab)
                            ]
                        )
                    ],
                    sm=12, md=12, xl=6
                ),
                dbc.Col(
                    [
                        dbc.Tabs(
                            [
                                dbc.Tab(label="Figure", children=plate_design_fig_tab),
                                dbc.Tab(label="Table", children=plate_design_table_tab),
                            ],
                            # className="mt-2"
                        )
                    ],
                    sm=12, md=12, xl=6
                )
            ],
            className="mt-2 ms-4 me-4"
        ),
    ]
)