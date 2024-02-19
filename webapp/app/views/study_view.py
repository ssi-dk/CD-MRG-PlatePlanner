from dash import dcc, html
import dash_bootstrap_components as dbc
import dash_ag_grid as dag

import plotly.graph_objects as go
import plotly.express as px

from app.constants.component_ids import DashIdStudy

randomization_options = [
    {"label": "None", "value": "none", "title": "No randomization of samples - samples have the same order as in the sample list"},
    {"label": "All samples", "value": "all", "title": "All samples in the list are randomized and then assigned plates."},
    {"label": "Sample groups", "value": "groups", "title": "Sample groups are randomized, but the order within each group is untouched. This often the case when case-control samples are analyzed."},
]

### Sample list import / inspect tab
sample_list_tab = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Stack(
                            [
                                dcc.Upload(
                                    id=DashIdStudy.SAMPLE_LIST_UPLOAD.value,
                                    children=dbc.Button('Upload File', id='upload-button'),
                                ),
                                dbc.Button("Example", outline=True, color="secondary", id=DashIdStudy.EXAMPLE_LIST_BTN.value)
                            ],
                            direction="horizontal",
                            gap=2
                        )
                    ],
                    width="auto"
                ),
                dbc.Col(
                    [
                        dbc.Label(
                            id=DashIdStudy.SAMPLE_LIST_FILENAME_LABEL.value,
                            children=[],
                            size="lg",
                        )
                    ],
                )
            ],
            className="mt-3",
            align="center"
        ),
        dbc.Row(
            dbc.Col(
                [
                    html.Div(
                            id=DashIdStudy.SAMPLE_LIST_DIV.value,
                            children=[]
                        )
                ]
            ),
            className="mt-2 mb-4"
        ),
        dbc.Row(dbc.Col(className="mb-5"))
    ]
)

### Plate assignment tab
plate_assign_tab = html.Div(
    [
        dbc.Row(
            [
                dbc.Label("Plate template", width=4),
                dbc.Col(
                    [
                        dbc.Select(id=DashIdStudy.PLATE_SELECT.value)
                    ],
                    width=6
                ),
            ],
            className="mt-4",
            align="center",
        ),
        dbc.Row(
            [
                dbc.Label("Randomization", width=4),
                dbc.Col(
                    [
                        dbc.Select(
                            id=DashIdStudy.RANDOMIZE_SELECT.value,
                            options=randomization_options,
                            value="all"
                        )
                    ],
                    width=6
                ),
            ],
            className="mt-2"
        ),
        dbc.Collapse(
            [
                dbc.Row(
                    [
                        dbc.Label("Group ID column", id=DashIdStudy.GROUP_COL_SELECT_LABEL.value, width=4),
                        dbc.Col(
                            [
                                dbc.Select(
                                    id=DashIdStudy.GROUP_COL_SELECT.value
                                ),
                            ],
                            width=6
                        ),
                    ],
                    className="mt-2"
                ),
            ],
            id=DashIdStudy.GROUP_COL_SELECT_COLLAPSE.value,
            is_open=False,
        ),
        dbc.Row(
            [
                dbc.Label("Samples per plate", width=4),
                dbc.Col(
                    [
                        dbc.Select(id=DashIdStudy.SAMPLES_PER_PLATE_SELECT.value)
                    ],
                    width=6
                ),
            ],
            className="mt-2"
        ),
        dbc.Collapse(
            [
                dbc.Row(
                    [
                        dbc.Label("Allow group split", width=4),
                        dbc.Col(
                            [
                                dbc.Switch(value=False, id=DashIdStudy.ALLOW_GROUP_SPLIT_SWITCH.value)
                            ],
                            width=6,
                            align="center",
                        )
                    ]
                ),
            ],
            id=DashIdStudy.ALLOW_GROUP_SPLIT_COLLAPSE.value,
            is_open=False
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Button(
                            "Distribute samples to plates",
                            outline=True,
                            color="success",
                            disabled=False,
                            id=DashIdStudy.DISTRIBUTE_SAMPLES_BTN.value
                        )
                    ],
                    width=4
                ),
                dbc.Col(
                    [
                        dbc.Alert(
                            "",
                            id=DashIdStudy.INPUT_ALERT.value,
                            color="primary",
                            dismissable=True,
                            is_open=False,
                            duration=4000
                        )
                    ],
                    width=8
                )
            ],
            className="mt-4",
            # align="center",
        )
    ]
)

### Plate layout tab
blank_fig = go.Figure()
blank_fig.update_layout(template='none')
blank_fig.update_layout(
    xaxis=dict(showgrid=False, zeroline=False, visible=False),
    yaxis=dict(showgrid=False, zeroline=False, visible=False),
    plot_bgcolor='rgba(0,0,0,0)', # Transparent background
    paper_bgcolor='rgba(0,0,0,0)', # Transparent paper
    margin=dict(l=0, r=0, b=0, t=0) # Remove margins
)

plate_fig_tab = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Label(children=[""], size="lg", id=DashIdStudy.FIG_PLATE_LABEL.value)
                    ]
                )
            ],
            className="mt-2 ms-2"
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Spinner(
                                dcc.Graph(
                                id=DashIdStudy.PLATE_LAYOUT_GRAPH.value,
                                figure=blank_fig,
                                config={
                                    'displayModeBar': False  # hide the toolbar
                                }
                            ),
                            color="secondary",
                            type="grow",
                        )
                    ]
                )
            ],
            # className="mt-2"
        )
    ]
)

plate_table_tab = html.Div(
    [
    dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Label(children=[""], size="lg", id=DashIdStudy.TABLE_PLATE_LABEL.value)
                    ]
                )
            ],
            className="mt-4 ms-2"
    ),
    dbc.Row(
        [
            dbc.Row(
                [
                    html.Div(id=DashIdStudy.PLATE_LAYOUT_TABLE_DIV.value, children=[])
                ]
            )
        ],
        className="mt-2"
    )
    ]
)

### PLATE LAYOUT TAB
plate_layout_select_dag = dag.AgGrid(
    id=DashIdStudy.PLATE_LAYOUT_SELECT_DAG.value,
    columnSize="responsiveSizeToFit",
    columnDefs=[
        {"headerName": "Plate", "field": "plate_id", "checkboxSelection": True},
        {"headerName": "Analytical samples", "field": "n_analytical_samples"},
    ],
    rowData=[{}],
    className="dbc ag-theme-alpine ag-theme-alpine2",

    style={"width": "100%", "height": "400px"},

    dashGridOptions={
        "rowSelection": "single",
        "noRowsOverlayComponent": "CustomNoRowsOverlay",
        "noRowsOverlayComponentParams": {
             "message": "No plates to show",
             "fontSize": 12,
         },
    },

    getRowId=f"params.data.plate_id",
)

plate_layout_tab = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        plate_layout_select_dag
                    ]
                )
            ],
            # className="mt-4"
        ),
        dbc.Row(
            [
                dbc.Label("Color", width="auto"),
                dbc.Col(
                    [
                        dbc.Select(
                            id=DashIdStudy.PL_FIG_COLOR_SELECT.value,
                            options=[],
                            value="sample_type"
                            
                        )
                    ],                                    
                ),
                dbc.Label("Label", width="auto"),
                dbc.Col(
                    [
                        dbc.Select(
                            id=DashIdStudy.PL_FIG_LABEL_SELECT.value,
                            options=[],
                            value="name"
                            
                        )
                    ],
                )
            ],
            className="mt-4",
            # style={"width": "700px"}
        )
    ]
)

### Export tab
export_tab = html.Div(
    [
        dbc.Row(
            [
                html.H5("Lists")
            ],
            className="mt-4"
        ),
        dbc.Row(
            [
                dbc.Label("Fields", width=2),
                dbc.Col(
                    [
                        dcc.Dropdown(multi=True, id=DashIdStudy.LIST_EXPORT_FIELDS_SELECT.value)
                    ]
                )
            ],
            className="mt-4"
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Button("Download", color="success", id=DashIdStudy.DOWNLOAD_LISTS_BTN.value),
                        dcc.Download(id=DashIdStudy.LIST_DOWNLOAD.value),
                    ]
                )
            ],
            className="mt-4"
        ),
        dbc.Row(dbc.Col(html.Hr())),
        dbc.Row(
            [
                html.H5("Figures")
            ],
            className="mt-4"
        ),
        dbc.Row(
            [
                dbc.Label("Color", width=2),
                dbc.Col(
                    [
                        dbc.Select(id=DashIdStudy.FIG_EXPORT_COLOR_SELECT.value)
                    ],
                ),
            ],
            className="mt-2"
        ),
        dbc.Row(
            [
                dbc.Label("Label", width=2),
                dbc.Col(
                    [
                        dbc.Select(id=DashIdStudy.FIG_EXPORT_LABEL_SELECT.value)
                    ],
                )
            ],
            className="mt-2"
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Button("Download", color="success", id=DashIdStudy.DOWNLOAD_FIGS_BTN.value),
                        dcc.Download(id=DashIdStudy.FIGS_DOWNLOAD.value),
                    ],
                    width=2,
                ),
                # dbc.Col(
                #     [
                #         dbc.Select(
                #             id=DashIdStudy.FIG_EXPORT_FORMAT_SELECT.value,
                #             options=["pdf", "png", "jpg"],
                #             value="pdf"
                #         )
                #     ],
                #     width="auto"
                # )
            ],
            className="mt-4"
        ),
    ]
)


### MAIN
study_tabs = dbc.Tabs(
    [
        dbc.Tab(
            label="Sample list",
            children=sample_list_tab,
            id=DashIdStudy.SAMPLE_LIST_TAB.value,
            tab_id=f"{DashIdStudy.SAMPLE_LIST_TAB.value}_tabid"
        ),
        dbc.Tab(
            label="Plate assignment",
            children=plate_assign_tab,
            id=DashIdStudy.PLATE_ASSIGN_TAB.value,
            tab_id=f"{DashIdStudy.PLATE_ASSIGN_TAB.value}_tabid"
        ),
        dbc.Tab(
            label="Plate layouts",
            children=plate_layout_tab,
            disabled=False,
            id=DashIdStudy.PLATE_LAYOUT_TAB.value,
            tab_id=f"{DashIdStudy.PLATE_LAYOUT_TAB.value}_tabid"
        ),
        dbc.Tab(
            label="Export",
            children=export_tab,
            disabled=False,
            id=DashIdStudy.PLATE_EXPORT_TAB.value,
            tab_id=f"{DashIdStudy.PLATE_EXPORT_TAB.value}_tabid"
        ),
    ],
    className="mt-4",
    id=DashIdStudy.STUDY_TABS.value,
)

layout_inspect_tabs = dbc.Tabs(
    [
        dbc.Tab(label="Figure", children=plate_fig_tab),
        dbc.Tab(label="Table", children=plate_table_tab),
    ],
    className="mt-4"
)


study_layout = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H4("Project Plate Plan"),
                    ]
                )
            ],
             className="ms-4 me-5"
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        study_tabs
                    ],
                    sm=12, md=12, lg=6
                ),
                dbc.Col(
                    [
                        layout_inspect_tabs
                    ],
                    sm=12, md=12, lg=6
                )
            ],
            className="ms-4 me-5"
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        
                    ]
                )
            ],
            className="mb-4"
        )
    ]
)