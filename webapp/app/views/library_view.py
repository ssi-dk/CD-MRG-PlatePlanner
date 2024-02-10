
from dash import html

import dash_bootstrap_components as dbc

import dash_ag_grid as dag

from app.constants.component_ids import DashIdPlateLib

# TABLE
column_defs = [
    {
        "headerName": "Name",
        "field": "plate_name",
        # "headerCheckboxSelection": True,
        "checkBoxSelection": True
    },
    {
        "headerName": "Size",
        "field": "size"
    },
    {
        "headerName": "Dimensions",
        "field": "dimensions"
    },
    {
        "headerName": "Analytical samples capacity",
        "field": "analytical_samples_capacity"
    },
    {
        "headerName": "Sample codes",
        "field": "sample_codes"
    }
]

plate_lib_dag = dag.AgGrid(
    id=DashIdPlateLib.PLATE_LIB_DAG.value,
    rowData=[],
    columnDefs=column_defs,
    columnSize="responsiveSizeToFit",

    dashGridOptions={
        "rowSelection": "single",
        "noRowsOverlayComponent": "CustomNoRowsOverlay",
        "noRowsOverlayComponentParams": {
             "message": "No plates to show",
             "fontSize": 12,
         },
    },

    getRowId=f"params.data.plate_name",

    persistence=True,
    persistence_type="session",

    className="dbc ag-theme-alpine ag-theme-alpine2",

    style={"width": "100%", "height": "30vh"},
)

lib_layout = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        plate_lib_dag
                    ],
                    # width=11
                )
            ],
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.ButtonGroup(
                            [
                                dbc.Button("Remove", id=DashIdPlateLib.REMOVE_PLATE_BTN.value, outline=True, color="primary"),
                                # dbc.Button("Preview", outline=True, color="success")
                            ]
                        )
                    ]
                )
            ],
            className="mt-4"
        )
    ]
)