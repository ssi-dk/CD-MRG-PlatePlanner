
from dash import html

import dash_bootstrap_components as dbc

import dash_ag_grid as dag

from app.component_ids.component_ids import DashIdPlateLib

# TABLE

column_defs = [
    {
        "headerName": "Name",
        "field": "plate_name",
        "headerCheckboxSelection": True,
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
        "headerName": "Total QC samples",
        "field": "n_qc_samples"
    },
    {
        "headerName": "QC sample types",
        "field": "qc_sample_types"
    }
]

plate_lib_dag = dag.AgGrid(
    rowData=[],
    columnDefs=column_defs,
    columnSize="responsiveSizeToFit",
    className="dbc ag-theme-alpine ag-theme-alpine2",

)

lib_layout = html.Div(
    [
        dbc.Row(dbc.Col(), className="mt-2"),
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
                        dbc.Button("Remove selected")
                    ]
                )
            ],
            className="mt-4"
        )
    ]
)