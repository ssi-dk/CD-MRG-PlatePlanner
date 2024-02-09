import base64
from io import StringIO, BytesIO
from pathlib import Path

from dash import Input, Output, State, no_update, callback_context, MATCH, ALL
from dash import dcc, html

import dash_ag_grid as dag

from dash.exceptions import PreventUpdate

import dash_ag_grid as dag
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template

import pandas as pd

from plate_planner.plate import PlateFactory
from plate_planner.study import Study

# components ids
from app.component_ids.component_ids import DashIdPlateDesigner, DashIdMisc, DashIdStore, DashIdPlateLib, DashIdStudy


def register_callbacks(app):

    @app.callback(
        [
            Output(DashIdStudy.SAMPLE_LIST_DIV.value, "children"),
            Output(DashIdStudy.SAMPLE_LIST_FILENAME_LABEL.value, "children"),
            Output(DashIdStore.SAMPLE_LIST.value, "data"),
        ],
        [
            Input(DashIdStudy.SAMPLE_LIST_UPLOAD.value, "contents"),
        ],
        [
            State(DashIdStudy.SAMPLE_LIST_UPLOAD.value, "filename"),
        ],
    )
    def upload_sample_list(content, filename):

        if content:
            filename = Path(filename)

            content_type, content_string = content.split(',')
            decoded = base64.b64decode(content_string)
            decoded_content = decoded.decode('utf-8')

            try:
                match filename.suffix:
                    case ".csv":
                        df = pd.read_csv(
                            StringIO(decoded_content), index_col=False
                        )
                        print(df)
                        
                    case [".xls", ".xlsx"]:
                        df = pd.read_excel(BytesIO(decoded), index_col=False)

                
                    
            except Exception as e:
                print(e)
                raise ValueError("Could not parse content in file")
            

            # create dag table
            column_defs = [
                {"headerName": col, "field": col,}
                for col in df.columns
            ]

            df_dict = df.to_dict("records")

            table = dag.AgGrid(
                id=DashIdStudy.PLATE_LAYOUT_DAG.value,
                rowData=df_dict,
                columnDefs=column_defs,

                persistence=True,
                persistence_type="session",

                defaultColDef={
                    "filter": True,
                    "sortable": True,
                    "floatingFilter": True,
                    },

                className="dbc ag-theme-alpine ag-theme-alpine2",
                
            )

            return table, filename.as_posix(), df_dict

        raise PreventUpdate
    

    @app.callback(
        [
            Output(DashIdStudy.SAMPLE_LIST_DIV.value, "children", allow_duplicate=True),
            Output(DashIdStore.SAMPLE_LIST.value, "data", allow_duplicate=True),
            Output(DashIdStudy.SAMPLE_LIST_FILENAME_LABEL.value, "children", allow_duplicate=True),

            Output(DashIdStudy.PL_FIG_LABEL_SELECT.value, "options"),
            Output(DashIdStudy.PL_FIG_COLOR_SELECT.value, "options"),
        ],
        [
            Input(DashIdStudy.EXAMPLE_LIST_BTN.value, "n_clicks")
        ],
        [
            State(DashIdStudy.SAMPLE_LIST_UPLOAD.value, "filename"),
        ],
        prevent_initial_call="initial_duplicate"
    )
    def upload_example_sample_list(n_clicks, filename):

        if n_clicks:
            
            df = pd.read_csv(("data/fake_case_control_Npairs_523_Ngroups_5.csv"), index_col=False)

            # create dag table
            column_defs = [
                {"headerName": col, "field": col,}
                for col in df.columns
            ]

            df_dict = df.to_dict("records")

            table = dag.AgGrid(
                id=DashIdStudy.PLATE_LAYOUT_DAG.value,
                rowData=df_dict,
                columnDefs=column_defs,

                persistence=True,
                persistence_type="session",

                defaultColDef={
                    "filter": True,
                    "sortable": True,
                    "floatingFilter": True,
                    },

                className="dbc ag-theme-alpine ag-theme-alpine2",
            )

            print(df.columns)

            columns = df.columns.to_list() + ["name", "sample_code",  "sample_name"]
            print(columns)
            
            return table, df_dict, "Example study sample list", columns, columns

        raise PreventUpdate
    
    # Show input componentws relevant for sample group randomization mode
    @app.callback(
        [
            Output(DashIdStudy.GROUP_COL_SELECT_COLLAPSE.value, "is_open"),
            Output(DashIdStudy.ALLOW_GROUP_SPLIT_COLLAPSE.value, "is_open"),
        ],
        [
            Input(DashIdStudy.RANDOMIZE_SELECT.value, "value")
        ]
    )
    def show_group_col_select(randomize_option):
        if randomize_option == "groups":
            return True, True
        else:
            return False, False
        
    # Update samples per plate options
    @app.callback(
        [
            Output(DashIdStudy.SAMPLES_PER_PLATE_SELECT.value, "options"),
            Output(DashIdStudy.SAMPLES_PER_PLATE_SELECT.value, "value")
        ],
        [
            Input(DashIdStudy.PLATE_SELECT.value, "value")
        ],
        [
            State(DashIdStore.PLATE_LIBRARY.value, "data")
        ]
    )
    def update_samples_per_plate(plate_selected, plate_lib_store):

        if not plate_selected:
            raise PreventUpdate
        
        plate = PlateFactory.dict_to_plate(plate_lib_store[plate_selected])
        options = [i for i in range(plate.capacity,1, -1)]

        return options, options[0]
    
    # Distribute samples to plates
    @app.callback(
        [
            Output(DashIdStudy.INPUT_ALERT.value, "is_open"),
            Output(DashIdStudy.INPUT_ALERT.value, "children"),

            Output(DashIdStore.PLATE_LAYOUTS.value, "data"),
            Output(DashIdStore.STUDY.value, "data"),

            Output(DashIdStudy.PLATE_LAYOUT_SELECT_DAG.value, "rowData"),

            Output(DashIdStudy.STUDY_TABS.value, "active_tab")
        ],
        [
            Input(DashIdStudy.DISTRIBUTE_SAMPLES_BTN.value, "n_clicks")
        ],
        [
            State(DashIdStudy.PLATE_SELECT.value, "value"),
            State(DashIdStore.SAMPLE_LIST.value, "data"),
            State(DashIdStore.PLATE_LIBRARY.value, "data"),

            # randomize settings
            State(DashIdStudy.RANDOMIZE_SELECT.value, "value"),

            # sample group settings 
            State(DashIdStudy.GROUP_COL_SELECT.value, "value"),
            State(DashIdStudy.ALLOW_GROUP_SPLIT_SWITCH.value, "value"),

            # desired samples per plate
            State(DashIdStudy.SAMPLES_PER_PLATE_SELECT.value, "value"),


        ]
    )
    def distribute_samples_to_plates(
            n_clicks,
            selected_plate, sample_list_store, plate_lib_store,
            randomize_option,
            group_column, allow_group_split,
            samples_per_plate
        ):

        if not selected_plate:
            return True, "No plate selected", no_update, no_update, no_update, no_update
        
        if not sample_list_store:
            return True, "No sample list loaded", no_update, no_update, no_update, no_update

        plate_dict = plate_lib_store[selected_plate]

        plate = PlateFactory.dict_to_plate(plate_dict)

        sample_list_df = pd.DataFrame(sample_list_store)

        study = Study()
        study.load_sample_list(sample_list_df)

        # randomize options
        match randomize_option:
            case "all":
                study.randomize_order(reproducible=False)
            case "groups":
                study._column_with_group_index = group_column
                study.randomize_order(case_control=True, reproducible=False)

        study.distribute_samples_to_plates(
            plate_layout=plate,
            allow_group_split=allow_group_split,
            N_samples_desired_plate=samples_per_plate
        )

        # store plates to store
        plates_store = {}
        plate_select_rowdata = []
        for plate in study:
            plates_store[plate.plate_id] = plate.as_dict()

            n_analytical_samples = plate.summary_dict()["analytical_samples_capacity"]

            plate_select_rowdata.append(
                {
                    "plate_id": plate.plate_id,
                    "n_analytical_samples": n_analytical_samples,
                }
            )

        # switch to plate layout tab
        active_tab = f"{DashIdStudy.PLATE_LAYOUT_TAB.value}_tabid"


        return no_update, no_update, plates_store, study.to_dict(), plate_select_rowdata, active_tab
    

    # User select plate from table
    @app.callback(
        [
            Output(DashIdStudy.PLATE_LAYOUT_GRAPH.value, "figure"),
            Output(DashIdStudy.PLATE_LAYOUT_TABLE_DIV.value, "children"),
        ],
        [
            Input(DashIdStudy.PLATE_LAYOUT_SELECT_DAG.value, "selectedRows"),
            Input(DashIdMisc.DARK_MODE_SWITCH.value, "value"),

            Input(DashIdStudy.PL_FIG_LABEL_SELECT.value, "value"),
            Input(DashIdStudy.PL_FIG_COLOR_SELECT.value, "value"),
        ],
        [
            State(DashIdStore.PLATE_LAYOUTS.value, "data")
        ]
    )
    def show_study_plate_layout(
            selected_row, dark_mode_off,
            label_select, color_select,
            plates_store
        ):

        if not selected_row:
            raise PreventUpdate
        
        template = "journal" if dark_mode_off else "journal_dark"
        
        plate_id = str(selected_row[0]["plate_id"])

        plate = PlateFactory.dict_to_plate(plates_store[plate_id])

        fig = plate.as_plotly_figure(
            annotation_metadata_key=label_select,
            color_metadata_key=color_select,
            theme=template,
            show_grid=False,
            fig_height=500,
            fig_width=750,
            dark_mode=not dark_mode_off,
            colormap_discrete="Set2"
        )

        plate_df = plate.as_dataframe()

        columns = plate_df.columns
        exclude = ["coordinate", "index", "empty", "rgb_color"]
        columns = [col for col in columns if col not in exclude]
        column_defs=[{"headerName": col, "field": col} for col in columns]

        plate_dag = dag.AgGrid(
            id="study_plate_dag",
            rowData=plate_df.to_dict("records"),
            # columnSize="autoSize",
            columnDefs=column_defs
        )

        return fig, plate_dag
        
        

        



