import base64
from io import StringIO, BytesIO

from dash import Input, Output, State, no_update, callback_context, MATCH, ALL
from dash import dcc, html

from dash.exceptions import PreventUpdate

from plate_planner.plate import PlateFactory

import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template

import pandas as pd

# components ids
from app.constants.component_ids import DashIdPlateDesigner, DashIdMisc, DashIdStore, DashIdPlateLib, DashIdStudy


load_figure_template(["journal", "journal_dark"])

def remove_dynamic_row(triggered_id, row_container, remove_btn_type):
    """Removes a dynamic row from a container based on the triggered remove button's ID.

    This function iterates through a list of rows (each represented as a dictionary of components),
    identifies the row that contains a remove button with the specific index and type, and removes it from the list.

    Args:
        triggered_id (dict): The ID of the component that triggered the callback. Expected to be a dictionary with
                             keys 'type' and 'index', where 'index' corresponds to the row to be removed.
        row_container (list): The list of rows (container) from which a row is to be removed. Each row is expected
                              to be a dictionary with a specific structure.
        remove_btn_type (str): The type of the remove button to be matched in the 'id' of the component.

    Returns:
        list: A new list of rows with the specified row removed.

    Example:
        triggered_id = {"type": "custom-remove-btn-type", "index": "row-3"}
        row_container = [...]
        updated_container = remove_dynamic_row(triggered_id, row_container, "custom-remove-btn-type")
    """
    index_to_remove = triggered_id.get("index")

    print(f"INDEX TO REMOVE: {index_to_remove}")

    new_row_container = []
    for row in row_container:
        component_ids = []
        for col in row.get('props', {}).get('children', []):
            for comp in col.get('props', {}).get('children', []):
                if isinstance(comp, dict):
                    comp_id = comp.get('props', {}).get('id', {})
                    if comp_id:
                        component_ids.append(comp_id)

        remove_button_present = any(
            comp_id.get('type') == remove_btn_type and comp_id.get('index') == index_to_remove
            for comp_id in component_ids
        )

        if not remove_button_present:
            new_row_container.append(row)

    return new_row_container



def register_callbacks(app):

    # Plate page loads
    @app.callback(
            [
                Output(DashIdPlateLib.PLATE_LIB_DAG.value, "rowData"),
            ],
            [
                Input(DashIdMisc.LOCATION.value, "pathname")
            ],
            [
                State(DashIdStore.PLATE_LIBRARY.value, "data")
            ]
    )
    def page_load(url, plate_lib_store):

        if url == "/plates":
            print(plate_lib_store.keys())
             # --- Update library table ---
            # Create plate objects from store
            plates = [PlateFactory.dict_to_plate(plate_dict) for name, plate_dict in plate_lib_store.items()]
            # Create dataframe for lib table
            summaries = [plate.summary_dict() for plate in plates]
            df = pd.DataFrame(summaries)
            df.insert(0, value=list(plate_lib_store.keys()), column="plate_name")

            return [df.to_dict("records")]

        raise PreventUpdate
    
    # Add a QC sample definition
    @app.callback(
        [
            Output(DashIdPlateDesigner.QC_SAMPLE_DIV.value, "children", allow_duplicate=True)
        ],
        [
            Input(DashIdPlateDesigner.ADD_QC_SAMPLE_BTN.value, "n_clicks")
        ],
        [
            State(DashIdPlateDesigner.QC_SAMPLE_DIV.value, "children")
        ],
        prevent_initial_call='initial_duplicate'
    )
    def add_qc_sample_input(n_clicks, qc_sample_rows):

        if not callback_context.triggered_id == DashIdPlateDesigner.ADD_QC_SAMPLE_BTN.value:
            raise PreventUpdate
        
        qc_sample_row = dbc.Row(
                            [
                                dbc.Label(f"{n_clicks}", width="auto"),
                                dbc.Col(
                                    [  
                                        dbc.Input(
                                            placeholder="Name",
                                            id={
                                                "type": DashIdPlateDesigner.QC_SAMPLE_NAME_INPUT_TYPE.value,
                                                "index": f"{DashIdPlateDesigner.QC_SAMPLE_INDEX.value}_{n_clicks}"
                                            },
                                            valid=False,
                                            invalid=True
                                        ),
                                    ],
                                    width=6
                                ),
                                dbc.Col(
                                    [
                                        dbc.Input(
                                            placeholder="Abbreviation",
                                            id={
                                                "type": DashIdPlateDesigner.QC_SAMPLE_ABBREV_INPUT_TYPE.value,
                                                "index": f"{DashIdPlateDesigner.QC_SAMPLE_INDEX.value}_{n_clicks}"
                                            },
                                            valid=False,
                                            invalid=True
                                        ),
                                    ]
                                ),
                                dbc.Col(
                                    [
                                        dbc.Button(
                                            "Remove",
                                            color="primary",
                                            outline=True,
                                            id={
                                                    "type": DashIdPlateDesigner.QC_SAMPLE_REMOVE_BTN_TYPE.value,
                                                    "index": f"{DashIdPlateDesigner.QC_SAMPLE_INDEX.value}_{n_clicks}"
                                            },
                                        ),
                                    ],
                                    width="auto"
                                )
                            ],
                            className="mb-2"
                        )
        
        print(f"INDEX: {DashIdPlateDesigner.QC_SAMPLE_INDEX.value}_{n_clicks}")
        
        qc_sample_rows.append(qc_sample_row)

        return [qc_sample_rows]
    
    # Remove a QC sample defintion
    # NOTE RESET all defined QC rounds to guarantee consistency
    # TODO handle this more elegantly perhaps
    @app.callback(
        [
            Output(DashIdPlateDesigner.QC_SAMPLE_DIV.value, "children", allow_duplicate=True),
            Output(DashIdPlateDesigner.SPEC_ROUND_DIV.value, "children", allow_duplicate=True)
        ],
        [
            Input({"type": DashIdPlateDesigner.QC_SAMPLE_REMOVE_BTN_TYPE.value, "index": ALL}, "n_clicks")
        ],
        [
            State(DashIdPlateDesigner.QC_SAMPLE_DIV.value, "children")
        ],
        prevent_initial_call='initial_duplicate'
    )
    def remove_qc_sample_input(n_clicks_list, qc_sample_rows):

        remove_btn_type = DashIdPlateDesigner.QC_SAMPLE_REMOVE_BTN_TYPE.value

        # Check if any button was clicked
        if not any(n_clicks and n_clicks > 0 for n_clicks in n_clicks_list):
            raise PreventUpdate

        if not callback_context.triggered:
            raise PreventUpdate

        # Get the id of the button that triggered the callback
        triggered_id = callback_context.triggered_id


        # If the callback was triggered by a pattern-matching callback, triggered_id is a dictionary
        if not isinstance(triggered_id, dict):
            raise PreventUpdate

        new_qc_sample_rows = remove_dynamic_row(triggered_id, qc_sample_rows, remove_btn_type)

        return new_qc_sample_rows, []
    
    # Check validity of input for QC sample definitions
    @app.callback(
    [
        Output({'type': DashIdPlateDesigner.QC_SAMPLE_NAME_INPUT_TYPE.value, 'index': MATCH}, 'valid'),
        Output({'type': DashIdPlateDesigner.QC_SAMPLE_NAME_INPUT_TYPE.value, 'index': MATCH}, 'invalid'),
        Output({'type': DashIdPlateDesigner.QC_SAMPLE_ABBREV_INPUT_TYPE.value, 'index': MATCH}, 'valid'),
        Output({'type': DashIdPlateDesigner.QC_SAMPLE_ABBREV_INPUT_TYPE.value, 'index': MATCH}, 'invalid')
    ],
    [
        Input({'type': DashIdPlateDesigner.QC_SAMPLE_NAME_INPUT_TYPE.value, 'index': ALL}, 'value'),
        Input({'type': DashIdPlateDesigner.QC_SAMPLE_ABBREV_INPUT_TYPE.value, 'index': ALL}, 'value')
    ],
    [
        State({'type': DashIdPlateDesigner.QC_SAMPLE_NAME_INPUT_TYPE.value, 'index': MATCH}, 'id'),
        State({'type': DashIdPlateDesigner.QC_SAMPLE_ABBREV_INPUT_TYPE.value, 'index': MATCH}, 'id')
    ]
    )
    def update_input_validity(name_values, abbrev_values, name_id, abbrev_id):
        if not callback_context.triggered:
            raise PreventUpdate
        
        prop_id = callback_context.triggered[0]['prop_id']
        value = callback_context.triggered[0]['value']

        if 'qc-sample-name-input-type' in prop_id:
            # Logic for "Name" input field
            is_valid = bool(value)
            return is_valid, not is_valid, no_update, no_update
        elif 'qc-sample-abbrev-input-type' in prop_id:
            # Logic for "Abbreviation" input field
            is_valid = len(value) <= 3
            return no_update, no_update, is_valid, not is_valid
        else:
            raise PreventUpdate
        

    # Add spec round definition
    @app.callback(
        [
            Output(DashIdPlateDesigner.SPEC_ROUND_DIV.value, "children"),
            Output(DashIdPlateDesigner.SPEC_ROUND_ALERT.value, "is_open")
        ],
        [
            Input(DashIdPlateDesigner.ADD_SPEC_ROUND_BTN.value, "n_clicks")
        ],
        [
            State(DashIdPlateDesigner.SPEC_ROUND_DIV.value, "children"),

            State({"type": DashIdPlateDesigner.QC_SAMPLE_NAME_INPUT_TYPE.value, "index": ALL}, "value"),
            State({"type": DashIdPlateDesigner.QC_SAMPLE_ABBREV_INPUT_TYPE.value, "index": ALL}, "value")
        ]
    )
    def add_spec_qc_round(n_clicks, spec_round_rows, qc_sample_names, qc_sample_abbrev):

        if None in qc_sample_abbrev or len(qc_sample_abbrev) == 0:
            return no_update, True

        options = [abbrev for abbrev in qc_sample_abbrev]
        options = options if options is not None else []

        if not callback_context.triggered:
            raise PreventUpdate

        spec_round_row = dbc.Row(
                            [
                                # dbc.Label(f"{n_clicks}"),
                                dbc.Col(
                                    [
                                        dbc.InputGroup(
                                            [
                                                dbc.InputGroupText("Round"),
                                                dbc.Input(
                                                    placeholder="Name",
                                                    type="number",
                                                    min=1,
                                                    value=1,
                                                    id={
                                                        "type": DashIdPlateDesigner.SPEC_ROUND_INPUT_TYPE.value,
                                                        "index": f"{DashIdPlateDesigner.SPEC_ROUND_INDEX.value}_{n_clicks}"
                                                    },
                                                ),
                                            ]
                                        )
                                    ],
                                ),
                                dbc.Col(
                                    [
                                        dcc.Dropdown(
                                            # placeholder="Abbreviation",
                                            options=options,
                                            id={
                                                "type": DashIdPlateDesigner.SPEC_ROUND_SELECT_TYPE.value,
                                                "index": f"{DashIdPlateDesigner.SPEC_ROUND_INDEX.value}_{n_clicks}"
                                            },

                                        ),
                                    ],
                                ),
                                dbc.Col(
                                    [
                                        dbc.Button(
                                            "Remove",
                                            color="primary",
                                            outline=True,
                                            id={
                                                "type": DashIdPlateDesigner.REMOVE_SPEC_ROUND_BTN_TYPE.value,
                                                "index": f"{DashIdPlateDesigner.SPEC_ROUND_INDEX.value}_{n_clicks}"
                                            }
                                        ),
                                    ],
                                    width="auto"
                                )
                            ],
                            className="mb-2"
                        )
                
        spec_round_rows.append(spec_round_row)

        return spec_round_rows, False
    
    # Remove spec round definition
    @app.callback(
        [
            Output(DashIdPlateDesigner.SPEC_ROUND_DIV.value, "children", allow_duplicate=True)
        ],
        [
            Input({"type": DashIdPlateDesigner.REMOVE_SPEC_ROUND_BTN_TYPE.value, "index": ALL}, "n_clicks")
        ],
        [
            State(DashIdPlateDesigner.SPEC_ROUND_DIV.value, "children")
        ],
        prevent_initial_call='initial_duplicate'
    )
    def remove_spec_round(n_clicks_list, spec_rounds_rows):

        remove_btn_type = DashIdPlateDesigner.REMOVE_SPEC_ROUND_BTN_TYPE.value

        # Check if any button was clicked
        if not any(n_clicks and n_clicks > 0 for n_clicks in n_clicks_list):
            raise PreventUpdate

        if not callback_context.triggered:
            raise PreventUpdate

        # Get the id of the button that triggered the callback
        triggered_id = callback_context.triggered_id

        # If the callback was triggered by a pattern-matching callback, triggered_id is a dictionary
        if not isinstance(triggered_id, dict):
            raise PreventUpdate

        new_spec_rounds_rows = remove_dynamic_row(triggered_id, spec_rounds_rows, remove_btn_type)

        return [new_spec_rounds_rows]
    
    # Add start round definitions
    @app.callback(
        [
            Output(DashIdPlateDesigner.START_ROUND_DIV.value, "children"),
            Output(DashIdPlateDesigner.START_ROUND_ALERT.value, "is_open")
        ],
        [
            Input(DashIdPlateDesigner.ADD_START_ROUND_BTN.value, "n_clicks")
        ],
        [
            State(DashIdPlateDesigner.START_ROUND_DIV.value, "children"),

            State({"type": DashIdPlateDesigner.QC_SAMPLE_NAME_INPUT_TYPE.value, "index": ALL}, "value"),
            State({"type": DashIdPlateDesigner.QC_SAMPLE_ABBREV_INPUT_TYPE.value, "index": ALL}, "value"),

            State(DashIdStore.CURRENT_PLATE_DESIGN.value, "data")
        ]
    )
    def add_start_qc_round(n_clicks, start_round_rows, qc_sample_names, qc_sample_abbrev, plate_json):

        if None in qc_sample_abbrev or len(qc_sample_abbrev) == 0:
            return no_update, True

        options = [abbrev for abbrev in qc_sample_abbrev]
        options = options if options is not None else []

        if not callback_context.triggered:
            raise PreventUpdate
        
        current_plate_design = PlateFactory.dict_to_plate(plate_json)

        default_position = len(start_round_rows)  + 1

        position_options = []
        for label, value in  zip(current_plate_design._alphanumerical_coordinates, range(1, current_plate_design.capacity+1)):
            position_options.append({"label": label, "value": value})

        start_round_row = dbc.Row(
                            [
                                # dbc.Label(f"{len(spec_round_rows)+1}", width="auto"),
                                dbc.Col(
                                    [
                                        dbc.InputGroup(
                                            [
                                                dbc.InputGroupText("Position"),
                                                dbc.Select(
                                                    id={
                                                        "type": DashIdPlateDesigner.START_ROUND_POS_SELECT_TYPE.value,
                                                        "index": f"{DashIdPlateDesigner.START_ROUND_INDEX.value}_{n_clicks}"
                                                    },
                                                    options=position_options,
                                                    value=default_position
                                                )
                                            ]
                                        )
                                    ],
                                ),
                                dbc.Col(
                                    [
                                        dcc.Dropdown(
                                            # placeholder="Abbreviation",
                                            options=options,
                                            id={
                                                "type": DashIdPlateDesigner.START_ROUND_SELECT_TYPE.value,
                                                "index": f"{DashIdPlateDesigner.START_ROUND_INDEX.value}_{n_clicks}"
                                            },

                                        ),
                                    ],
                                ),
                                dbc.Col(
                                    [
                                        dbc.Button(
                                            "Remove",
                                            color="primary",
                                            outline=True,
                                            id={
                                                "type": DashIdPlateDesigner.REMOVE_START_ROUND_BTN_TYPE.value,
                                                "index": f"{DashIdPlateDesigner.START_ROUND_INDEX.value}_{n_clicks}"
                                            }
                                        ),
                                    ],
                                    width="auto"
                                )
                            ],
                            className="mb-2"
                        )
        
        start_round_rows.append(start_round_row)

        return start_round_rows, False
    
    # Remove start round definition
    @app.callback(
        [
            Output(DashIdPlateDesigner.START_ROUND_DIV.value, "children", allow_duplicate=True)
        ],
        [
            Input({"type": DashIdPlateDesigner.REMOVE_START_ROUND_BTN_TYPE.value, "index": ALL}, "n_clicks")
        ],
        [
            State(DashIdPlateDesigner.START_ROUND_DIV.value, "children")
        ],
        prevent_initial_call='initial_duplicate'
    )
    def remove_start_round(n_clicks_list, spec_rounds_rows):

        remove_btn_type = DashIdPlateDesigner.REMOVE_START_ROUND_BTN_TYPE.value

        # Check if any button was clicked
        if not any(n_clicks and n_clicks > 0 for n_clicks in n_clicks_list):
            raise PreventUpdate

        if not callback_context.triggered:
            raise PreventUpdate

        # Get the id of the button that triggered the callback
        triggered_id = callback_context.triggered_id

        # If the callback was triggered by a pattern-matching callback, triggered_id is a dictionary
        if not isinstance(triggered_id, dict):
            raise PreventUpdate

        new_spec_rounds_rows = remove_dynamic_row(triggered_id, spec_rounds_rows, remove_btn_type)

        return [new_spec_rounds_rows]
    
    # Add end round definitions
    @app.callback(
        [
            Output(DashIdPlateDesigner.END_ROUND_DIV.value, "children"),
            Output(DashIdPlateDesigner.END_ROUND_ALERT.value, "is_open")
        ],
        [
            Input(DashIdPlateDesigner.ADD_END_ROUND_BTN.value, "n_clicks")
        ],
        [
            State(DashIdPlateDesigner.END_ROUND_DIV.value, "children"),

            State({"type": DashIdPlateDesigner.QC_SAMPLE_NAME_INPUT_TYPE.value, "index": ALL}, "value"),
            State({"type": DashIdPlateDesigner.QC_SAMPLE_ABBREV_INPUT_TYPE.value, "index": ALL}, "value"),

            State(DashIdStore.CURRENT_PLATE_DESIGN.value, "data")
        ]
    )
    def add_end_qc_round(n_clicks, end_round_rows, qc_sample_names, qc_sample_abbrev, plate_json):


        if None in qc_sample_abbrev or len(qc_sample_abbrev) == 0:
            return no_update, True

        options = [abbrev for abbrev in qc_sample_abbrev]
        options = options if options is not None else []

        if not callback_context.triggered:
            raise PreventUpdate
        
        current_plate_design = PlateFactory.dict_to_plate(plate_json)

        position_options = []
        for label, value in  zip(current_plate_design._alphanumerical_coordinates[::-1], range(1, current_plate_design.capacity+1)):
            position_options.append({"label": label, "value": value})

        default_position = len(end_round_rows) + 1
        
        end_round_row = dbc.Row(
                            [
                                # dbc.Label(f"{len(spec_round_rows)+1}", width="auto"),
                                dbc.Col(
                                    [
                                        dbc.InputGroup(
                                            [
                                                dbc.InputGroupText("Position"),
                                                dbc.Select(
                                                    id={
                                                        "type": DashIdPlateDesigner.END_ROUND_POS_SELECT_TYPE.value,
                                                        "index": f"{DashIdPlateDesigner.END_ROUND_INDEX.value}_{n_clicks}"
                                                    },
                                                    options=position_options,
                                                    value=default_position
                                                )
                                            ]
                                        )
                                    ],
                                ),
                                dbc.Col(
                                    [
                                        dcc.Dropdown(
                                            # placeholder="Abbreviation",
                                            options=options,
                                            id={
                                                "type": DashIdPlateDesigner.END_ROUND_SELECT_TYPE.value,
                                                "index": f"{DashIdPlateDesigner.END_ROUND_INDEX.value}_{n_clicks}"
                                            },

                                        ),
                                    ],
                                ),
                                dbc.Col(
                                    [
                                        dbc.Button(
                                            "Remove",
                                            color="primary",
                                            outline=True,
                                            id={
                                                "type": DashIdPlateDesigner.REMOVE_END_ROUND_BTN_TYPE.value,
                                                "index": f"{DashIdPlateDesigner.END_ROUND_INDEX.value}_{n_clicks}"
                                            }
                                        ),
                                    ],
                                    width="auto"
                                )
                            ],
                            className="mb-2"
                        )
        
        end_round_rows.append(end_round_row)

        return end_round_rows, False
    
    # Remove end round definition
    @app.callback(
        [
            Output(DashIdPlateDesigner.END_ROUND_DIV.value, "children", allow_duplicate=True)
        ],
        [
            Input({"type": DashIdPlateDesigner.REMOVE_END_ROUND_BTN_TYPE.value, "index": ALL}, "n_clicks")
        ],
        [
            State(DashIdPlateDesigner.END_ROUND_DIV.value, "children")
        ],
        prevent_initial_call='initial_duplicate'
    )
    def remove_end_round(n_clicks_list, spec_rounds_rows):

        remove_btn_type = DashIdPlateDesigner.REMOVE_END_ROUND_BTN_TYPE.value

        # Check if any button was clicked
        if not any(n_clicks and n_clicks > 0 for n_clicks in n_clicks_list):
            raise PreventUpdate

        if not callback_context.triggered:
            raise PreventUpdate

        # Get the id of the button that triggered the callback
        triggered_id = callback_context.triggered_id

        # If the callback was triggered by a pattern-matching callback, triggered_id is a dictionary
        if not isinstance(triggered_id, dict):
            raise PreventUpdate

        new_spec_rounds_rows = remove_dynamic_row(triggered_id, spec_rounds_rows, remove_btn_type)

        return [new_spec_rounds_rows]

    # Add alternating round definition
    @app.callback(
        [
            Output(DashIdPlateDesigner.ALTERNATE_ROUND_DIV.value, "children"),
            Output(DashIdPlateDesigner.ALTERNATE_ROUND_ALERT.value, "is_open")
        ],
        [
            Input(DashIdPlateDesigner.ADD_ALTERNATE_ROUND_BTN.value, "n_clicks")
        ],
        [
            State(DashIdPlateDesigner.ALTERNATE_ROUND_DIV.value, "children"),

            State({"type": DashIdPlateDesigner.QC_SAMPLE_NAME_INPUT_TYPE.value, "index": ALL}, "value"),
            State({"type": DashIdPlateDesigner.QC_SAMPLE_ABBREV_INPUT_TYPE.value, "index": ALL}, "value"),

            State(DashIdStore.CURRENT_PLATE_DESIGN.value, "data")
        ]
    )
    def add_alternating_qc_round(n_clicks, end_round_rows, qc_sample_names, qc_sample_abbrev, plate_json):

        if None in qc_sample_abbrev or len(qc_sample_abbrev) == 0:
            return no_update, True

        options = [abbrev for abbrev in qc_sample_abbrev]
        options = options if options is not None else []

        if not callback_context.triggered:
            raise PreventUpdate
        
        default_position = len(end_round_rows) + 1
        
        end_round_row = dbc.Row(
                            [
                                # dbc.Label(f"{len(spec_round_rows)+1}", width="auto"),
                                dbc.Col(
                                    [
                                        dbc.InputGroup(
                                            [
                                                dbc.InputGroupText("Round"),
                                                dbc.Input(
                                                    type="number",
                                                    min=1,
                                                    value=1,
                                                    id={
                                                        "type": DashIdPlateDesigner.ALTERNATE_ROUND_INPUT_TYPE.value,
                                                        "index": f"{DashIdPlateDesigner.ALTERNATE_ROUND_INDEX.value}_{n_clicks}"
                                                    },
                                                )
                                            ]
                                        )
                                    ],
                                ),
                                dbc.Col(
                                    [
                                        dcc.Dropdown(
                                            # placeholder="Abbreviation",
                                            options=options,
                                            id={
                                                "type": DashIdPlateDesigner.ALTERNATE_ROUND_SELECT_TYPE.value,
                                                "index": f"{DashIdPlateDesigner.ALTERNATE_ROUND_INDEX.value}_{n_clicks}"
                                            },

                                        ),
                                    ],
                                ),
                                dbc.Col(
                                    [
                                        dbc.Button(
                                            "Remove",
                                            color="primary",
                                            outline=True,
                                            id={
                                                "type": DashIdPlateDesigner.REMOVE_ALTERNATE_ROUND_BTN_TYPE.value,
                                                "index": f"{DashIdPlateDesigner.ALTERNATE_ROUND_INDEX.value}_{n_clicks}"
                                            }
                                        ),
                                    ],
                                    width="auto"
                                )
                            ],
                            className="mb-2"
                        )
        
        end_round_rows.append(end_round_row)

        return end_round_rows, False
    
    # Remove alternating definition
    @app.callback(
        [
            Output(DashIdPlateDesigner.ALTERNATE_ROUND_DIV.value, "children", allow_duplicate=True)
        ],
        [
            Input({"type": DashIdPlateDesigner.REMOVE_ALTERNATE_ROUND_BTN_TYPE.value, "index": ALL}, "n_clicks")
        ],
        [
            State(DashIdPlateDesigner.ALTERNATE_ROUND_DIV.value, "children")
        ],
        prevent_initial_call='initial_duplicate'
    )
    def remove_alternating_round(n_clicks_list, spec_rounds_rows):

        remove_btn_type = DashIdPlateDesigner.REMOVE_ALTERNATE_ROUND_BTN_TYPE.value

        # Check if any button was clicked
        if not any(n_clicks and n_clicks > 0 for n_clicks in n_clicks_list):
            raise PreventUpdate

        if not callback_context.triggered:
            raise PreventUpdate

        # Get the id of the button that triggered the callback
        triggered_id = callback_context.triggered_id

        # If the callback was triggered by a pattern-matching callback, triggered_id is a dictionary
        if not isinstance(triggered_id, dict):
            raise PreventUpdate

        new_spec_rounds_rows = remove_dynamic_row(triggered_id, spec_rounds_rows, remove_btn_type)

        return [new_spec_rounds_rows]


    # Plate viz options changed
    @app.callback(
        [
            Output(DashIdPlateDesigner.PREVIEW_GRAPH.value, "figure", allow_duplicate=True)
        ],
        [
            Input(DashIdPlateDesigner.COLOR_SELECT.value, "value"),
            Input(DashIdPlateDesigner.LABEL_SELECT.value, "value"),
        ],
        prevent_initial_call="initial_duplicate"
    )
    def update_plate_viz(color_option, label_option):
        raise PreventUpdate
    
    # ##################################################################################################
    # CREATE/PREVIEW clicked
    @app.callback(
        [
            Output(DashIdPlateDesigner.PREVIEW_GRAPH.value, "figure", allow_duplicate=True),
            Output(DashIdPlateDesigner.PREVIEW_DAG.value, "rowData", allow_duplicate=True),

            Output(DashIdPlateDesigner.SIZE_SELECT.value, "value"),
            Output(DashIdPlateDesigner.SIZE_SELECT.value, "options"),

            Output(DashIdPlateDesigner.N_ROWS_SELECT.value, "value"),
            Output(DashIdPlateDesigner.N_COLUMNS_SELECT.value, "value"),

            Output(DashIdStore.PLATE_LIBRARY.value, "data"),

            Output(DashIdPlateLib.PLATE_LIB_DAG.value, "rowData", allow_duplicate=True),

            Output(DashIdStore.CURRENT_PLATE_DESIGN.value, "data"),
        ],
        [
            # Preview button
            Input(DashIdPlateDesigner.PREVIEW_BTN.value, "n_clicks"),

            # Create button
            Input(DashIdPlateDesigner.CREATE_BTN.value, "n_clicks"),

            Input(DashIdPlateDesigner.SIZE_SELECT.value, "value"),
            Input(DashIdPlateDesigner.N_ROWS_SELECT.value, "value"),
            Input(DashIdPlateDesigner.N_COLUMNS_SELECT.value, "value"),

            Input(DashIdMisc.DARK_MODE_SWITCH.value, "value"),

            # Plate viz settings
            Input(DashIdPlateDesigner.COLOR_SELECT.value, "value"),
            Input(DashIdPlateDesigner.LABEL_SELECT.value, "value"),
        ],
        [
            # QC frequency
            State(DashIdPlateDesigner.QC_SAMPLE_SPACING_INPUT.value, "value"),
            State(DashIdPlateDesigner.START_WITH_QC_CHECKBOX.value, "value"),

            # Plate size select options
            State(DashIdPlateDesigner.SIZE_SELECT.value, "options"),

            # QC sample names
            State({"type": DashIdPlateDesigner.QC_SAMPLE_NAME_INPUT_TYPE.value, "index": ALL}, "value"),
            State({"type": DashIdPlateDesigner.QC_SAMPLE_ABBREV_INPUT_TYPE.value, "index": ALL}, "value"),

            # Start pattern
            State({"type": DashIdPlateDesigner.START_ROUND_POS_SELECT_TYPE.value, "index": ALL}, "value"),
            State({"type": DashIdPlateDesigner.START_ROUND_SELECT_TYPE.value, "index": ALL}, "value"),

            # End
            State({"type": DashIdPlateDesigner.END_ROUND_POS_SELECT_TYPE.value, "index": ALL}, "value"),
            State({"type": DashIdPlateDesigner.END_ROUND_SELECT_TYPE.value, "index": ALL}, "value"),

            # Specific
            State({"type": DashIdPlateDesigner.SPEC_ROUND_INPUT_TYPE.value, "index": ALL}, "value"),
            State({"type": DashIdPlateDesigner.SPEC_ROUND_SELECT_TYPE.value, "index": ALL}, "value"),

            # Alternate
            State({"type": DashIdPlateDesigner.ALTERNATE_ROUND_INPUT_TYPE.value, "index": ALL}, "value"),
            State({"type": DashIdPlateDesigner.ALTERNATE_ROUND_SELECT_TYPE.value, "index": ALL}, "value"),

            # Store
            State(DashIdStore.PLATE_LIBRARY.value, "data"),

            # Plate name
            State(DashIdPlateDesigner.NAME_INPUT.value, "value"),

            # Url
            State(DashIdMisc.LOCATION.value, "pathname")
        ],
        prevent_initial_call="initial_duplicate"
    )
    def create_preview_plate(n_clicks_preview, n_clicks_create, plate_size, n_rows, n_cols, dark_mode_off, 
                     color_option, label_option,
                     qc_sample_spacing, start_with_qc,
                     plate_size_options,
                     qc_sample_names, qc_sample_abbrev,
                     start_positions, start_samples,
                     end_positions, end_samples,
                     spec_round_number, spec_round_samples,
                     alt_round_number, alt_round_samples,
                     plate_lib_store,
                     plate_name,
                     url
                     ):
        
        def group_values(group_index, group_samples):
            grouped_values = {}
            for index, value in zip(group_index, group_samples):
                if index in grouped_values:
                    grouped_values[index].append(value)
                else:
                    grouped_values[index] = [value]

            # Group values based on the group_indices
            for index, value in zip(spec_round_number, spec_round_samples):
                if index in grouped_values:
                    grouped_values[index].append(value)
                else:
                    grouped_values[index] = [value]

            return grouped_values
        
        if (url != "/plates"):
            raise PreventUpdate

        template = "journal" if dark_mode_off else "journal_dark"

        if callback_context.triggered_id == DashIdPlateDesigner.SIZE_SELECT.value:
            match plate_size:
                case "6":
                    n_rows = 2
                    n_cols = 3
                case "12":
                    n_rows = 3
                    n_cols = 4
                case "24":
                    n_rows = 4
                    n_cols = 6
                case "48":
                    n_rows = 6
                    n_cols = 8
                case "96":
                    n_rows = 8
                    n_cols = 12
                case "384":
                    n_rows = 16
                    n_cols = 24
                case _:
                    raise PreventUpdate
                
        if (callback_context.triggered_id == DashIdPlateDesigner.N_ROWS_SELECT.value) or (callback_context.triggered_id == DashIdPlateDesigner.N_COLUMNS_SELECT.value):

            if n_rows/2 * 3 == n_cols:
                plate_size = f"{n_rows*n_cols}"
            else:
                plate_size = f"{n_rows*n_cols} (custom)"
        
        plate_size_options[-1] = plate_size

        # if not n_clicks:
        #     raise PreventUpdate
        
        # QC SAMPLE ABBREVIATION-NAME MAPPING
        qc_config_samplenames = {abbrev: name for abbrev,name in zip(qc_sample_abbrev, qc_sample_names)}

        # SETUP QC PATTERNS
        qc_patterns = {}
    
        # START
        qc_pattern_start = [start_samples[int(index)-1] for index in start_positions]
        qc_patterns["start"] = qc_pattern_start

        # END
        qc_pattern_end = [end_samples[int(index)-1] for index in end_positions[::-1]]
        qc_patterns["end"] = qc_pattern_end

        # SPECIFIC
        grouped_values = group_values(spec_round_number, spec_round_samples)

        for key in grouped_values.keys():
            round_name = f"round_{key}"
            qc_patterns[round_name] = grouped_values[key]
            # print(f"{round_name}: {grouped_values[key]}")

        # ALTERNATING
        grouped_values = group_values(alt_round_number, alt_round_samples)
        alternating_pattern = []
        for key in grouped_values.keys():
            alternating_pattern.append(grouped_values[key])

        if alternating_pattern:
            qc_patterns["alternating"] = alternating_pattern

        # REPEAT
        

        # qc_pattern_specific =
        print(qc_patterns)

        QC_config_dict =  {
            'QC': {
                'start_with_QC_round': start_with_qc,
                'run_QC_after_n_specimens': qc_sample_spacing,
                'names': qc_config_samplenames,
                'patterns': qc_patterns
            }
        }

        if qc_patterns:
            plate = PlateFactory.create_plate(plate_dim=[n_rows, n_cols], QC_config = QC_config_dict)
        else:
            plate = PlateFactory.create_plate(plate_dim=[n_rows, n_cols],)
        
        fig = plate.as_plotly_figure(
            annotation_metadata_key=label_option,
            color_metadata_key=color_option,
            theme=template,
            show_grid=False,
            fig_height=500,
            fig_width=750,
            dark_mode=not dark_mode_off,
            colormap_discrete="Set2"
        )

        plate_df_dict = plate.as_dataframe().to_dict("records")

        if callback_context.triggered_id == DashIdPlateDesigner.CREATE_BTN.value:
            if not plate_lib_store:
                plate_lib_store = {}
            
            # --- Add plate to store ---
            plate_lib_store[plate_name] = plate.as_dict()

            # -- Update plate select options
            plate_select_options = [name for name in plate_lib_store.keys()]

            # --- Update library table ---
            # Create plate objects from store
            plates = [PlateFactory.dict_to_plate(plate_dict) for name, plate_dict in plate_lib_store.items()]
            # Create dataframe for lib table
            summaries = [plate.summary_dict() for plate in plates]
            df = pd.DataFrame(summaries)
            df.insert(0, value=list(plate_lib_store.keys()), column="plate_name")

            return fig, plate_df_dict, plate_size, plate_size_options, n_rows, n_cols, plate_lib_store, df.to_dict("records"), plate.as_dict()

        return fig, plate_df_dict, plate_size, plate_size_options, n_rows, n_cols, no_update, no_update, plate.as_dict()

    # Create plate -> Show info alert
    @app.callback(
        [
            Output(DashIdPlateDesigner.ADDED_TO_LIB_ALERT.value, "is_open"),
            Output(DashIdPlateDesigner.ADDED_TO_LIB_ALERT.value, "children"),
        ],
        [
            Input(DashIdPlateDesigner.CREATE_BTN.value, "n_clicks")
        ],
        [
            State(DashIdPlateDesigner.NAME_INPUT.value, "value"),
        ]
    )
    def add_plate_to_lib(n_clicks, plate_name):

        if n_clicks:
            emphasized_plate_name = html.Span(plate_name, style={'font-weight': 'bold'})
            message = [emphasized_plate_name, ' added to library.']

            return True, message

        raise PreventUpdate
    
    # Remove plate from library
    @app.callback(
        [
            Output(DashIdStore.PLATE_LIBRARY.value, "data", allow_duplicate=True),
            Output(DashIdPlateLib.PLATE_LIB_DAG.value, "rowData", allow_duplicate=True),
        ],
        [
            Input(DashIdPlateLib.REMOVE_PLATE_BTN.value, "n_clicks")
        ],
        [
            State(DashIdStore.PLATE_LIBRARY.value, "data"),
            State(DashIdPlateLib.PLATE_LIB_DAG.value, "selectedRows"),
            State(DashIdPlateLib.PLATE_LIB_DAG.value, "rowData"),
        ],
        prevent_initial_call="initial_duplicate"
    )
    def remove_plate_lib(n_clicks, plate_lib_store, selected, row_data):
        if not n_clicks or not selected:
            raise PreventUpdate

        try:
            # Remove the selected plate from the store
            selected_plate_name = selected[0]["plate_name"]
            if selected_plate_name in plate_lib_store:
                del plate_lib_store[selected_plate_name]

            # Update rowData for AG Grid, removing the selected row
            updated_row_data = [row for row in row_data if row['plate_name'] != selected_plate_name]

            return plate_lib_store, updated_row_data

        except Exception as e:
            print(e)
            # Log the error or handle it as necessary
            print(f"Error removing plate: {e}")
            raise PreventUpdate
        

