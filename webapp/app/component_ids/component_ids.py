from enum import Enum

class DashIdStore(Enum):
    CURRENT_PLATE_DESIGN = "current-plate-design-store"

class DashIdMisc(Enum):
    DARK_MODE_SWITCH = "dark-mode-switch"
    NAVBAR = "navbar-simple"


class DashIdPlateDesigner(Enum):
    PREVIEW_GRAPH = "plate-preview-graph"

    NAME_INPUT = "plate-name-input"
    SIZE_SELECT = "plate-size-select"
    N_ROWS_SELECT = "plate-n-row-select"
    N_COLUMNS_SELECT = "plate-n-columns-select"

    # Plate preview
    COLOR_SELECT = "color-wells-select"
    LABEL_SELECT = "label-wells-select"

    CREATE_BTN = "create-plate-btn"
    PREVIEW_BTN = "preview-plate-btn"
    RESET_BTN = "reset-plate-btn"

    ## QC sample definitions
    QC_SAMPLE_SPACING_INPUT = "number-of-samples-between-qc-rounds-input"
    START_WITH_QC_CHECKBOX = "start-with-qc-checkbox"

    # container
    QC_SAMPLE_DIV = "qc-sample-div"

    # Buttons
    ADD_QC_SAMPLE_BTN = "add-qc-sample-btn"
    REMOVE_QC_SAMPLE_BTN_TYPE = "remove-qc-sample-btn-type"

    # Inputs
    QC_SAMPLE_TYPE = "qc-sample-type"
    QC_SAMPLE_INDEX = "qc-sample-index"

    QC_SAMPLE_NAME_INPUT_TYPE = "qc-sample-name-input-type"

    QC_SAMPLE_ABBREV_INPUT_TYPE = "qc-sample-abbrev-input-type"

    QC_SAMPLE_REMOVE_BTN_TYPE = "qc-sample-remove-btn-type"

    # start
    START_ROUND_ALERT = "start-round-alert"
    START_ROUND_DIV = "start-round-div"
    START_ROUND_INDEX = "start-round-index"

    START_ROUND_POS_SELECT_TYPE = "start-round-pos-select-type"
    START_ROUND_SELECT_TYPE = "start-round-select"

    ADD_START_ROUND_BTN = "add-start-round-btn"
    REMOVE_START_ROUND_BTN_TYPE = "add-start-round-btn-type"

    # end
    END_ROUND_ALERT = "end-round-alert"
    END_ROUND_DIV = "end-round-div"
    END_ROUND_INDEX = "end-round-index"

    END_ROUND_POS_SELECT_TYPE = "end-round-pos-select-type"
    END_ROUND_SELECT_TYPE = "end-round-select"

    ADD_END_ROUND_BTN = "add-end-round-btn"
    REMOVE_END_ROUND_BTN_TYPE = "add-end-round-btn-type"

    # spec round
    SPEC_ROUND_ALERT = "specific-round-alert"
    SPEC_ROUND_DIV = "specific-round-div"
    SPEC_ROUND_INDEX = "specific-round-index"

    SPEC_ROUND_INPUT_TYPE = "specific-round-input-type"
    SPEC_ROUND_SELECT_TYPE = "specific-round-select"

    ADD_SPEC_ROUND_BTN = "add-specific-round-btn"
    REMOVE_SPEC_ROUND_BTN_TYPE = "add-specific-round-btn-type"

    # alternate
    ALTERNATE_ROUND_ALERT = "alternating-round-alert"
    ALTERNATE_ROUND_DIV = "alternating-round-div"
    ALTERNATE_ROUND_INDEX = "alternating-round-index"

    ALTERNATE_ROUND_INPUT_TYPE = "alternating-round-input-type"
    ALTERNATE_ROUND_SELECT_TYPE = "alternating-round-select"

    ADD_ALTERNATE_ROUND_BTN = "add-alternating-round-btn"
    REMOVE_ALTERNATE_ROUND_BTN_TYPE = "add-alternating-round-btn-type"

    # 
    ADDED_TO_LIB_ALERT = "added-to-lib-alert"


class DashIdPlateLib(Enum):
    PLATE_LIB_DAG = "plate-lib-dag"




