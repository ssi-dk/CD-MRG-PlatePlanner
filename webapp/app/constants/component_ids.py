from enum import Enum

class DashIdStore(Enum):
    CURRENT_PLATE_DESIGN = "current-plate-design-store"
    PLATE_LIBRARY = "plate-library-store"
    SAMPLE_LIST = "sample-list-store"
    PLATE_LAYOUTS = "plate-layout-store"
    STUDY = "study-store"

class DashIdMisc(Enum):
    DARK_MODE_SWITCH = "dark-mode-switch"
    NAVBAR = "navbar-simple"
    LOCATION = "current-page"

class DashIdPlateDesigner(Enum):
    PREVIEW_GRAPH = "plate-preview-graph"

    PREVIEW_DAG = "plate-preview-dag"

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

    REMOVE_PLATE_BTN = "remove-plate-btn"


class DashIdStudy(Enum):
    SAMPLE_LIST_DAG = "sample-list-dag"
    SAMPLE_LIST_DIV = "sample-list-div"
    SAMPLE_LIST_FILENAME_LABEL = "sample-list-filename-label"
    SAMPLE_LIST_UPLOAD = "sample-list-upload"

    PLATE_LAYOUT_DAG = "plate-layout-dag"

    EXAMPLE_LIST_BTN = "example-list-btn"

    PLATE_SELECT = "plate-select"

    SAMPLES_PER_PLATE_SELECT = "samples-per-plate-select"

    RANDOMIZE_SELECT = "randomize-checkbox"

    GROUP_COL_SELECT = "group-col-select"
    GROUP_COL_SELECT_COLLAPSE = "group-col-select-collapse"
    GROUP_COL_SELECT_LABEL = "group-col-select-label-collapse"

    ALLOW_GROUP_SPLIT_SWITCH = "allow-group-split-switch"
    ALLOW_GROUP_SPLIT_COLLAPSE = "allow-group-split-collapse"

    DISTRIBUTE_SAMPLES_BTN = "distribute-samples-plates"

    INPUT_ALERT = "distribute-input-alert"

    PLATE_LAYOUT_SELECT_DAG = "plate-layout-select-"

    SAMPLE_LIST_TAB = "sample-list-tab"
    PLATE_ASSIGN_TAB = "plate-assign-tab"
    PLATE_LAYOUT_TAB = "plate-layout-tab"
    PLATE_EXPORT_TAB = "plate-export-tab"

    PLATE_LAYOUT_GRAPH = "plate-layout-graph"
    PLATE_LAYOUT_TABLE_DIV = "plate-layout-table-div"

    PLATE_ATTRIBUTE_DISTR_GRAPH = "plate-attribute-distribution-graph"

    STUDY_TABS = "study-tabs"

    # Plate layout label
    FIG_PLATE_LABEL = "pl-figure-name-label"
    TABLE_PLATE_LABEL = "pl-table-name-label"

    # Plate layout fig settings
    PL_FIG_COLOR_SELECT = "pl-fig-color-wells-select"
    PL_FIG_LABEL_SELECT = "pl-fig-label-wells-select"

    # Export
    LIST_EXPORT_FIELDS_SELECT = "list-export-fields-select"
    FIG_EXPORT_FORMAT_SELECT = "fig-export-format-select"
    FIG_EXPORT_COLOR_SELECT = "fig-export-color-select"
    FIG_EXPORT_LABEL_SELECT = "fig-export-label-select"

    DOWNLOAD_LISTS_BTN = "download-lists-button"
    LIST_DOWNLOAD = "list-download"

    DOWNLOAD_FIGS_BTN = "download-figures-button"
    FIGS_DOWNLOAD = "figs-download"

