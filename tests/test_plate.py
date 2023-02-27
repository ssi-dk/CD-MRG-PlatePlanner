import pytest
from plate_layout.plate import Plate
from plate_layout.plate import Well

#@pytest.fixture

# Well class

def test_should_create_well():
    well =  Well("A0", (0,0)) 
    
    assert well.name == "A0"
    assert well.coordinate == (0,0)
    

def test_should_set_metadata():
    pass


# Plate
def test_should_create_plate_from_tuple():
    plate_96 = Plate((8,12))
    
    assert plate_96.plate_id == 1
    assert len(plate_96.rows) == 8
    assert len(plate_96.columns) == 12
    
    
def test_should_create_plate_from_key_word_args():
    plate_96 = Plate(rows=8, columns=12)
    
    assert len(plate_96.rows) == 8
    assert len(plate_96.columns) == 12
    
def test_should_create_plate_id_from_key_word_args():
    plate_96 = Plate(rows=8, columns=12, plate_id=5)
    assert plate_96.plate_id == 5
    
    
def test_should_create_plate_index_coordinates():
    plate_96 = Plate((8,12))
    
    plate_96._coordinates = Plate.create_index_coordinates(plate_96.rows, plate_96.columns)
    
    assert plate_96._coordinates[0] == (7,0)
    assert plate_96._coordinates[11] == (7,11)
    assert plate_96._coordinates[84] == (0,0)
    assert plate_96._coordinates[95] == (0,11)
   
    
def test_should_create_plate_alphanumerical_coordinates():
    plate_96 = Plate((8,12))
    
    plate_96._alphanumerical_coordinates = Plate.create_alphanumerical_coordinates(plate_96.rows, plate_96.columns)
    
    assert plate_96._alphanumerical_coordinates[0] == "A_1"
    assert plate_96._alphanumerical_coordinates[11] == "A_12"
    assert plate_96._alphanumerical_coordinates[84] == "H_1"
    assert plate_96._alphanumerical_coordinates[95] == "H_12"
    

@pytest.mark.dependancy(depends=[test_should_create_well,
                                 test_should_create_plate_index_coordinates, 
                                 test_should_create_plate_alphanumerical_coordinates])
def test_should_create_well_objects_in_plate():
    plate_96 = Plate((8,12))
    
    plate_96._coordinates = Plate.create_index_coordinates(plate_96.rows, plate_96.columns)
    plate_96._alphanumerical_coordinates =Plate.create_alphanumerical_coordinates(plate_96.rows, plate_96.columns)
    Plate.define_empty_wells(plate_96)
    
    well = plate_96.wells[0]
    assert well.name == "A_1"
    assert well.metadata["index"] == 0
    assert well.coordinate == (7,0)
    
    well = plate_96.wells[95]
    assert well.name == "H_12"
    assert well.metadata["index"]== 95
    assert well.coordinate == (0,11)
    

def test_should_return_plate_capacity():
    plate_96 = Plate((8,12))
    assert plate_96.capacity == 96 
    

@pytest.mark.dependancy(depends=[test_should_create_well_objects_in_plate])   
def test_should_return_length_of_plate(): 
    plate_96 = Plate((8,12))
    assert len(plate_96) == 96
      
def test_should_get_well_by_index():
    plate_96 = Plate((8,12))
    
    well = plate_96.wells[0]
    
    assert plate_96[0].name == well.name 
    
def test_should_get_well_by_name():
    plate_96 = Plate((8,12))
    
    assert plate_96['A_1'].name ==  plate_96.wells[0].name
    assert plate_96['A_1'].coordinate == plate_96.wells[0].coordinate
    
    assert plate_96['B_12'].name ==  plate_96.wells[23].name
    assert plate_96['B_12'].coordinate == plate_96.wells[23].coordinate
    
    assert plate_96['H_1'].name ==  plate_96.wells[84].name
    assert plate_96['H_1'].coordinate == plate_96.wells[84].coordinate
    
    assert plate_96['H_12'].name ==  plate_96.wells[95].name
    assert plate_96['H_12'].coordinate == plate_96.wells[95].coordinate


def test_should_get_well_by_coordinate():
    plate_96 = Plate((8,12))
    
    assert plate_96[(7,0)].name == plate_96[0].name
    assert plate_96[(7,0)].coordinate == plate_96[0].coordinate
    
    assert plate_96[(6,11)].name == plate_96[23].name
    assert plate_96[(6,11)].coordinate == plate_96[23].coordinate
    
    assert plate_96[(0,0)].name == plate_96[84].name
    assert plate_96[(0,0)].coordinate == plate_96[84].coordinate
    
    assert plate_96[(0,11)].name == plate_96[95].name
    assert plate_96[(0,11)].coordinate == plate_96[95].coordinate

def test_should_get_well_by_key():
    plate_96 = Plate((8,12))
    
    assert plate_96[0].name == plate_96[0].name
    assert plate_96[0].coordinate == plate_96[0].coordinate
    
    assert plate_96[23].name == plate_96[23].name
    assert plate_96[23].coordinate == plate_96[23].coordinate
    
    assert plate_96[84].name == plate_96[84].name
    assert plate_96[84].coordinate == plate_96[84].coordinate
    
    assert plate_96[95].name == plate_96[95].name
    assert plate_96[95].coordinate == plate_96[95].coordinate


def test_should_set_well_by_key():
    assert 1 == 2, "REMINDER: This test is not written yet" 