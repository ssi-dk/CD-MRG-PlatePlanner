import pytest
from plate_layout.plate import Plate
from plate_layout.plate import Well

#@pytest.fixture

# Well class

def test_should_create_well():
    assert Well("A0", (0,0)) is not None


# Plate
def test_should_create_plate_from_tuple():
    plate_96 = Plate((8,12))
    
    assert len(plate_96.rows) == 8
    assert len(plate_96.columns) == 12
    
def test_should_create_plate_from_key_word_args():
    plate_96 = Plate(rows=8, columns=12)
    
    assert len(plate_96.rows) == 8
    assert len(plate_96.columns) == 12
    
def test_should_build_coordinate_system():
    plate_96 = Plate((8,12))
    
    assert plate_96.coordinate_system[0] == (7,0)
    assert plate_96.coordinate_system[11] == (7,11)
    assert plate_96.coordinate_system[84] == (0,0)
    assert plate_96.coordinate_system[95] == (0,11)
    
    
def test_should_create_wells_in_plate():
    plate_96 = Plate((8,12))
    
    assert len(plate_96.wells) == 96
    
    

def test_should_return_plate_length():
    plate_96 = Plate((8,12))
    assert len(plate_96) == 96 
