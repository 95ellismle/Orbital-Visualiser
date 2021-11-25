'''Will test the src/IO module'''
import pytest
import tempfile
from pathlib import Path
from unittest import mock

from src import IO


@mock.patch('src.IO.is_tachyon_exe')
def test_get_tachyon_path(tachyon_func):
    '''Test finding the path to the tachyon renderer.
    This needs to work for Mac and Linux
    '''
    tachyon_func.side_effect = lambda i: i.is_file()

    # Set up a directory that mocks the linux VMD directory
    _tmp_dir = tempfile.TemporaryDirectory()
    tmp_dir = Path(_tmp_dir.name)
    tachyon_dir = tmp_dir / 'lib/tachyon'
    tachyon_dir.mkdir(parents=True)
    tachyon_path = tachyon_dir / 'tachyon_LINUXAMD64'
    tachyon_path.touch()
    vmd_dir = tmp_dir / 'LINUXAMD64'
    vmd_dir.mkdir()
    vmd_path = vmd_dir / 'vmd_LINUXAMD64'
    vmd_path.touch()

    # Test the linux get_tachyon_path
    path = IO.get_tachyon_path(vmd_path)
    assert path == tachyon_path
    del _tmp_dir

    # Set up the Mac directory
    _tmp_dir = tempfile.TemporaryDirectory()
    tmp_dir = Path(_tmp_dir.name)
    tachyon_path = tmp_dir / 'tachyon_MACOSXARM64'
    vmd_path = tmp_dir / 'vmd_MACOSXARM64'
    tachyon_path.touch()

    # Test the Mac Path
    path = IO.get_tachyon_path(vmd_path)
    assert path == tachyon_path

