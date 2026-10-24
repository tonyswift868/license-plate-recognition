from dataclasses import replace
import pytest
from lpr_tools.data.manifest import ManifestRow, read_manifest, write_manifest, validate_manifest, parse_quad


def row(path="img.png",**kwargs):
    return ManifestRow(path,"陕A12345","blue","clear","constructed",**kwargs)


def test_roundtrip(tmp_path):
    p=tmp_path/"清单.csv";(tmp_path/"图像.png").touch()
    rows=[row("图像.png",quad=((1.,2.),(3.,2.),(3.,4.),(1.,4.)),note="a,b"),
          ManifestRow("np.png","","unknown","np","constructed")]
    write_manifest(p,rows)
    assert read_manifest(p)==rows
    assert not p.read_bytes().startswith(b"\xef\xbb\xbf")
    validate_manifest(rows[:1],base_dir=tmp_path,check_files=True)
    with pytest.raises(ValueError,match="missing"):read_manifest(p,check_files=True)


@pytest.mark.parametrize("change",[{"category":"bad"},{"plate_color":"pink"},{"plate_no":"陕AI2345"},
    {"path":""},{"category":"np"},{"quad":((1,2),)}, {"quad":((float('nan'),2),)*4}])
def test_bad_rows(change):
    with pytest.raises(ValueError):validate_manifest([replace(row(),**change)])


def test_duplicate_normalization(tmp_path):
    with pytest.raises(ValueError,match="duplicate"):
        write_manifest(tmp_path/"m.csv",[row(),row("./img.png")])
    assert parse_quad("") is None
    with pytest.raises(ValueError):parse_quad("1 2 3 4")
    p=tmp_path/"bad.csv";p.write_text("bad,header\n",encoding="utf-8")
    with pytest.raises(ValueError,match="columns"):read_manifest(p)
