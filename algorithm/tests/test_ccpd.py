from pathlib import Path
import pytest
from lpr_tools.data.ccpd import (parse_ccpd_name, build_index, read_index, PROVINCES_CCPD, ALPHABETS, ADS, main)

EXAMPLE="025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg"


def named(labels):
    fields=EXAMPLE.split("-");fields[4]="_".join(map(str,labels));return "-".join(fields)


def test_appendix():
    r=parse_ccpd_name(EXAMPLE)
    assert r.plate_no=="皖AY339S" and r.layout==7
    assert r.quad==[(154,383),(363,402),(386,473),(177,454)]
    assert r.bbox==(154,383,232,90)
    assert (r.tilt_h,r.tilt_v,r.brightness,r.blurriness,r.area)==(95,113,37,15,"025")
    assert r.skip_reason==""
    assert tuple(map(len,(PROVINCES_CCPD,ALPHABETS,ADS)))==(34,25,35)


def test_green_and_skips():
    r=parse_ccpd_name(named([26,0,3,25,26,27,28,29]))
    assert r.plate_no=="陕AD12345" and r.layout==8 and not r.skip_reason
    for labels in ([32,0,0,0,0,0,0],[31,0,0,0,0,0,0]):
        assert parse_ccpd_name(named(labels)).skip_reason=="unsupported_province"
    for labels in ([33,0,0,0,0,0,0],[0,24,0,0,0,0,0],[0,0,34,0,0,0,0]):
        assert parse_ccpd_name(named(labels)).skip_reason=="placeholder"
    assert parse_ccpd_name(named([0,0,0,0,0,0,0,0])).skip_reason=="invalid_format"


@pytest.mark.parametrize("name",["bad.jpg",EXAMPLE.replace("0_0_22","99_0_22"),
    EXAMPLE.replace("95_113","95"),EXAMPLE.replace("386&473_177&454_154&383_363&402","386&473"),
    EXAMPLE.replace("154&383_386&473","154&383_154&383"),named([0,0,0])])
def test_malformed(name):
    with pytest.raises(ValueError):parse_ccpd_name(name)


def touch(root,path):
    p=root/path;p.parent.mkdir(parents=True,exist_ok=True);p.touch();return p


def test_build_two_datasets(tmp_path,capsys):
    raw=tmp_path/"raw"
    for subset in ("ccpd_base","ccpd_db","ccpd_tilt","ccpd_blur","ccpd_weather"):
        touch(raw,f"ccpd2019/{subset}/{EXAMPLE}")
    touch(raw,"ccpd2019/ccpd_np/0001.jpg")
    for split in ("train","val","test"):
        p=touch(raw,f"ccpd2020/ccpd_green/{split}/{named([26,0,3,25,26,27,28,29])}")
    for file,subset in (("train","ccpd_base"),("val","ccpd_db"),("test","ccpd_tilt"),("ccpd_blur","ccpd_blur")):
        p=touch(raw,f"ccpd2019/splits/{file}.txt");p.write_text(f"{subset}/{EXAMPLE}\n",encoding="utf-8")
    target=tmp_path/"index.csv"
    rows=build_index(raw,target)
    assert len(rows)==9 and read_index(target)==rows
    assert not target.read_bytes().startswith(b"\xef\xbb\xbf")
    base=next(r for r in rows if r["subset"]=="ccpd_base")
    assert base["split"]=="train" and base["quad"]=="154 383 363 402 386 473 177 454"
    assert base["bbox"]=="154 383 232 90" and Path(base["path"]).exists()
    assert next(r for r in rows if r["subset"]=="ccpd_np")["has_plate"]=="0"
    assert {r["plate_color"] for r in rows if r["dataset"]=="ccpd2020"}=={"green"}
    assert "skip_reasons" in capsys.readouterr().out
    assert main(["--raw",str(raw),"--out",str(target)])==0


def test_split_conflict(tmp_path):
    raw=tmp_path/"raw";touch(raw,f"ccpd2019/ccpd_base/{EXAMPLE}")
    for split in ("train","val"):
        touch(raw,f"ccpd2019/splits/{split}.txt").write_text(EXAMPLE,encoding="utf-8")
    with pytest.raises(ValueError,match="conflicting"):build_index(raw,tmp_path/"i.csv")
