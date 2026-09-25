"""AT-2 storage, AT-3 bridge."""
import os

import pytest

from src.foundation import bridge
from src.io.storage import Storage


def test_roundtrip_local_and_file(tmp_path):
    st = Storage({}, 1, str(tmp_path / "work"))
    for uri in (str(tmp_path / "a" / "x.txt"), f"file://{tmp_path}/b/y.txt"):
        st.write_text(uri, "hello")
        assert st.read_text(uri) == "hello" and st.exists(uri)
    assert st.glob(str(tmp_path), "a/*.txt") == [str(tmp_path / "a" / "x.txt")]


def test_upload_retry_on_injected_failure(tmp_path):
    st = Storage({}, 2, str(tmp_path / "work"), retry_sleep_s=0)
    st.fs = lambda uri: (__import__("fsspec").filesystem("memory"), uri.split("://", 1)[1])
    st.is_local = staticmethod(lambda uri: False)
    calls = {"n": 0}
    real_put = Storage._put

    def flaky(self, fs, local, remote):
        calls["n"] += 1
        if calls["n"] == 1:
            raise IOError("injected")
        real_put(self, fs, local, remote)
    Storage._put = flaky
    try:
        st.write_bytes("memory://bucket/obj.bin", b"12345")
    finally:
        Storage._put = real_put
    assert calls["n"] == 2
    import fsspec
    assert fsspec.filesystem("memory").cat("bucket/obj.bin") == b"12345"


def test_upload_gives_up(tmp_path):
    st = Storage({}, 1, str(tmp_path / "work"), retry_sleep_s=0)
    st.fs = lambda uri: (__import__("fsspec").filesystem("memory"), uri.split("://", 1)[1])
    st.is_local = staticmethod(lambda uri: False)
    st._put = lambda fs, l, r: (_ for _ in ()).throw(IOError("down"))
    with pytest.raises(IOError, match="after 2 attempts"):
        st.write_bytes("memory://bucket/z.bin", b"1")


def test_bridge_names_and_forbidden():
    for m, names in bridge.ALLOWED.items():
        mod = getattr(bridge, f"fl_{m}")
        assert all(hasattr(mod, n) for n in names)
    for name in ("config", "data", "train", "scheduler"):
        with pytest.raises(bridge.BridgeError):
            bridge.import_foundation_module(name)
