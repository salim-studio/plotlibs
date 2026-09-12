"""Rebrand tests: plotlibs name, legacy shim, brand identity."""
import warnings


def test_package_name_and_version():
    import plotlibs as pl
    assert pl.__version__ == "0.3.0"
    assert pl.__brand__ == "plotlibs"


def test_legacy_shim_still_works():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        import plotlib as old
        import plotlibs as new
        assert old.__version__ == new.__version__
        old.close()
        old.plot([1, 2, 3], [1, 4, 9])
        assert len(old.gca().lines) == 1
        old.close()


def test_brand_palette_and_theme():
    from plotlibs.colors import BRAND, CYCLE, to_rgb
    assert BRAND["primary"] == (79, 70, 229)
    assert BRAND["accent"] == (6, 182, 212)
    assert CYCLE[0] == (79, 70, 229)  # signature default, not old tableau blue
    assert to_rgb(None) == (79, 70, 229)
    import plotlibs as pl
    assert "plotlibs" in pl.style.available()
    assert "plotlibs-dark" in pl.style.available()
    pl.style.use("plotlibs")
    pl.close()
    pl.plot([1, 2], [3, 4])
    assert pl.gca().lines[0]["color"] == (79, 70, 229)
    pl.close()


def test_brand_assets_exist():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    assert (root / "assets" / "logo.svg").exists()
    assert (root / "assets" / "banner.svg").exists()
    assert (root / "BRANDING.md").exists()
    assert (root / "LICENSE").exists()
