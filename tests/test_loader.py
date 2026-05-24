"""
tests/test_loader.py
====================
Unit tests for neurosim.loader.

Design: All tests run on synthetic numpy arrays only.
No NIfTI files, no BIDS directories, no internet access required.
The neuroimaging dependencies (nibabel, nilearn, pybids) are tested
for graceful ImportError handling, not for functional correctness
(that requires integration tests on real data, tracked separately).
"""

import warnings

import numpy as np
import pytest

from neurosim.loader import from_arrays, load_connectome


# from_arrays — the dependency-free entry point

class TestFromArrays:

    def test_basic_valid_input(self):
        X = np.random.default_rng(0).normal(0, 1, (50, 200))
        result = from_arrays(X)
        assert result["X"].shape == (50, 200)
        assert result["N"] == 50
        assert result["T"] == 200
        assert result["SC"] is None

    def test_with_sc(self):
        rng = np.random.default_rng(1)
        X = rng.normal(0, 1, (20, 100))
        SC_raw = np.abs(rng.normal(0, 1, (20, 20)))
        SC_raw = (SC_raw + SC_raw.T) / 2
        np.fill_diagonal(SC_raw, 0)
        result = from_arrays(X, SC=SC_raw)
        assert result["SC"].shape == (20, 20)
        # Should enforce symmetry and zero diagonal
        assert np.allclose(result["SC"], result["SC"].T, atol=1e-10)
        assert np.allclose(np.diag(result["SC"]), 0.0)

    def test_subject_id_stored(self):
        X = np.random.default_rng(2).normal(0, 1, (10, 50))
        result = from_arrays(X, subject_id="sub-01")
        assert result["subject_id"] == "sub-01"

    def test_nan_raises(self):
        X = np.random.default_rng(3).normal(0, 1, (10, 50))
        X[3, 7] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            from_arrays(X)

    def test_inf_raises(self):
        X = np.random.default_rng(4).normal(0, 1, (10, 50))
        X[0, 0] = np.inf
        with pytest.raises(ValueError, match="infinite"):
            from_arrays(X)

    def test_wrong_ndim_raises(self):
        X = np.ones((10,))  # 1D — wrong
        with pytest.raises(ValueError, match="2D"):
            from_arrays(X)

    def test_transposed_warning(self):
        # T > N is fine, but N > T should warn (likely transposed)
        X = np.random.default_rng(5).normal(0, 1, (200, 20))  # 200 regions, 20 timepoints
        with pytest.warns(UserWarning, match="T < N"):
            from_arrays(X)

    def test_sc_shape_mismatch_raises(self):
        X = np.random.default_rng(6).normal(0, 1, (20, 100))
        SC_wrong = np.ones((30, 30))  # wrong size
        with pytest.raises(ValueError, match="SC shape"):
            from_arrays(X, SC=SC_wrong)

    def test_asymmetric_sc_warns_and_fixes(self):
        rng = np.random.default_rng(7)
        X = rng.normal(0, 1, (10, 50))
        SC_asym = np.abs(rng.normal(0, 1, (10, 10)))  # not symmetric
        with pytest.warns(UserWarning, match="symmetric"):
            result = from_arrays(X, SC=SC_asym)
        assert np.allclose(result["SC"], result["SC"].T, atol=1e-10)

    def test_validate_false_skips_checks(self):
        # With validate=False, NaN should not raise
        X = np.full((5, 20), np.nan)
        result = from_arrays(X, validate=False)  # should not raise
        assert result["X"].shape == (5, 20)

    def test_float_conversion(self):
        X = np.ones((10, 50), dtype=np.int32)
        result = from_arrays(X)
        assert result["X"].dtype == float


# load_connectome — file-based loading

class TestLoadConnectome:

    def test_npy_loading(self, tmp_path):
        SC = np.abs(np.random.default_rng(10).normal(0, 1, (20, 20)))
        SC = (SC + SC.T) / 2
        np.fill_diagonal(SC, 0)
        path = tmp_path / "SC.npy"
        np.save(str(path), SC)
        loaded = load_connectome(path)
        assert loaded.shape == (20, 20)
        assert np.allclose(loaded, loaded.T, atol=1e-10)
        assert np.allclose(np.diag(loaded), 0.0)

    def test_csv_loading(self, tmp_path):
        rng = np.random.default_rng(11)
        SC = np.abs(rng.normal(0, 1, (10, 10)))
        SC = (SC + SC.T) / 2
        np.fill_diagonal(SC, 0)
        path = tmp_path / "SC.csv"
        np.savetxt(str(path), SC, delimiter=",")
        loaded = load_connectome(path)
        assert loaded.shape == (10, 10)
        assert np.allclose(loaded, SC, atol=1e-6)

    def test_npz_loading(self, tmp_path):
        SC = np.abs(np.random.default_rng(12).normal(0, 1, (15, 15)))
        SC = (SC + SC.T) / 2
        np.fill_diagonal(SC, 0)
        path = tmp_path / "SC.npz"
        np.savez(str(path), SC=SC)
        loaded = load_connectome(path)
        assert loaded.shape == (15, 15)

    def test_nonexistent_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_connectome(tmp_path / "nonexistent.npy")

    def test_non_square_raises(self, tmp_path):
        SC_nonsquare = np.ones((10, 15))
        path = tmp_path / "bad.npy"
        np.save(str(path), SC_nonsquare)
        with pytest.raises(ValueError, match="square"):
            load_connectome(path)

    def test_normalise_flag(self, tmp_path):
        SC = np.abs(np.random.default_rng(13).normal(0, 5, (10, 10)))
        SC = (SC + SC.T) / 2
        np.fill_diagonal(SC, 0)
        path = tmp_path / "SC.npy"
        np.save(str(path), SC)
        loaded = load_connectome(path, normalise=True)
        assert loaded.max() <= 1.0 + 1e-10

    def test_threshold_pct(self, tmp_path):
        rng = np.random.default_rng(14)
        SC = np.abs(rng.normal(0, 1, (20, 20)))
        SC = (SC + SC.T) / 2
        np.fill_diagonal(SC, 0)
        path = tmp_path / "SC.npy"
        np.save(str(path), SC)
        loaded_raw  = load_connectome(path)
        loaded_thresh = load_connectome(path, threshold_pct=50.0)
        # Thresholded version should have more zeros
        assert (loaded_thresh == 0).sum() > (loaded_raw == 0).sum()

    def test_symmetry_enforced(self, tmp_path):
        # Slightly asymmetric SC — should be symmetrised on load
        rng = np.random.default_rng(15)
        SC = np.abs(rng.normal(0, 1, (10, 10)))
        np.fill_diagonal(SC, 0)
        # Intentionally NOT symmetrised
        path = tmp_path / "SC.npy"
        np.save(str(path), SC)
        loaded = load_connectome(path)
        assert np.allclose(loaded, loaded.T, atol=1e-10)

    def test_zero_diagonal_enforced(self, tmp_path):
        SC = np.abs(np.random.default_rng(16).normal(0, 1, (10, 10)))
        SC = (SC + SC.T) / 2
        # Leave diagonal non-zero — should be zeroed on load
        path = tmp_path / "SC.npy"
        np.save(str(path), SC)
        loaded = load_connectome(path)
        assert np.allclose(np.diag(loaded), 0.0)



# Atlas registry validation (no download required)

class TestAtlasRegistry:

    def test_unknown_atlas_raises(self):
        pytest.importorskip("nibabel", reason="nibabel not installed")
        from neurosim.loader import load_atlas
        with pytest.raises(ValueError, match="Unknown atlas"):
            load_atlas("nonexistent_atlas_xyz")

    def test_schaefer_invalid_n_rois_raises(self):
        from neurosim.loader import load_atlas, SCHAEFER_N_ROIS
        bad_n = 999  # not in SCHAEFER_N_ROIS
        pytest.importorskip("nibabel", reason="nibabel not installed")
        pytest.importorskip("nilearn", reason="nilearn not installed")
        with pytest.raises(ValueError, match="Schaefer"):
            load_atlas("schaefer400", n_rois=bad_n)


# Graceful ImportError for optional dependencies

class TestOptionalDependencies:

    def test_from_arrays_works_without_nibabel(self):
        """from_arrays must work even if nibabel is not installed."""
        X = np.random.default_rng(20).normal(0, 1, (10, 50))
        result = from_arrays(X)  # should never import nibabel
        assert result["N"] == 10

    def test_load_connectome_works_without_nibabel(self, tmp_path):
        """load_connectome (npy) must work even without nibabel."""
        SC = np.eye(5)
        path = tmp_path / "SC.npy"
        np.save(str(path), SC)
        loaded = load_connectome(path)
        assert loaded.shape == (5, 5)
