#!/usr/bin/python3

#    Copyright Brandon Stafford
#
#    This file is part of Pysolar.
#
#    Pysolar is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation; either version 3 of the License, or
#    (at your option) any later version.
#
#    Pysolar is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License along
#    with Pysolar. If not, see <http://www.gnu.org/licenses/>.

"""
Tests for vectorised solar angle calculations.
Compares the vectorised implementation against the original scalar functions.

Note: The vectorised implementation uses a simplified algorithm similar to
get_altitude_fast/get_azimuth_fast, while the original uses the full NREL SPA
algorithm. Therefore, we expect small differences (typically < 0.05 degrees).
Tests use 1 decimal place tolerance (0.1 degrees) which is acceptable for
most solar applications.
"""

import datetime
import unittest
import numpy as np
import warnings
from pysolar import solar, radiation
from pysolar.vectorised import get_solar_angles_vector, get_radiation_direct_vector

# Suppress numpy datetime64 timezone warnings (known limitation)
warnings.filterwarnings('ignore', message='.*no explicit representation of timezones.*')


class TestVectorisedSingleValue(unittest.TestCase):
    """Test vectorised implementation with single values against original functions."""

    def test_single_datetime_greenwich(self):
        """Test single datetime at Greenwich Observatory."""
        when = datetime.datetime(2016, 12, 19, 23, 0, 0, tzinfo=datetime.timezone.utc)
        lat = 51.4826
        lon = 0.0

        # Original functions
        az_orig = solar.get_azimuth(lat, lon, when)
        alt_orig = solar.get_altitude(lat, lon, when)
        zenith_orig = 90.0 - alt_orig

        # Vectorised function
        time_np = np.datetime64(when)
        az_vec, zenith_vec = get_solar_angles_vector(lat, lon, time_np)  # scalar case - names unchanged

        # Compare results - allow 0.1 degree tolerance (different algorithms)
        self.assertAlmostEqual(az_orig, az_vec, places=1,
                             msg=f"Azimuth mismatch: original={az_orig}, vectorised={az_vec}")
        self.assertAlmostEqual(zenith_orig, zenith_vec, places=1,
                             msg=f"Zenith mismatch: original={zenith_orig}, vectorised={zenith_vec}")

    def test_single_datetime_new_zealand(self):
        """Test single datetime in New Zealand."""
        when = datetime.datetime(2016, 12, 19, 23, 0, 0, tzinfo=datetime.timezone.utc)
        lat = -43.0
        lon = 172.0

        # Original functions
        az_orig = solar.get_azimuth(lat, lon, when)
        alt_orig = solar.get_altitude(lat, lon, when)
        zenith_orig = 90.0 - alt_orig

        # Vectorised function
        time_np = np.datetime64(when)
        az_vec, zenith_vec = get_solar_angles_vector(lat, lon, time_np)  # scalar case - names unchanged

        # Compare results - allow 0.1 degree tolerance (different algorithms)
        self.assertAlmostEqual(az_orig, az_vec, places=1,
                             msg=f"Azimuth mismatch: original={az_orig}, vectorised={az_vec}")
        self.assertAlmostEqual(zenith_orig, zenith_vec, places=1,
                             msg=f"Zenith mismatch: original={zenith_orig}, vectorised={zenith_vec}")

    def test_single_datetime_scandinavia(self):
        """Test single datetime in Scandinavia."""
        when = datetime.datetime(2016, 12, 19, 23, 0, 0, tzinfo=datetime.timezone.utc)
        lat = 59.6365662
        lon = 12.5350953

        # Original functions
        az_orig = solar.get_azimuth(lat, lon, when)
        alt_orig = solar.get_altitude(lat, lon, when)
        zenith_orig = 90.0 - alt_orig

        # Vectorised function
        time_np = np.datetime64(when)
        az_vec, zenith_vec = get_solar_angles_vector(lat, lon, time_np)  # scalar case - names unchanged

        # Compare results - allow 0.1 degree tolerance (different algorithms)
        self.assertAlmostEqual(az_orig, az_vec, places=1,
                             msg=f"Azimuth mismatch: original={az_orig}, vectorised={az_vec}")
        self.assertAlmostEqual(zenith_orig, zenith_vec, places=1,
                             msg=f"Zenith mismatch: original={zenith_orig}, vectorised={zenith_vec}")

    def test_single_datetime_reda_andreas_reference(self):
        """Test against the Reda & Andreas (2005) reference values from test_solar.py."""
        # This is the reference case used in test_solar.py
        when = datetime.datetime(2003, 10, 17, 19, 30, 30, tzinfo=datetime.timezone.utc)
        lat = 39.742476
        lon = -105.1786

        # Original functions
        az_orig = solar.get_azimuth(lat, lon, when)
        alt_orig = solar.get_altitude(lat, lon, when)
        zenith_orig = 90.0 - alt_orig

        # Vectorised function
        time_np = np.datetime64(when)
        az_vec, zenith_vec = get_solar_angles_vector(lat, lon, time_np)  # scalar case - names unchanged

        # Compare results - may have larger differences due to different algorithms
        self.assertAlmostEqual(az_orig, az_vec, places=1,
                             msg=f"Azimuth mismatch: original={az_orig}, vectorised={az_vec}")
        self.assertAlmostEqual(zenith_orig, zenith_vec, places=1,
                             msg=f"Zenith mismatch: original={zenith_orig}, vectorised={zenith_vec}")


class TestVectorisedArrays(unittest.TestCase):
    """Test vectorised implementation with arrays of values."""

    def test_array_of_times_single_location(self):
        """Test array of times at a single location."""
        # Generate times over a day
        base_time = datetime.datetime(2016, 12, 19, 0, 0, 0, tzinfo=datetime.timezone.utc)
        times = [base_time + datetime.timedelta(hours=h) for h in range(24)]
        times_np = np.array([np.datetime64(t) for t in times])

        lat = 51.4826  # Greenwich
        lon = 0.0

        # Calculate using vectorised function
        az_vec, zenith_vec = get_solar_angles_vector(lat, lon, times_np)  # array case

        # Calculate using original functions
        az_orig = np.array([solar.get_azimuth(lat, lon, t) for t in times])
        alt_orig = np.array([solar.get_altitude(lat, lon, t) for t in times])
        zenith_orig = 90.0 - alt_orig

        # Compare arrays - allow 0.1 degree tolerance (different algorithms)
        np.testing.assert_array_almost_equal(az_vec, az_orig, decimal=1,
                                           err_msg="Azimuth arrays don't match")
        np.testing.assert_array_almost_equal(zenith_vec, zenith_orig, decimal=1,
                                           err_msg="Zenith arrays don't match")

    def test_array_of_locations_single_time(self):
        """Test array of locations at a single time."""
        when = datetime.datetime(2016, 12, 19, 12, 0, 0, tzinfo=datetime.timezone.utc)
        time_np = np.datetime64(when)

        # Various locations around the world
        lats = np.array([51.4826, -43.0, 59.6365662, 0.0, -33.8688])
        lons = np.array([0.0, 172.0, 12.5350953, -78.1834, 151.2093])

        # Calculate using vectorised function
        az_vec, zenith_vec = get_solar_angles_vector(lats, lons, time_np)  # array case

        # Calculate using original functions
        az_orig = np.array([solar.get_azimuth(lat, lon, when)
                           for lat, lon in zip(lats, lons)])
        alt_orig = np.array([solar.get_altitude(lat, lon, when)
                            for lat, lon in zip(lats, lons)])
        zenith_orig = 90.0 - alt_orig

        # Compare arrays - allow 0.1 degree tolerance (different algorithms)
        np.testing.assert_array_almost_equal(az_vec, az_orig, decimal=1,
                                           err_msg="Azimuth arrays don't match")
        np.testing.assert_array_almost_equal(zenith_vec, zenith_orig, decimal=1,
                                           err_msg="Zenith arrays don't match")

    def test_full_grid_times_and_locations(self):
        """Test grid of times and locations (broadcasting)."""
        # Multiple times
        base_time = datetime.datetime(2016, 6, 21, 0, 0, 0, tzinfo=datetime.timezone.utc)
        times = [base_time + datetime.timedelta(hours=h) for h in range(0, 24, 6)]
        times_np = np.array([np.datetime64(t) for t in times])

        # Multiple locations
        lats = np.array([60.0, 0.0, -60.0])
        lons = np.array([0.0, 120.0, -120.0])

        # Test each combination
        results_vec_az = []
        results_vec_zenith = []
        results_orig_az = []
        results_orig_zenith = []

        for lat, lon in zip(lats, lons):
            for time, time_np in zip(times, times_np):
                # Vectorised
                az_v, z_v = get_solar_angles_vector(lat, lon, time_np)  # scalar case - names unchanged
                results_vec_az.append(az_v)
                results_vec_zenith.append(z_v)

                # Original
                az_o = solar.get_azimuth(lat, lon, time)
                alt_o = solar.get_altitude(lat, lon, time)
                results_orig_az.append(az_o)
                results_orig_zenith.append(90.0 - alt_o)

        # Compare - allow 0.1 degree tolerance (different algorithms)
        np.testing.assert_array_almost_equal(
            np.array(results_vec_az),
            np.array(results_orig_az),
            decimal=1,
            err_msg="Azimuth grid doesn't match"
        )
        np.testing.assert_array_almost_equal(
            np.array(results_vec_zenith),
            np.array(results_orig_zenith),
            decimal=1,
            err_msg="Zenith grid doesn't match"
        )

    def test_sunrise_sunset_consistency(self):
        """Test consistency around sunrise/sunset when zenith angle crosses 90 degrees."""
        when = datetime.datetime(2016, 6, 21, 5, 0, 0, tzinfo=datetime.timezone.utc)
        lat = 51.4826  # Greenwich
        lon = 0.0

        # Test times around sunrise
        times = [when + datetime.timedelta(minutes=m) for m in range(-60, 61, 10)]
        times_np = np.array([np.datetime64(t) for t in times])

        # Vectorised
        az_vec, zenith_vec = get_solar_angles_vector(lat, lon, times_np)  # array case

        # Original
        az_orig = np.array([solar.get_azimuth(lat, lon, t) for t in times])
        alt_orig = np.array([solar.get_altitude(lat, lon, t) for t in times])
        zenith_orig = 90.0 - alt_orig

        # Compare - allow 0.1 degree tolerance (different algorithms)
        np.testing.assert_array_almost_equal(az_vec, az_orig, decimal=1,
                                           err_msg="Sunrise azimuth mismatch")
        np.testing.assert_array_almost_equal(zenith_vec, zenith_orig, decimal=1,
                                           err_msg="Sunrise zenith mismatch")


class TestVectorisedEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""

    def test_north_pole_summer_solstice(self):
        """Test at North Pole during summer solstice."""
        when = datetime.datetime(2016, 6, 21, 12, 0, 0, tzinfo=datetime.timezone.utc)
        lat = 90.0
        lon = 0.0

        time_np = np.datetime64(when)

        # Vectorised
        az_vec, zenith_vec = get_solar_angles_vector(lat, lon, time_np)  # scalar case - names unchanged

        # Original
        az_orig = solar.get_azimuth(lat, lon, when)
        alt_orig = solar.get_altitude(lat, lon, when)
        zenith_orig = 90.0 - alt_orig

        # At north pole, zenith angle should be roughly equal to (90 - declination)
        # which is about 66.5 degrees at summer solstice
        self.assertLess(zenith_vec, 90.0, "Sun should be above horizon at north pole in summer")
        self.assertAlmostEqual(zenith_orig, zenith_vec, places=1,
                             msg=f"Zenith mismatch at pole: original={zenith_orig}, vectorised={zenith_vec}")

    def test_equator_equinox(self):
        """Test at equator during equinox."""
        when = datetime.datetime(2016, 3, 20, 12, 0, 0, tzinfo=datetime.timezone.utc)
        lat = 0.0
        lon = 0.0

        time_np = np.datetime64(when)

        # Vectorised
        az_vec, zenith_vec = get_solar_angles_vector(lat, lon, time_np)  # scalar case - names unchanged

        # Original
        az_orig = solar.get_azimuth(lat, lon, when)
        alt_orig = solar.get_altitude(lat, lon, when)
        zenith_orig = 90.0 - alt_orig

        # At equator during equinox at noon, sun should be nearly overhead
        self.assertLess(zenith_vec, 10.0, "Sun should be nearly overhead at equator during equinox")
        self.assertAlmostEqual(zenith_orig, zenith_vec, places=1)

    def test_negative_zenith_handling(self):
        """Test that zenith angles stay within valid range (0-180 degrees)."""
        when = datetime.datetime(2016, 12, 19, 12, 0, 0, tzinfo=datetime.timezone.utc)
        times_np = np.array([np.datetime64(when)])

        # Test various locations
        lats = np.linspace(-90, 90, 10)
        lons = np.linspace(-180, 180, 10)

        for lat in lats:
            for lon in lons:
                az_vec, zenith_vec = get_solar_angles_vector(lat, lon, times_np)  # array case

                self.assertGreaterEqual(zenith_vec[0], 0.0,
                                      f"Zenith angle should be >= 0 at lat={lat}, lon={lon}")
                self.assertLessEqual(zenith_vec[0], 180.0,
                                   f"Zenith angle should be <= 180 at lat={lat}, lon={lon}")
                self.assertGreaterEqual(az_vec[0], 0.0,
                                      f"Azimuth should be >= 0 at lat={lat}, lon={lon}")
                self.assertLess(az_vec[0], 360.0,
                              f"Azimuth should be < 360 at lat={lat}, lon={lon}")


class TestVectorisedPerformance(unittest.TestCase):
    """Test that vectorised implementation handles large arrays efficiently."""

    def test_large_time_array(self):
        """Test with a large array of times (1 year at hourly intervals)."""
        base_time = datetime.datetime(2016, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
        times = [base_time + datetime.timedelta(hours=h) for h in range(0, 365*24, 24)]
        times_np = np.array([np.datetime64(t) for t in times])

        lat = 51.4826
        lon = 0.0

        # This should complete without error
        az_vec, zenith_vec = get_solar_angles_vector(lat, lon, times_np)  # array case

        # Verify output shape
        self.assertEqual(az_vec.shape, times_np.shape)
        self.assertEqual(zenith_vec.shape, times_np.shape)

        # Spot check a few values
        mid_idx = len(times) // 2
        az_orig = solar.get_azimuth(lat, lon, times[mid_idx])
        alt_orig = solar.get_altitude(lat, lon, times[mid_idx])

        self.assertAlmostEqual(az_vec[mid_idx], az_orig, places=1)
        self.assertAlmostEqual(zenith_vec[mid_idx], 90.0 - alt_orig, places=1)


class TestRadiationVectorised(unittest.TestCase):
    """Test vectorised radiation calculations."""

    def test_single_radiation_daytime(self):
        """Test radiation calculation for a single daytime value."""
        when = datetime.datetime(2016, 6, 21, 16, 0, 0, tzinfo=datetime.timezone.utc)
        lat = 42.364908
        lon = -71.112828

        # Original method
        alt_orig = solar.get_altitude(lat, lon, when)
        rad_orig = radiation.get_radiation_direct(when, alt_orig)

        # Vectorised method
        az_vec, zen_vec = get_solar_angles_vector(lat, lon, np.datetime64(when))
        alt_vec = 90.0 - zen_vec
        rad_vec = get_radiation_direct_vector(alt_vec, np.datetime64(when))

        # Should match within 1 W/m²
        self.assertAlmostEqual(rad_orig, rad_vec, delta=1.0,
                             msg=f"Radiation mismatch: original={rad_orig}, vectorised={rad_vec}")

    def test_single_radiation_nighttime(self):
        """Test radiation is zero at night."""
        when = datetime.datetime(2016, 6, 21, 3, 0, 0, tzinfo=datetime.timezone.utc)
        lat = 42.364908
        lon = -71.112828

        # Original method
        alt_orig = solar.get_altitude(lat, lon, when)
        rad_orig = radiation.get_radiation_direct(when, alt_orig)

        # Vectorised method
        az_vec, zen_vec = get_solar_angles_vector(lat, lon, np.datetime64(when))
        alt_vec = 90.0 - zen_vec
        rad_vec = get_radiation_direct_vector(alt_vec, np.datetime64(when))

        # Both should be zero at night
        self.assertEqual(rad_orig, 0.0)
        self.assertEqual(rad_vec, 0.0)

    def test_radiation_array_full_day(self):
        """Test radiation over a full day."""
        base_time = datetime.datetime(2016, 6, 21, 0, 0, 0, tzinfo=datetime.timezone.utc)
        times = [base_time + datetime.timedelta(hours=h) for h in range(24)]
        times_np = np.array([np.datetime64(t) for t in times])

        lat = 42.364908
        lon = -71.112828

        # Original method
        rad_orig = np.array([
            radiation.get_radiation_direct(t, solar.get_altitude(lat, lon, t))
            for t in times
        ])

        # Vectorised method
        az_vec, zen_vec = get_solar_angles_vector(lat, lon, times_np)
        alt_vec = 90.0 - zen_vec
        rad_vec = get_radiation_direct_vector(alt_vec, times_np)

        # Compare arrays - allow 1 W/m² tolerance
        np.testing.assert_allclose(rad_vec, rad_orig, atol=1.0,
                                  err_msg="Radiation arrays don't match")

    def test_radiation_peak_at_solar_noon(self):
        """Test that peak radiation occurs near solar noon."""
        # Summer solstice, full day
        base_time = datetime.datetime(2016, 6, 21, 0, 0, 0, tzinfo=datetime.timezone.utc)
        times = [base_time + datetime.timedelta(minutes=m) for m in range(0, 24*60, 30)]
        times_np = np.array([np.datetime64(t) for t in times])

        lat = 42.364908
        lon = -71.112828

        # Vectorised method
        az_vec, zen_vec = get_solar_angles_vector(lat, lon, times_np)
        alt_vec = 90.0 - zen_vec
        rad_vec = get_radiation_direct_vector(alt_vec, times_np)

        # Peak should be during daylight hours (roughly 10-20 UTC for Boston in June)
        peak_idx = np.argmax(rad_vec)
        peak_hour = times[peak_idx].hour

        self.assertGreater(peak_hour, 10, "Peak should be after 10 AM")
        self.assertLess(peak_hour, 20, "Peak should be before 8 PM")
        self.assertGreater(rad_vec[peak_idx], 800, "Peak radiation should be > 800 W/m²")

    def test_radiation_multiple_locations(self):
        """Test radiation at multiple locations simultaneously."""
        when = datetime.datetime(2016, 6, 21, 16, 0, 0, tzinfo=datetime.timezone.utc)
        time_np = np.datetime64(when)

        # Different latitudes
        lats = np.array([0.0, 30.0, 45.0, 60.0])  # Equator to high latitude
        lons = np.array([0.0, 0.0, 0.0, 0.0])

        # Vectorised calculation
        az_vec, zen_vec = get_solar_angles_vector(lats, lons, time_np)
        alt_vec = 90.0 - zen_vec
        rad_vec = get_radiation_direct_vector(alt_vec, time_np)

        # All should be valid (>= 0)
        self.assertTrue(np.all(rad_vec >= 0), "All radiation values should be non-negative")

        # Verify array shape
        self.assertEqual(rad_vec.shape, lats.shape)

    def test_radiation_zero_altitude(self):
        """Test radiation at horizon (altitude = 0)."""
        # Manually set altitude to exactly 0
        alt = np.array([0.0])
        when = np.array([np.datetime64('2016-06-21T12:00:00')])

        rad = get_radiation_direct_vector(alt, when)

        # Should be very close to zero or zero
        self.assertLess(rad[0], 1.0, "Radiation at horizon should be near zero")

    def test_radiation_high_altitude(self):
        """Test radiation at high solar altitude (near overhead)."""
        # Manually set high altitude
        alt = np.array([85.0])  # Nearly overhead
        when = np.array([np.datetime64('2016-06-21T12:00:00')])

        rad = get_radiation_direct_vector(alt, when)

        # Should be high radiation
        self.assertGreater(rad[0], 800, "Radiation at high altitude should be > 800 W/m²")


if __name__ == "__main__":
    unittest.main(verbosity=2)
