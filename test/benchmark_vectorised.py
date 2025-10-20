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
Performance benchmark comparing three solar calculation approaches:
1. Full NREL SPA (get_azimuth/get_altitude) - most accurate
2. Vectorised implementation - balance of speed and accuracy
3. Fast scalar (get_azimuth_fast/get_altitude_fast) - fastest scalar

Key comparisons:
- Full vs Vectorised: How much speed gain for small accuracy loss?
- Full vs Fast: How does existing fast version compare?
"""

import datetime
import numpy as np
import time
import warnings
from pysolar import solar, radiation
from pysolar.vectorised import get_solar_angles_vector, get_radiation_direct_vector

# Suppress numpy datetime64 timezone warnings (known limitation)
warnings.filterwarnings('ignore', message='.*no explicit representation of timezones.*')


def benchmark_full(times, lats, lons):
    """Benchmark the full NREL SPA implementation."""
    n = len(times)
    azimuths = np.zeros(n)
    zeniths = np.zeros(n)

    start = time.perf_counter()
    for i in range(n):
        azimuths[i] = solar.get_azimuth(lats[i], lons[i], times[i])
        altitude = solar.get_altitude(lats[i], lons[i], times[i])
        zeniths[i] = 90.0 - altitude
    end = time.perf_counter()

    elapsed = end - start
    return azimuths, zeniths, elapsed


def benchmark_fast(times, lats, lons):
    """Benchmark the original fast scalar implementation."""
    n = len(times)
    azimuths = np.zeros(n)
    zeniths = np.zeros(n)

    start = time.perf_counter()
    for i in range(n):
        azimuths[i] = solar.get_azimuth_fast(lats[i], lons[i], times[i])
        altitude = solar.get_altitude_fast(lats[i], lons[i], times[i])
        zeniths[i] = 90.0 - altitude
    end = time.perf_counter()

    elapsed = end - start
    return azimuths, zeniths, elapsed


def benchmark_vectorised(times_np, lats, lons):
    """Benchmark the vectorised implementation."""
    start = time.perf_counter()
    azimuths, zeniths = get_solar_angles_vector(lats, lons, times_np)
    end = time.perf_counter()

    elapsed = end - start
    return azimuths, zeniths, elapsed


def run_benchmark(n_calculations):
    """Run benchmark with n_calculations data points."""
    print(f"\n{'='*80}")
    print(f"Benchmark: {n_calculations:,} solar position calculations")
    print(f"{'='*80}")

    # Generate test data
    base_time = datetime.datetime(2016, 6, 21, 0, 0, 0, tzinfo=datetime.timezone.utc)
    times = [base_time + datetime.timedelta(hours=i*24/n_calculations)
             for i in range(n_calculations)]

    # Vary locations too
    lats = np.linspace(-60, 60, n_calculations)
    lons = np.linspace(-180, 180, n_calculations)

    # Convert to numpy datetime64 for vectorised version
    times_np = np.array([np.datetime64(t) for t in times])

    # Warm up
    if n_calculations >= 10:
        _, _, _ = benchmark_full(times[:10], lats[:10], lons[:10])
        _, _, _ = benchmark_fast(times[:10], lats[:10], lons[:10])
        _, _, _ = benchmark_vectorised(times_np[:10], lats[:10], lons[:10])

    # Benchmark full NREL SPA
    print("\n1. Full NREL SPA:")
    print("   Running...", end=" ", flush=True)
    az_full, zen_full, time_full = benchmark_full(times, lats, lons)
    print(f"✓")
    print(f"   Time: {time_full:.4f}s | Rate: {n_calculations/time_full:,.0f} calcs/sec")

    # Benchmark vectorised implementation
    print("\n2. Vectorised implementation:")
    print("   Running...", end=" ", flush=True)
    az_vec, zen_vec, time_vec = benchmark_vectorised(times_np, lats, lons)
    print(f"✓")
    print(f"   Time: {time_vec:.4f}s | Rate: {n_calculations/time_vec:,.0f} calcs/sec")

    # Benchmark fast scalar
    print("\n3. Fast scalar (get_*_fast):")
    print("   Running...", end=" ", flush=True)
    az_fast, zen_fast, time_fast = benchmark_fast(times, lats, lons)
    print(f"✓")
    print(f"   Time: {time_fast:.4f}s | Rate: {n_calculations/time_fast:,.0f} calcs/sec")

    # Comparison 1: Full vs Vectorised
    print(f"\n{'─'*80}")
    print("COMPARISON 1: Full NREL SPA vs Vectorised")
    print(f"{'─'*80}")

    speedup_vec = time_full / time_vec
    print(f"  Speedup: {speedup_vec:.1f}x faster")
    print(f"  Time saved: {time_full - time_vec:.4f}s ({(1-time_vec/time_full)*100:.1f}% reduction)")

    # Accuracy vs full
    az_diff_vec = np.abs((az_vec - az_full + 180) % 360 - 180)
    zen_diff_vec = np.abs(zen_vec - zen_full)

    print(f"  Accuracy loss:")
    print(f"    Azimuth  - Mean: {np.mean(az_diff_vec):.4f}° | Max: {np.max(az_diff_vec):.4f}°")
    print(f"    Zenith   - Mean: {np.mean(zen_diff_vec):.4f}° | Max: {np.max(zen_diff_vec):.4f}°")

    # Comparison 2: Full vs Fast
    print(f"\n{'─'*80}")
    print("COMPARISON 2: Full NREL SPA vs Fast Scalar")
    print(f"{'─'*80}")

    speedup_fast = time_full / time_fast
    print(f"  Speedup: {speedup_fast:.1f}x faster")
    print(f"  Time saved: {time_full - time_fast:.4f}s ({(1-time_fast/time_full)*100:.1f}% reduction)")

    # Accuracy vs full
    az_diff_fast = np.abs((az_fast - az_full + 180) % 360 - 180)
    zen_diff_fast = np.abs(zen_fast - zen_full)

    print(f"  Accuracy loss:")
    print(f"    Azimuth  - Mean: {np.mean(az_diff_fast):.4f}° | Max: {np.max(az_diff_fast):.4f}°")
    print(f"    Zenith   - Mean: {np.mean(zen_diff_fast):.4f}° | Max: {np.max(zen_diff_fast):.4f}°")

    return {
        'n': n_calculations,
        'time_full': time_full,
        'time_vec': time_vec,
        'time_fast': time_fast,
        'speedup_vec': speedup_vec,
        'speedup_fast': speedup_fast,
        'vec_az_max': np.max(az_diff_vec),
        'vec_zen_max': np.max(zen_diff_vec),
        'fast_az_max': np.max(az_diff_fast),
        'fast_zen_max': np.max(zen_diff_fast),
    }


def main():
    """Run benchmarks with different sizes."""
    print("\n" + "="*80)
    print("Solar Position Algorithm Performance Benchmark")
    print("Comparing three implementations against Full NREL SPA")
    print("="*80)

    # Different test sizes
    test_sizes = [1, 100, 1_000, 10_000]
    results = []

    for size in test_sizes:
        result = run_benchmark(size)
        results.append(result)

    # Summary table
    print(f"\n\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"\n{'N':<10} {'Full (s)':<12} {'Vector (s)':<12} {'Fast (s)':<12} {'Vec/Full':<12} {'Fast/Full':<12}")
    print(f"{'-'*80}")

    for r in results:
        print(f"{r['n']:<10,} {r['time_full']:<12.4f} {r['time_vec']:<12.4f} "
              f"{r['time_fast']:<12.4f} {r['speedup_vec']:>10.1f}x {r['speedup_fast']:>10.1f}x")

    print(f"\n{'Accuracy vs Full (max error in degrees):':<10}")
    print(f"{'-'*80}")
    print(f"{'N':<10} {'Vectorised Az':<18} {'Vectorised Zen':<18} {'Fast Az':<18} {'Fast Zen':<18}")
    print(f"{'-'*80}")

    for r in results:
        print(f"{r['n']:<10,} {r['vec_az_max']:<18.4f} {r['vec_zen_max']:<18.4f} "
              f"{r['fast_az_max']:<18.4f} {r['fast_zen_max']:<18.4f}")

    print(f"\n{'='*80}")

    # Overall conclusions
    avg_speedup_vec = np.mean([r['speedup_vec'] for r in results])
    avg_speedup_fast = np.mean([r['speedup_fast'] for r in results])
    max_error_vec = max([max(r['vec_az_max'], r['vec_zen_max']) for r in results])
    max_error_fast = max([max(r['fast_az_max'], r['fast_zen_max']) for r in results])

    print(f"\nKEY FINDINGS:")
    print(f"  Vectorised vs Full:")
    print(f"    • Average speedup: {avg_speedup_vec:.1f}x")
    print(f"    • Max error: {max_error_vec:.4f}° ({max_error_vec*60:.2f} arcmin)")
    print(f"\n  Fast Scalar vs Full:")
    print(f"    • Average speedup: {avg_speedup_fast:.1f}x")
    print(f"    • Max error: {max_error_fast:.4f}° ({max_error_fast*60:.2f} arcmin)")
    print(f"\n  Conclusion: Vectorised is {avg_speedup_vec/avg_speedup_fast:.1f}x faster than Fast")
    print(f"              with {max_error_fast/max_error_vec:.1f}x better accuracy!")
    print(f"\n")


def benchmark_radiation(n_calculations):
    """Benchmark radiation calculations."""
    print(f"\n{'='*80}")
    print(f"RADIATION BENCHMARK: {n_calculations:,} calculations")
    print(f"{'='*80}")

    # Generate test data
    base_time = datetime.datetime(2016, 6, 21, 0, 0, 0, tzinfo=datetime.timezone.utc)
    times = [base_time + datetime.timedelta(hours=i*24/n_calculations)
             for i in range(n_calculations)]
    times_np = np.array([np.datetime64(t) for t in times])

    lat = 42.364908
    lon = -71.112828

    # Benchmark original method (get_altitude + get_radiation_direct)
    print("\n1. Original (scalar loop):")
    print("   Running...", end=" ", flush=True)
    start = time.perf_counter()
    rad_orig = np.array([
        radiation.get_radiation_direct(t, solar.get_altitude(lat, lon, t))
        for t in times
    ])
    time_orig = time.perf_counter() - start
    print(f"✓")
    print(f"   Time: {time_orig:.4f}s | Rate: {n_calculations/time_orig:,.0f} calcs/sec")

    # Benchmark vectorised method
    print("\n2. Vectorised (combined angles + radiation):")
    print("   Running...", end=" ", flush=True)
    start = time.perf_counter()
    _, zen_vec = get_solar_angles_vector(lat, lon, times_np)
    alt_vec = 90.0 - zen_vec
    rad_vec = get_radiation_direct_vector(alt_vec, times_np)
    time_vec = time.perf_counter() - start
    print(f"✓")
    print(f"   Time: {time_vec:.4f}s | Rate: {n_calculations/time_vec:,.0f} calcs/sec")

    # Results
    speedup = time_orig / time_vec
    rad_diff = np.abs(rad_vec - rad_orig)

    # Filter out NaN and inf values if any
    valid_diff = rad_diff[np.isfinite(rad_diff)]
    valid_rad = rad_vec[np.isfinite(rad_vec)]

    print(f"\n{'─'*80}")
    print(f"Speedup: {speedup:.1f}x faster")
    print(f"Time saved: {time_orig - time_vec:.4f}s ({(1-time_vec/time_orig)*100:.1f}% reduction)")
    print(f"\nAccuracy:")
    if len(valid_diff) > 0:
        print(f"  Mean difference: {np.mean(valid_diff):.2f} W/m²")
        print(f"  Max difference:  {np.max(valid_diff):.2f} W/m²")
    else:
        print(f"  No valid differences to report")
    if len(valid_rad) > 0:
        print(f"  Mean radiation:  {np.mean(valid_rad):.2f} W/m²")
    print(f"{'─'*80}")


def main_with_radiation():
    """Run all benchmarks including radiation."""
    # Run solar position benchmarks
    main()

    # Run radiation benchmarks
    print("\n" + "="*80)
    print("RADIATION BENCHMARKS")
    print("="*80)

    for size in [100, 1_000, 10_000]:
        benchmark_radiation(size)

    print("\n")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--with-radiation":
        main_with_radiation()
    else:
        main()
