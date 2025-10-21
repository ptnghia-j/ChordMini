#!/usr/bin/env python3
"""
Comprehensive test script for BTC API integration.
Tests both BTC model variants through Flask API endpoints and validates
that API responses match direct btc_chord_recognition.py outputs exactly.
"""

import os
import sys
import time
import json
import requests
import tempfile
import hashlib
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from btc_chord_recognition import btc_chord_recognition

class BTCAPIIntegrationTester:
    def __init__(self, base_url="http://localhost:5001"):
        self.base_url = base_url
        self.test_results = []
        self.temp_dir = tempfile.mkdtemp(prefix="btc_api_test_")
        
    def test_server_health(self):
        """Test if the server is running and responsive."""
        print("🔍 Testing server health...")
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                print("✅ Server is healthy and responsive")
                return True
            else:
                print(f"❌ Server health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Server health check failed: {e}")
            return False
    
    def test_btc_model_availability(self):
        """Test BTC model availability through API."""
        print("\n🔍 Testing BTC model availability...")
        
        try:
            # Test model info endpoint
            response = requests.get(f"{self.base_url}/api/model-info", timeout=10)
            if response.status_code == 200:
                data = response.json()
                chord_models = data.get('available_chord_models', [])
                
                btc_sl_available = 'btc-sl' in chord_models
                btc_pl_available = 'btc-pl' in chord_models
                
                print(f"BTC-SL available: {'✅' if btc_sl_available else '❌'}")
                print(f"BTC-PL available: {'✅' if btc_pl_available else '❌'}")
                
                return btc_sl_available, btc_pl_available
            else:
                print(f"❌ Model info request failed: {response.status_code}")
                return False, False
                
        except Exception as e:
            print(f"❌ Model availability check failed: {e}")
            return False, False
    
    def test_btc_endpoint(self, model_variant, audio_file):
        """Test a specific BTC endpoint with an audio file."""
        print(f"\n🔍 Testing BTC-{model_variant.upper()} API endpoint...")
        
        endpoint = f"/api/recognize-chords-btc-{model_variant}"
        url = f"{self.base_url}{endpoint}"
        
        try:
            # Prepare the file for upload
            with open(audio_file, 'rb') as f:
                files = {'file': (os.path.basename(audio_file), f, 'audio/mpeg')}
                
                start_time = time.time()
                response = requests.post(url, files=files, timeout=300)
                api_time = time.time() - start_time
            
            if response.status_code == 200:
                api_data = response.json()
                
                if api_data.get('success', False):
                    print(f"✅ BTC-{model_variant.upper()} API call successful")
                    print(f"   Processing time: {api_time:.2f}s")
                    print(f"   Model used: {api_data.get('model_used', 'unknown')}")
                    print(f"   Total chords: {api_data.get('total_chords', 0)}")
                    
                    return {
                        'success': True,
                        'api_time': api_time,
                        'api_data': api_data,
                        'chords': api_data.get('chords', [])
                    }
                else:
                    print(f"❌ BTC-{model_variant.upper()} API returned success=False")
                    print(f"   Error: {api_data.get('error', 'Unknown error')}")
                    return {'success': False, 'error': api_data.get('error', 'Unknown error')}
            else:
                print(f"❌ BTC-{model_variant.upper()} API request failed: {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"   Error: {error_data.get('error', 'Unknown error')}")
                except:
                    print(f"   Response: {response.text[:200]}...")
                return {'success': False, 'error': f"HTTP {response.status_code}"}
                
        except Exception as e:
            print(f"❌ BTC-{model_variant.upper()} API test failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def test_direct_btc_call(self, model_variant, audio_file):
        """Test direct btc_chord_recognition.py call for comparison."""
        print(f"\n🔍 Testing direct BTC-{model_variant.upper()} call...")
        
        output_file = os.path.join(self.temp_dir, f"direct_{model_variant}.lab")
        
        try:
            start_time = time.time()
            success = btc_chord_recognition(audio_file, output_file, model_variant)
            direct_time = time.time() - start_time
            
            if success and os.path.exists(output_file):
                # Parse the lab file to match API format
                chords = []
                with open(output_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 3:
                            start_time_val = float(parts[0])
                            end_time_val = float(parts[1])
                            chord_name = parts[2]
                            chords.append({
                                'start': start_time_val,
                                'end': end_time_val,
                                'chord': chord_name,
                                'confidence': 1.0
                            })
                
                print(f"✅ Direct BTC-{model_variant.upper()} call successful")
                print(f"   Processing time: {direct_time:.2f}s")
                print(f"   Total chords: {len(chords)}")
                
                return {
                    'success': True,
                    'direct_time': direct_time,
                    'chords': chords,
                    'output_file': output_file
                }
            else:
                print(f"❌ Direct BTC-{model_variant.upper()} call failed")
                return {'success': False, 'error': 'Direct call failed'}
                
        except Exception as e:
            print(f"❌ Direct BTC-{model_variant.upper()} test failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def compare_api_vs_direct(self, api_result, direct_result, model_variant):
        """Compare API results with direct call results."""
        print(f"\n🔍 Comparing API vs Direct results for BTC-{model_variant.upper()}...")
        
        if not api_result['success'] or not direct_result['success']:
            print("❌ Cannot compare - one or both calls failed")
            return False
        
        api_chords = api_result['chords']
        direct_chords = direct_result['chords']
        
        # Compare chord counts
        if len(api_chords) != len(direct_chords):
            print(f"❌ Chord count mismatch: API={len(api_chords)}, Direct={len(direct_chords)}")
            return False
        
        # Compare individual chords
        mismatches = 0
        for i, (api_chord, direct_chord) in enumerate(zip(api_chords, direct_chords)):
            api_start = round(api_chord['start'], 6)
            api_end = round(api_chord['end'], 6)
            api_name = api_chord['chord']
            
            direct_start = round(direct_chord['start'], 6)
            direct_end = round(direct_chord['end'], 6)
            direct_name = direct_chord['chord']
            
            if (api_start != direct_start or 
                api_end != direct_end or 
                api_name != direct_name):
                mismatches += 1
                if mismatches <= 3:  # Show first 3 mismatches
                    print(f"   Mismatch {mismatches} at index {i}:")
                    print(f"     API:    {api_start:.6f}-{api_end:.6f} {api_name}")
                    print(f"     Direct: {direct_start:.6f}-{direct_end:.6f} {direct_name}")
        
        if mismatches == 0:
            print("✅ Perfect match between API and direct results!")
            
            # Compare processing times
            api_time = api_result['api_time']
            direct_time = direct_result['direct_time']
            time_ratio = api_time / direct_time if direct_time > 0 else float('inf')
            
            print(f"   Processing times: API={api_time:.2f}s, Direct={direct_time:.2f}s")
            print(f"   Time ratio: {time_ratio:.2f}x")
            
            return True
        else:
            print(f"❌ {mismatches} mismatches found between API and direct results")
            return False
    
    def test_error_handling(self):
        """Test API error handling with invalid inputs."""
        print("\n🔍 Testing error handling...")
        
        # Test with non-existent file
        try:
            response = requests.post(
                f"{self.base_url}/api/recognize-chords-btc-sl",
                data={'audio_path': '/nonexistent/file.mp3'},
                timeout=30
            )
            
            if response.status_code == 404:
                print("✅ Correctly handles non-existent file (404)")
            else:
                print(f"❌ Unexpected response for non-existent file: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error handling test failed: {e}")
    
    def run_comprehensive_test(self, audio_file):
        """Run comprehensive API integration tests."""
        print("🎵 BTC API Integration Test Suite")
        print("=" * 60)
        print(f"Test audio file: {audio_file}")
        print(f"Server URL: {self.base_url}")
        print(f"Temporary directory: {self.temp_dir}")
        
        # Test server health
        if not self.test_server_health():
            print("❌ Server health check failed - aborting tests")
            return False
        
        # Test model availability
        btc_sl_available, btc_pl_available = self.test_btc_model_availability()
        
        if not btc_sl_available and not btc_pl_available:
            print("❌ No BTC models available - aborting tests")
            return False
        
        # Test both model variants
        variants_to_test = []
        if btc_sl_available:
            variants_to_test.append('sl')
        if btc_pl_available:
            variants_to_test.append('pl')
        
        all_tests_passed = True
        
        for variant in variants_to_test:
            print(f"\n{'='*20} Testing BTC-{variant.upper()} {'='*20}")
            
            # Test API endpoint
            api_result = self.test_btc_endpoint(variant, audio_file)
            
            # Test direct call
            direct_result = self.test_direct_btc_call(variant, audio_file)
            
            # Compare results
            comparison_passed = self.compare_api_vs_direct(api_result, direct_result, variant)
            
            if not comparison_passed:
                all_tests_passed = False
            
            # Store results
            self.test_results.append({
                'variant': variant,
                'api_result': api_result,
                'direct_result': direct_result,
                'comparison_passed': comparison_passed
            })
        
        # Test error handling
        self.test_error_handling()
        
        # Generate summary
        self.generate_summary()
        
        return all_tests_passed
    
    def generate_summary(self):
        """Generate a comprehensive test summary."""
        print(f"\n📊 TEST SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r['comparison_passed'])
        
        print(f"Total BTC model tests: {total_tests}")
        print(f"Passed tests: {passed_tests}")
        print(f"Failed tests: {total_tests - passed_tests}")
        print(f"Success rate: {(passed_tests/total_tests)*100:.1f}%" if total_tests > 0 else "N/A")
        
        if passed_tests == total_tests:
            print("\n🎯 ALL TESTS PASSED!")
            print("✅ Complete functional equivalence between API and direct calls")
            print("✅ No fallback to Chord-CNN-LSTM detected")
            print("✅ BTC models are fully operational through API")
        else:
            print(f"\n❌ {total_tests - passed_tests} tests failed")
            for result in self.test_results:
                if not result['comparison_passed']:
                    variant = result['variant']
                    print(f"   FAILED: BTC-{variant.upper()}")
        
        # Performance summary
        if self.test_results:
            print(f"\n⏱️  Performance Summary:")
            for result in self.test_results:
                variant = result['variant']
                if result['api_result']['success'] and result['direct_result']['success']:
                    api_time = result['api_result']['api_time']
                    direct_time = result['direct_result']['direct_time']
                    print(f"   BTC-{variant.upper()}: API={api_time:.2f}s, Direct={direct_time:.2f}s")
    
    def cleanup(self):
        """Clean up temporary files."""
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
            print(f"\n🧹 Cleaned up temporary directory: {self.temp_dir}")
        except Exception as e:
            print(f"Warning: Could not clean up {self.temp_dir}: {e}")

def main():
    """Main test function."""
    # Use the same test audio file
    audio_file = "./test/_zU7r8ATFmg.mp3"
    
    if not os.path.exists(audio_file):
        print(f"❌ Test audio file not found: {audio_file}")
        return 1
    
    tester = BTCAPIIntegrationTester()
    
    try:
        success = tester.run_comprehensive_test(audio_file)
        return 0 if success else 1
    finally:
        tester.cleanup()

if __name__ == "__main__":
    sys.exit(main())
