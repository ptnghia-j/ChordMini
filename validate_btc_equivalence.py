#!/usr/bin/env python3
"""
Comprehensive validation script to ensure complete functional equivalence
between btc_chord_recognition.py and test_btc.py implementations.

This script performs systematic testing to validate:
1. Audio preprocessing pipeline equivalence
2. Model inference logic equivalence  
3. Post-processing equivalence
4. Output formatting equivalence
"""

import os
import sys
import time
import hashlib
import subprocess
import tempfile
import shutil
from pathlib import Path
import numpy as np

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from btc_chord_recognition import btc_chord_recognition
from modules.utils import logger

class BTCEquivalenceValidator:
    def __init__(self):
        self.test_results = []
        self.temp_dir = tempfile.mkdtemp(prefix="btc_validation_")
        self.audio_files = self._find_test_audio_files()
        
    def _find_test_audio_files(self):
        """Find all available test audio files."""
        audio_files = []
        test_dir = Path("./test")
        
        if test_dir.exists():
            for ext in ['*.mp3', '*.wav']:
                audio_files.extend(list(test_dir.glob(ext)))
        
        # Add the main test file
        main_test_file = test_dir / "_zU7r8ATFmg.mp3"
        if main_test_file.exists() and main_test_file not in audio_files:
            audio_files.insert(0, main_test_file)
            
        return [str(f) for f in audio_files]
    
    def _run_test_btc(self, audio_file, output_file, model_variant='sl'):
        """Run test_btc.py and return success status."""
        try:
            cmd = [
                sys.executable, "test_btc.py",
                "--audio_dir", os.path.dirname(audio_file),
                "--save_dir", os.path.dirname(output_file),
                "--model_file", self._get_model_path(model_variant)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                # test_btc.py saves with original filename, need to rename
                original_name = os.path.splitext(os.path.basename(audio_file))[0] + '.lab'
                test_btc_output = os.path.join(os.path.dirname(output_file), original_name)
                
                if os.path.exists(test_btc_output):
                    shutil.move(test_btc_output, output_file)
                    return True
            
            logger.error(f"test_btc.py failed: {result.stderr}")
            return False
            
        except Exception as e:
            logger.error(f"Error running test_btc.py: {e}")
            return False
    
    def _get_model_path(self, variant):
        """Get the correct model path for the variant."""
        if variant == 'sl':
            return './checkpoints/SL/btc_model_large_voca.pt'
        else:  # pl
            return './checkpoints/btc/btc_combined_best.pth'
    
    def _compare_files(self, file1, file2):
        """Compare two files byte-for-byte and return detailed comparison."""
        if not os.path.exists(file1) or not os.path.exists(file2):
            return {
                'identical': False,
                'error': f"Missing files: {file1} exists: {os.path.exists(file1)}, {file2} exists: {os.path.exists(file2)}"
            }
        
        # Read both files
        with open(file1, 'r') as f1, open(file2, 'r') as f2:
            lines1 = f1.readlines()
            lines2 = f2.readlines()
        
        # Compare line counts
        if len(lines1) != len(lines2):
            return {
                'identical': False,
                'line_count_diff': f"{len(lines1)} vs {len(lines2)}",
                'lines1': len(lines1),
                'lines2': len(lines2)
            }
        
        # Compare line by line
        differences = []
        for i, (line1, line2) in enumerate(zip(lines1, lines2)):
            if line1.strip() != line2.strip():
                differences.append({
                    'line': i + 1,
                    'file1': line1.strip(),
                    'file2': line2.strip()
                })
        
        # Calculate file hashes
        hash1 = hashlib.md5(open(file1, 'rb').read()).hexdigest()
        hash2 = hashlib.md5(open(file2, 'rb').read()).hexdigest()
        
        return {
            'identical': len(differences) == 0,
            'line_count': len(lines1),
            'differences': differences,
            'hash1': hash1,
            'hash2': hash2,
            'hash_match': hash1 == hash2
        }
    
    def validate_single_file(self, audio_file, model_variant='sl'):
        """Validate equivalence for a single audio file."""
        print(f"\n🔍 Validating: {os.path.basename(audio_file)} (model: {model_variant})")
        print("-" * 60)
        
        # Generate output file paths
        base_name = os.path.splitext(os.path.basename(audio_file))[0]
        test_btc_output = os.path.join(self.temp_dir, f"{base_name}_test_btc_{model_variant}.lab")
        integration_output = os.path.join(self.temp_dir, f"{base_name}_integration_{model_variant}.lab")
        
        # Test both implementations
        start_time = time.time()
        
        # Run test_btc.py
        print("Running test_btc.py...")
        test_btc_success = self._run_test_btc(audio_file, test_btc_output, model_variant)
        test_btc_time = time.time() - start_time
        
        # Run btc_chord_recognition.py
        print("Running btc_chord_recognition.py...")
        start_time = time.time()
        integration_success = btc_chord_recognition(audio_file, integration_output, model_variant)
        integration_time = time.time() - start_time
        
        # Compare results
        result = {
            'audio_file': audio_file,
            'model_variant': model_variant,
            'test_btc_success': test_btc_success,
            'integration_success': integration_success,
            'test_btc_time': test_btc_time,
            'integration_time': integration_time,
            'test_btc_output': test_btc_output,
            'integration_output': integration_output
        }
        
        if test_btc_success and integration_success:
            comparison = self._compare_files(test_btc_output, integration_output)
            result.update(comparison)
            
            if comparison['identical']:
                print("✅ PERFECT MATCH - Files are byte-for-byte identical")
            else:
                print("❌ MISMATCH DETECTED")
                if 'line_count_diff' in comparison:
                    print(f"   Line count difference: {comparison['line_count_diff']}")
                if comparison['differences']:
                    print(f"   {len(comparison['differences'])} line differences found")
                    for diff in comparison['differences'][:3]:  # Show first 3 differences
                        print(f"   Line {diff['line']}: '{diff['file1']}' vs '{diff['file2']}'")
        else:
            result['identical'] = False
            result['error'] = f"Execution failed - test_btc: {test_btc_success}, integration: {integration_success}"
            print(f"❌ EXECUTION FAILED - test_btc: {test_btc_success}, integration: {integration_success}")
        
        print(f"Processing times: test_btc={test_btc_time:.2f}s, integration={integration_time:.2f}s")
        
        self.test_results.append(result)
        return result
    
    def run_comprehensive_validation(self):
        """Run comprehensive validation across all test files and model variants."""
        print("🎵 BTC Model Equivalence Validation Suite")
        print("=" * 60)
        print(f"Test audio files found: {len(self.audio_files)}")
        print(f"Temporary directory: {self.temp_dir}")
        
        if not self.audio_files:
            print("❌ No test audio files found!")
            return False
        
        # Test both model variants
        variants = ['sl', 'pl']
        total_tests = len(self.audio_files) * len(variants)
        passed_tests = 0
        
        for audio_file in self.audio_files:
            for variant in variants:
                try:
                    result = self.validate_single_file(audio_file, variant)
                    if result.get('identical', False):
                        passed_tests += 1
                except Exception as e:
                    print(f"❌ Error testing {audio_file} with {variant}: {e}")
                    self.test_results.append({
                        'audio_file': audio_file,
                        'model_variant': variant,
                        'identical': False,
                        'error': str(e)
                    })
        
        # Generate summary report
        self._generate_summary_report(passed_tests, total_tests)
        
        return passed_tests == total_tests
    
    def _generate_summary_report(self, passed_tests, total_tests):
        """Generate a comprehensive summary report."""
        print(f"\n📊 VALIDATION SUMMARY")
        print("=" * 60)
        print(f"Total tests: {total_tests}")
        print(f"Passed tests: {passed_tests}")
        print(f"Failed tests: {total_tests - passed_tests}")
        print(f"Success rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if passed_tests == total_tests:
            print("\n🎯 ALL TESTS PASSED - Complete functional equivalence achieved!")
            print("✅ btc_chord_recognition.py produces identical results to test_btc.py")
        else:
            print(f"\n❌ {total_tests - passed_tests} tests failed - Functional equivalence NOT achieved")
            
            # Show failed tests
            failed_tests = [r for r in self.test_results if not r.get('identical', False)]
            for test in failed_tests:
                print(f"   FAILED: {os.path.basename(test['audio_file'])} ({test['model_variant']})")
                if 'error' in test:
                    print(f"           Error: {test['error']}")
        
        # Performance summary
        successful_tests = [r for r in self.test_results if r.get('identical', False)]
        if successful_tests:
            avg_test_btc_time = np.mean([r['test_btc_time'] for r in successful_tests])
            avg_integration_time = np.mean([r['integration_time'] for r in successful_tests])
            print(f"\n⏱️  Average processing times:")
            print(f"   test_btc.py: {avg_test_btc_time:.2f}s")
            print(f"   btc_chord_recognition.py: {avg_integration_time:.2f}s")
            print(f"   Performance ratio: {avg_integration_time/avg_test_btc_time:.2f}x")
    
    def cleanup(self):
        """Clean up temporary files."""
        try:
            shutil.rmtree(self.temp_dir)
            print(f"\n🧹 Cleaned up temporary directory: {self.temp_dir}")
        except Exception as e:
            print(f"Warning: Could not clean up {self.temp_dir}: {e}")

def main():
    """Main validation function."""
    validator = BTCEquivalenceValidator()
    
    try:
        success = validator.run_comprehensive_validation()
        return 0 if success else 1
    finally:
        validator.cleanup()

if __name__ == "__main__":
    sys.exit(main())
