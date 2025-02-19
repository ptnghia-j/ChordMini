"""
Programming Problem:
--------------------
To address enharmonic ambiguity, especially during key modulations, implement dynamic programming (DP),
inspired by the Viterbi algorithm. This post-processing step optimizes chord label accuracy by integrating
local key awareness and applying harmonic rules.

The DP formulation:
    - Let V(t)[j] be the maximum cumulative score ending at chord label j at time t.
    - At each time step t, update the state score:
          V*(j) = max_{i} { V(t-1)[i] + transition(i, j) } + emission(t, j)
      where 'transition(i, j)' accounts for harmonic compatibility and 'emission(t, j)' is the score for chord j at time t.
    - Use backtracking to recover the optimal sequence of corrected chord labels.

This approach efficiently resolves enharmonic ambiguities by ensuring the most harmonically consistent path is chosen.
"""

class Solution:
    def enharmonic_resolution(self, scores):
        """
        Performs dynamic programming on a n x 3 grid (n time steps, 3 enharmonic states: -1, 0, +1).
        From each state, allowed transitions are from the previous row's same column, or one column left/right.
        
        Args:
            scores (List[List[float]]): A 2D list with dimensions n x 3, where scores[t][j] is the score at time t and state j.
            
        Returns:
            List[int]: The optimal sequence of state indices (in range 0 to 2) representing the corrected chord labels.
        """
        if not scores:
            return []
        T = len(scores)
        N = len(scores[0])  # Expected to be 3.
        # Initialize DP and backpointer tables.
        dp = [scores[0][:]]  # For t=0, dp equals the emission scores.
        backptr = [[-1]*N for _ in range(T)]
        
        # DP recursion for t >= 1.
        for t in range(1, T):
            dp.append([0.0] * N)
            for j in range(N):
                best_prev = float("-inf")
                best_prev_idx = -1
                # Allowed moves: from previous state at column (j-1), (j), or (j+1) if within bounds.
                for k in (j-1, j, j+1):
                    if 0 <= k < N and dp[t-1][k] > best_prev:
                        best_prev = dp[t-1][k]
                        best_prev_idx = k
                dp[t][j] = best_prev + scores[t][j]
                backptr[t][j] = best_prev_idx
        
        # Backtracking: select the best final state.
        best_final_idx = max(range(N), key=lambda j: dp[T-1][j])
        sequence = [0] * T
        sequence[T-1] = best_final_idx
        for t in range(T-1, 0, -1):
            sequence[t-1] = backptr[t][sequence[t]]
        return sequence

if __name__ == "__main__":
    import unittest

    class TestSolution(unittest.TestCase):
        def setUp(self):
            self.sol = Solution()

        def test_empty_scores(self):
            scores = []
            self.assertEqual(self.sol.enharmonic_resolution(scores), [])

        def test_single_time_step(self):
            # Only one time step; use a 1x3 grid.
            scores = [[0, 10, 5]]
            # Best index on time 0 is 1 (10 is highest).
            self.assertEqual(self.sol.enharmonic_resolution(scores), [1])

        def test_multiple_time_steps(self):
            # Time 0: [1, 2, 5] -> best: index 2 (score=5)
            # Time 1: [3, 1, 0]:
            #    For col0: allowed from indexes 0 and 1 of time 0: max = 2 (from index1) -> 2+3 = 5.
            #    For col1: allowed from indexes 0,1,2: max = 5 (from index2) -> 5+1 = 6.
            #    For col2: allowed from indexes 1,2: max = 5 (from index2) -> 5+0 = 5.
            # Time 2: [0, 0, 1]:
            #    For col0: allowed from indexes (0,1) of time 1: max = 6 (from col1) -> 6+0 = 6.
            #    For col1: allowed from indexes (0,1,2): max = 6 (from col1) -> 6+0 = 6.
            #    For col2: allowed from indexes (1,2): max = 6 (from col1) -> 6+1 = 7.
            # Backtracking yields expected sequence: [2, 1, 2]
            scores = [
                [1, 2, 5],
                [3, 1, 0],
                [0, 0, 1]
            ]
            self.assertEqual(self.sol.enharmonic_resolution(scores), [2, 1, 2])

        def test_complex_dp(self):
            # Construct a more complex 5x3 grid where a fixed greedy approach would fail.
            # The grid is defined as follows:
            # Time 0: [1, 3, 2]           -> best: index 1 (score=3)
            # Time 1: [2, 1, 3]           -> transitions:
            #         For col0: from time0 allowed (col0, col1): max = 3 (index1) -> 3+2=5.
            #         For col1: from time0 allowed (col0, col1, col2): max = 3 (index1) -> 3+1=4.
            #         For col2: from time0 allowed (col1, col2): max = 3 (index1) -> 3+3=6.
            # Time 2: [1, 2, 1]           ->
            #         For col0: allowed from time1 (col0, col1): max = 5 (index0) -> 5+1=6.
            #         For col1: allowed (col0, col1, col2): max = 6 (index2) -> 6+2=8.
            #         For col2: allowed (col1, col2): max = 6 (index2) -> 6+1=7.
            # Time 3: [5, 1, 0]           ->
            #         For col0: allowed from time2 (col0, col1): max = 8 (index1) -> 8+5=13.
            #         For col1: allowed (col0, col1, col2): max = 8 (index1) -> 8+1=9.
            #         For col2: allowed (col1, col2): max = 8 (index1) -> 8+0=8.
            # Time 4: [0, 0, 6]           ->
            #         For col0: allowed from time3 (col0, col1): max = 13 (index0) -> 13+0=13.
            #         For col1: allowed (col0, col1, col2): max = 13 (index0) -> 13+0=13.
            #         For col2: allowed (col1, col2): max = 9 (index1) -> 9+6=15.
            # Backtracking yields expected sequence: [1, 2, 1, 1, 2]
            scores = [
                [1, 3, 2],
                [2, 1, 3],
                [1, 2, 1],
                [5, 1, 0],
                [0, 0, 6]
            ]
            self.assertEqual(self.sol.enharmonic_resolution(scores), [1, 2, 1, 1, 2])

    unittest.main()
