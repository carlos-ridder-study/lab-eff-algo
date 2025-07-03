#include <vector>
#include <iostream>
#include <sstream>
#include <iomanip>

using namespace std;

/**
 * Calculates the binomial coefficient n choose k.
 */
long long combinations(int n, int k){
    if (k < 0 || k > n) {
        return 0;
    }
    if(k == 0 || k == n) {
        return 1;
    }
    if (k > n / 2) {
        k = n - k; 
    }
    long long res = 1;
    for (int i = 1; i <= k; ++i) {
        res = res * (n - i + 1) / i;
    }
    return res;
}


/**
 * Computes the ratio of won wars for player A with fixed given strategy against all possible random
 * strategies of player B with b soldiers on k battlefields.
 */
float compute_win_percentage(vector<int>& strategy, int b, int k) {

    // dp[j][d] = number of ways to distribute j soldiers of B for a score diff of d
    int d_offset = k;
    std::vector<std::vector<int>> dp_prev( b + 1, std::vector<int>(2 * k + 1, 0));
    dp_prev[0][d_offset] = 1; // Base case: 0 soldiers, 0 battlefields, score diff 0

    // 3. Run the DP calculation for each battlefield
    for (int i = 1; i <= k; ++i) {
        int a_i = strategy[i - 1];
        std::vector<std::vector<int>> dp_curr(b + 1, std::vector<int>(2 * k + 1, 0));

        // Create prefix sums for the previous DP state to speed up range sums
        std::vector<std::vector<int>> prefix_sums(b + 1, std::vector<int>(2 * k + 1, 0));
        for (int d = -i+1; d <= i-1; ++d) {
            int d_id = d + d_offset;
            prefix_sums[0][d_id] = dp_prev[0][d_id];
            for (int j = 1; j <= b; ++j) {
                prefix_sums[j][d_id] = prefix_sums[j - 1][d_id] + dp_prev[j][d_id];
            }
        }
        /*
        cout << "Prefix sums for battlefield " << i << ":" << endl;
        for (int j = 0; j <= b; ++j) {
            for (int d = -i+1; d <= i-1; ++d) {
                int d_id = d + d_offset;
                cout << prefix_sums[j][d_id] << " ";
            }
            cout << endl;
        }
        */

        // Calculate the current DP state using the prefix sums
        for (int j = 0; j <= b; ++j) {
            // Score difference d can range from -i to i
            for (int d = -i; d <= i; ++d) {
                int d_id = d + d_offset;

                // Case 1: A wins battlefield i (B sends x < a_i soldiers)
                //         dp_curr[d_id][j] += sum_{x < a_i} dp_prev[d_id - 1][j - x] 
                //          = prefix_sums[d_id - 1][j] - prefix_sums[d_id - 1][j - a_i];
                if (d_id - 1 >= 0) {
                    int lower_bound = j - a_i + 1;
                    int sum = prefix_sums[j][d_id - 1];
                    // cout << "Calculating for d_id: " << d_id << ", j: " << j << ", prefix_sums[" << j << "][" << d_id - 1 << "] = " << prefix_sums[j][d_id - 1] << endl;
                    if (lower_bound > 0) {
                        sum -= prefix_sums[lower_bound - 1][d_id - 1];
                    }
                    dp_curr[j][d_id] += sum;
                }

                // Case 2: Tie on battlefield i (B sends x = a_i soldiers)
                //         dp_curr[d_id][j] += dp_prev[d_id][j - a_i];
                if (j >= a_i) {
                    dp_curr[j][d_id] += dp_prev[j - a_i][d_id];
                }

                // Case 3: B wins battlefield i (B sends x > a_i soldiers)
                //         dp_curr[d_id + 1][j] += sum_{a_i < x <= j} dp_prev[d_id + 1][j - x]
                if (d_id + 1 <= 2 * k) {
                    int upper_bound = j - a_i - 1;
                    if (upper_bound >= 0) {
                        dp_curr[j][d_id] += prefix_sums[upper_bound][d_id + 1];
                    }
                }
            }
        }
        /*
        cout << "DP state after battlefield " << i << ": " << endl;
        for (int j = 0; j <= b; ++j) {
            for (int d = -i; d <= i; ++d) {
                int d_id = d + d_offset;
                cout << dp_curr[j][d_id] << " ";
            }
            cout << endl;
        }
        */
        dp_prev = dp_curr; // Current state becomes previous for the next iteration
    }

    // Sum up all outcomes where A wins the war (score diff > 0)
    int a_wins_count = 0;
    for (int d = 1; d <= k; ++d) {
        a_wins_count += dp_prev[b][d_offset + d];
    }
    // cout << "A wins count: " << a_wins_count << endl;
    long double total_b_strategies = combinations(b + k -1, k - 1);
    // cout << "Total B strategies: C(" << b + k - 1<< "," << k-1 << ") = " << total_b_strategies << endl;
    
    float result = 0.0;
    if (total_b_strategies > 0) {
        result = a_wins_count / total_b_strategies;
    }
    return result;
}

int main(){
    string line;
    getline(cin, line);
    istringstream iss(line);
    int a, b, k;
    iss >> a >> b >> k;

    // even distribution is optimal deterministic strategy
    std::vector<int> a_strategy(k);
    int q = a / k;
    int r = a % k;
    for (int i = 0; i < k; ++i) {
        if (i < r) {
            a_strategy[i] = q + 1;
        } else {
            a_strategy[i] = q;
        }
    }

    float result = compute_win_percentage(a_strategy, b, k);

    // output the result rounded to 3 decimal places
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3) << result;
    std::string output = oss.str();

    // Remove trailing zeros and possibly the decimal point
    output.erase(output.find_last_not_of('0') + 1, std::string::npos);
    if (!output.empty() && output.back() == '.') output.pop_back();
    std::cout << output << std::endl;
    return 0;
}